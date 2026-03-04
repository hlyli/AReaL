import abc
import os
import subprocess
import sys
import uuid
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import ray
from ray.runtime_env import RuntimeEnv
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api import (
    InferenceEngine,
    LocalInfServerInfo,
    ModelRequest,
    ModelResponse,
    ParamSpec,
    Scheduler,
    WeightUpdateMeta,
    WorkflowLike,
)
from areal.api.cli_args import InferenceEngineConfig, PerfTracerConfig, vLLMConfig
from areal.api.io_struct import (
    HttpGenerationResult,
    HttpRequest,
    WeightUpdateRequests,
    get_versioned_lora_name,
)
from areal.infra import RemoteInfEngine, RolloutController, WorkflowExecutor
from areal.infra.platforms import current_platform
from areal.infra.utils.launcher import TRITON_CACHE_PATH
from areal.infra.utils.proc import kill_process_tree
from areal.infra.utils.ray import (
    create_resource_spec,
    get_placement_group_master_ip_and_port,
)
from areal.utils import logging, perf_tracer, stats_tracker
from areal.utils.network import format_host_for_url

logger = logging.getLogger("vLLMEngine")


def _copy_environ():
    _env = os.environ.copy()
    triton_cache_path = _env.get("TRITON_CACHE_PATH", TRITON_CACHE_PATH)
    _env["TRITON_CACHE_PATH"] = os.path.join(triton_cache_path, str(uuid.uuid4()))

    vllm_cache_path = _env.get("VLLM_CACHE_ROOT")
    if vllm_cache_path:
        _env["VLLM_CACHE_ROOT"] = os.path.join(vllm_cache_path, str(uuid.uuid4()))
    _env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
    return _env


class VLLMLauncher(abc.ABC):
    """
    Base class for launching VLLM instances through VLLMBackend
    """

    @abc.abstractmethod
    def launch_server(self, args: dict) -> Any: ...


class LocalVLLMLauncher(VLLMLauncher):
    """
    Classic launcher that launches a single node instance through POpen
    """

    def launch_server(self, server_args: dict[str, Any]) -> subprocess.Popen:
        """Launch vLLM server subprocess."""
        cmd = vLLMConfig.build_cmd_from_args(server_args)
        _env = _copy_environ()

        logger.info(f"Launching vLLM server with command: {' '.join(cmd)}")
        return subprocess.Popen(
            cmd,
            env=_env,
            stdout=sys.stdout,
            stderr=sys.stdout,
        )


@ray.remote
class RayVLLMNode:
    """
    Per node actor that launches POpen, in multinode, there are multiple of these actors that comprise 1 instance
    """

    def __init__(self):
        self.process = None

    def launch_server(self, server_args: dict[str, Any], headless):
        if headless:
            cmd = vLLMConfig.build_cmd_from_args_headless(server_args)
        else:
            cmd = vLLMConfig.build_cmd_from_args(server_args)

        _env = _copy_environ()

        logger.info(f"Launching vLLM server with command: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd, env=_env, stdout=sys.stdout, stderr=sys.stdout
        )

    def destroy(self):
        logger.info("Received termination, killing vllm server process")
        if self.process and self.process.poll() is None:
            kill_process_tree(self.process.pid, graceful=True)

    def __ray_shutdown__(self):
        self.destroy()


class RayVLLMLauncher(VLLMLauncher):
    """
    Launches multiple RayVLLMNode actors that each perform POpen.
    This class does management of Ray resources and scheduling of VLLM instances that span multinode
    """

    def __init__(self):
        # save actors as strings instead of ref as actor ref is not serializable in ProcessPoolExecutor
        self.actor_names: list[str] = []
        # for dp
        self.dp_ip = ""
        self.dp_port = 0
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _get_gpu(bundle):
        if "NPU" in bundle:
            return "NPU"
        elif "GPU" in bundle:
            return "GPU"

        return ""

    @staticmethod
    def _get_resource_spec_and_n_gpu(bundle):
        cpu = bundle["CPU"]
        # already in bytes since it's from the bundle
        memory = bundle["memory"]

        # cannot be autodetected since this launcher is launched with 0 gpus
        # must read from bundle
        n_gpu = 0
        device = "CPU"
        if device := RayVLLMLauncher._get_gpu(bundle):
            n_gpu = bundle[device]

        return create_resource_spec(device, cpu, n_gpu, memory), n_gpu

    def launch_server(
        self, server_args: dict[str, Any]
    ) -> list[ray.actor.ActorHandle[RayVLLMNode]]:
        pg = ray.util.get_current_placement_group()
        tp_size = server_args["tensor_parallel_size"]
        pp_size = server_args["pipeline_parallel_size"]
        dp_group_world_size = tp_size * pp_size
        dp_offset = 0
        is_head = True

        actors = []

        for i, bundle in enumerate(pg.bundle_specs):
            options, n_gpu = RayVLLMLauncher._get_resource_spec_and_n_gpu(bundle)
            current_args = server_args.copy()
            if is_head:
                self.dp_ip, self.dp_port = get_placement_group_master_ip_and_port(pg, i)
            logger.info(f"Launching actor {i}")

            # remove VISIBLE DEVICE envs as ray head had already set them
            # without removing them, vLLM cannot access devices
            # similarly with ray inherited env vars as they can cause scheduling issues
            _env = _copy_environ()
            new_env = {}
            for k, v in _env.items():
                if "VISIBLE_DEVICE" in k:
                    continue
                if "VISIBLE_CORE" in k:
                    continue
                if "RAY_" in k:
                    continue
                new_env[k] = v

            actor_name = str(uuid.uuid4())
            actor = RayVLLMNode.options(
                **options,
                name=actor_name,
                lifetime="detached",
                runtime_env=RuntimeEnv(env_vars=new_env),
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=i,
                ),
            ).remote()

            # cast to int as the args require integer values
            n_gpu = int(n_gpu)
            dp_local = n_gpu // dp_group_world_size
            current_args["data_parallel_size_local"] = dp_local
            current_args["data_parallel_address"] = self.dp_ip
            current_args["data_parallel_rpc_port"] = self.dp_port
            self.logger.info(f"Launching server for VLLM head {self.dp_ip}")
            if is_head:
                current_args["api_server_count"] = int(bundle["CPU"])
                actor.launch_server.remote(current_args, False)
                is_head = False
            else:
                current_args["data_parallel_start_rank"] = dp_offset
                actor.launch_server.remote(current_args, True)

            dp_offset += dp_local

            self.actor_names.append(actor_name)
            actors.append(actor)

        return actors


class VLLMBackend:
    """vLLM-specific backend implementation for remote inference."""

    def __init__(self, launcher_cls: type[VLLMLauncher] = LocalVLLMLauncher):
        self.launcher = launcher_cls()

    def build_generation_request(
        self, req: ModelRequest, with_lora: bool, version: int
    ) -> HttpRequest:
        """Build vLLM generation request."""
        gconfig = req.gconfig
        stop_token_ids = gconfig.stop_token_ids

        # NOTE: vLLM uses flat payload structure, not nested sampling_params
        payload = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_tokens": gconfig.max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": stop_token_ids,
            "ignore_eos": gconfig.ignore_eos,
            "skip_special_tokens": gconfig.skip_special_tokens,
            "return_tokens_as_token_ids": True,
            "logprobs": 0,
            "use_beam_search": gconfig.use_beam_search,
            "stream": False,
        }

        if with_lora:
            lora_name = gconfig.lora_name
            if not lora_name:
                raise ValueError(
                    "LoRA name (gconfig.lora_name) is required when use_lora is enabled."
                )
            payload["model"] = get_versioned_lora_name(lora_name, version)

        if req.vision_msg_vllm:
            images = iter(req.image_data)
            parsed_input = req.vision_msg_vllm[0]
            for msg in parsed_input:
                if isinstance(msg["content"], list):
                    for content in msg["content"]:
                        if content.get("type") == "image_url":
                            try:
                                base64_img = next(images)
                            except StopIteration:
                                raise ValueError(
                                    "Not enough images in req.image_data to match image_url entries."
                                )
                            content["image_url"] = {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            }
            payload["messages"] = parsed_input.copy()
            payload["logprobs"] = True
            return HttpRequest(endpoint="/v1/chat/completions", payload=payload)
        else:
            payload["prompt"] = req.input_ids.copy()
            return HttpRequest(endpoint="/v1/completions", payload=payload)

    def parse_generation_response(
        self, response: dict[str, Any]
    ) -> HttpGenerationResult:
        """Parse vLLM generation response."""
        meta_info = response["choices"][0]
        stop_reason = meta_info["finish_reason"]

        # Parse tokens from "token:123" format
        if "tokens" in meta_info["logprobs"]:
            output_tokens = meta_info["logprobs"]["tokens"]
            output_tokens = [int(t.split(":")[1]) for t in output_tokens]
            output_logprobs = meta_info["logprobs"]["token_logprobs"]
        elif "content" in meta_info["logprobs"]:
            outputs = meta_info["logprobs"]["content"]
            output_tokens = [int(t["token"].split(":")[1]) for t in outputs]
            output_logprobs = [t["logprob"] for t in outputs]
        else:
            raise ValueError("Unexpected vLLM response format.")

        if stop_reason == "abort" and len(output_tokens) == 0:
            return HttpGenerationResult(
                output_tokens=[],
                output_logprobs=[],
                stop_reason=stop_reason,
            )
        return HttpGenerationResult(
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            stop_reason=stop_reason,
        )

    def build_disk_weight_update_requests(
        self, meta: WeightUpdateMeta
    ) -> WeightUpdateRequests:
        """Build vLLM disk weight update requests."""
        if meta.use_lora:
            if meta.version is None:
                raise ValueError("Version is required for LoRA update.")
            lora_name = get_versioned_lora_name(meta.lora_name, meta.version)
            endpoint = "/v1/load_lora_adapter"
            payload = {
                "lora_path": str(meta.path),
                "lora_name": lora_name,
            }
        else:
            endpoint = "/areal_update_weights"
            payload = {"model_path": str(meta.path)}

        return WeightUpdateRequests(
            requests=[HttpRequest(endpoint=endpoint, payload=payload)]
        )

    def build_distributed_weight_update_requests(
        self,
        meta: WeightUpdateMeta,
        param_specs: list[ParamSpec],
    ) -> WeightUpdateRequests:
        """Build vLLM distributed weight update requests."""
        # vLLM uses two-step process: set metadata, then update
        # vLLM uses two-step process: set metadata, then update
        base_payload = {
            "names": [pspec.name for pspec in param_specs],
            "dtypes": [pspec.dtype for pspec in param_specs],
            "shapes": [pspec.shape for pspec in param_specs],
            "group_name": meta.nccl_group_name,
        }

        if meta.use_lora:
            if meta.version is None:
                raise ValueError("Version is required for LoRA update.")
            lora_name = get_versioned_lora_name(meta.lora_name, meta.version)
            lora_payload = {
                "lora_name": lora_name,
                "lora_int_id": meta.lora_int_id,
                "lora_target_modules": meta.peft_config["target_modules"],
                "lora_rank": meta.peft_config["r"],
                "lora_alpha": meta.peft_config["lora_alpha"],
                "lora_bias": meta.peft_config["bias"],
                "base_model_name": meta.base_model_name,
            }
            payload = {**base_payload, **lora_payload}
            meta_endpoint = "/areal_set_update_weight_meta_lora"
            update_endpoint = "/areal_update_weights_lora_xccl"
        else:
            payload = base_payload
            meta_endpoint = "/areal_set_update_weight_meta"
            update_endpoint = "/areal_update_weights_xccl"

        return WeightUpdateRequests(
            requests=[
                HttpRequest(
                    endpoint=meta_endpoint,
                    payload=payload,
                ),
                HttpRequest(
                    endpoint=update_endpoint,
                    payload={} if not meta.use_lora else payload,
                ),
            ]
        )

    def build_init_weights_group_request(
        self, addr: str, server_idx: int, meta: WeightUpdateMeta
    ) -> HttpRequest:
        """Build vLLM init weights group request."""
        assert meta.gen_allocation is not None
        gen_parallel = meta.gen_allocation.parallel
        rank_offset = 1 + server_idx * gen_parallel.tp_size * gen_parallel.pp_size
        payload = {
            "master_address": format_host_for_url(meta.nccl_master_address),
            "master_port": str(meta.nccl_master_port),
            "rank_offset": rank_offset,
            "world_size": gen_parallel.world_size + 1,
            "backend": meta.backend
            if meta.backend is not None
            else current_platform.communication_backend,
            "group_name": meta.nccl_group_name,
        }
        return HttpRequest(endpoint="/areal_init_weights_update_group", payload=payload)

    def get_pause_request(self) -> HttpRequest:
        """Get vLLM pause request."""
        return HttpRequest(endpoint="/areal_pause_generation", payload={})

    def get_resume_request(self) -> HttpRequest:
        """Get vLLM resume request."""
        return HttpRequest(endpoint="/areal_continue_generation", payload={})

    def get_health_check_request(self) -> HttpRequest:
        """Get vLLM health check request."""
        return HttpRequest(endpoint="/health", payload={}, method="GET")

    def get_offload_request(self) -> HttpRequest:
        """Get vLLM offload request.

        Uses vLLM's /sleep endpoint to offload model memory to CPU.
        Default level is 1.
        """
        return HttpRequest(endpoint="/sleep", payload={}, method="POST")

    def get_onload_request(self, tags: list[str] | None = None) -> HttpRequest:
        """Get vLLM onload request.

        Uses vLLM's /wake_up endpoint to reload model memory from CPU.
        vLLM reads parameters from query string.

        Parameters
        ----------
        tags : list[str], optional
            Tags to wake up specific components. If None, wakes up all components.
        """
        if tags is not None:
            # Build query string with multiple tags parameters
            tags_query = "&".join([f"tags={tag}" for tag in tags])
            endpoint = f"/wake_up?{tags_query}"
        else:
            endpoint = "/wake_up"
        return HttpRequest(endpoint=endpoint, payload={}, method="POST")

    def launch_server(self, server_args: dict[str, Any]) -> subprocess.Popen:
        return self.launcher.launch_server(server_args)


class RemotevLLMEngine(InferenceEngine):
    """vLLM remote inference engine.

    This class delegates all functionality to RemoteInfEngine with
    a VLLMBackend implementation. It maintains the same public API for
    backward compatibility.

    Parameters
    ----------
    config : InferenceEngineConfig
        Configuration for the inference engine
    """

    def __init__(self, config: InferenceEngineConfig):
        self.config = config
        # Pure composition - create internal engine with vLLM backend
        if ray.is_initialized() and any(
            spec.ray_placement_strategy == "deferred" for spec in config.scheduling_spec
        ):
            vllm_launcher_cls = RayVLLMLauncher
        else:
            vllm_launcher_cls = LocalVLLMLauncher
        self._engine = RemoteInfEngine(config, VLLMBackend(vllm_launcher_cls))

    def initialize(
        self,
        engine_id: str | None = None,
        addr: str | list[str] | None = None,
        train_data_parallel_size: int | None = None,
    ):
        """Initialize the engine by discovering and connecting to servers."""
        return self._engine.initialize(engine_id, addr, train_data_parallel_size)

    def destroy(self):
        """Destroy the engine and clean up resources."""
        return self._engine.destroy()

    @property
    def initialized(self) -> bool:
        return self._engine.initialized

    @property
    def workflow_executor(self) -> WorkflowExecutor:
        """Get the workflow executor of the inference engine."""
        return self._engine.workflow_executor

    def set_version(self, version: int):
        """Set the current weight version."""
        return self._engine.set_version(version)

    def get_version(self) -> int:
        """Get the current weight version."""
        return self._engine.get_version()

    def set_proxy_gateway_addr(self, addr: str) -> None:
        self._engine.set_proxy_gateway_addr(addr)

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Asynchronously generate a response for the given request."""
        return await self._engine.agenerate(req)

    def init_weights_update_group(
        self, meta: WeightUpdateMeta, xccl_group_ranks: list[int] | None = None
    ) -> Future[None]:
        """Initialize the weight update process group."""
        return self._engine.init_weights_update_group(
            meta, xccl_group_ranks=xccl_group_ranks
        )

    def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ) -> Future[None]:
        """Update weights from distributed memory."""
        return self._engine.update_weights_from_distributed(meta, param_specs)

    def update_weights_from_disk(self, meta: WeightUpdateMeta) -> Future[None]:
        """Update weights from disk."""
        return self._engine.update_weights_from_disk(meta)

    def submit(
        self,
        data: dict[str, Any],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        task_id: int | None = None,
        callback_addr: str | None = None,
        is_eval: bool = False,
        proxy_addr: str | None = None,
    ) -> int:
        """Submit a request to the inference engine."""
        return self._engine.submit(
            data=data,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            group_size=group_size,
            task_id=task_id,
            callback_addr=callback_addr,
            is_eval=is_eval,
            proxy_addr=proxy_addr,
        )

    def wait(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> list[dict[str, Any] | None]:
        """Wait for a specified number of requests to complete."""
        return self._engine.wait(count, timeout, raise_timeout)

    def wait_for_task(
        self, task_id: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> dict[str, Any] | None:
        """Wait for a specific task to complete by task_id."""
        return self._engine.wait_for_task(task_id, timeout, raise_timeout)

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
    ) -> dict[str, Any]:
        """Submit a batch of requests and wait for results.

        This method does not support asynchronous rollout and should be used for offline
        data collection or debugging, not in production experiments.
        """
        return self._engine.rollout_batch(
            data=data,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            group_size=group_size,
        )

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
    ):
        """Asynchronously submit and wait until a full batch is ready."""
        return self._engine.prepare_batch(
            dataloader=dataloader,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            group_size=group_size,
            dynamic_bs=dynamic_bs,
        )

    def pause(self):
        return self._engine.pause()

    def resume(self):
        return self._engine.resume()

    def pause_generation(self):
        return self._engine.pause_generation()

    def continue_generation(self):
        return self._engine.continue_generation()

    def launch_server(self, server_args: dict[str, Any]) -> LocalInfServerInfo:
        return self._engine.launch_server(server_args)

    def teardown_server(self):
        return self._engine.teardown_server()

    def offload(self):
        return self._engine.offload()

    def onload(self, tags: list[str] | None = None):
        return self._engine.onload(tags=tags)

    def export_stats(self) -> dict[str, float]:
        return stats_tracker.export_all(reduce_group=None)

    @classmethod
    def as_controller(
        cls, config: InferenceEngineConfig, scheduler: Scheduler
    ) -> RolloutController:
        return RolloutController(cls, config=config, scheduler=scheduler)

    def clear_batches(self, *args):
        """Placeholder method of single-controller API."""

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        perf_tracer.save(step=step, force=force)

    def config_perf_tracer(
        self, config: PerfTracerConfig, rank: int, role: str
    ) -> None:
        if perf_tracer.is_configured():
            return
        perf_tracer.configure(config, rank=rank, role=role)
