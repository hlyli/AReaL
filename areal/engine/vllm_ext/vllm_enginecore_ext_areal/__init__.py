from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.output_processor import RequestState
from vllm.v1.metrics.stats import LoRARequestStates
from vllm.v1.request import RequestStatus


# engine core related hook functions
def abort_all_reqs(self):
    """Abort all running and waiting requests and clean up resources."""
    scheduler = self.scheduler
    abort_lists = list(scheduler.running) + list(scheduler.waiting)

    if not abort_lists:
        # No requests to abort
        success = scheduler.reset_prefix_cache()
        if not success:
            raise RuntimeError(
                f"Prefix cache must be reset to prevent kv cache pollution! Reset: {success}"
            )
        return

    client_outputs = {}
    for req in abort_lists:
        engine_output = EngineCoreOutput(
            request_id=req.request_id,
            new_token_ids=[],
            finish_reason=FinishReason.ABORT,
            new_logprobs=None,
            new_prompt_logprobs_tensors=None,
            stop_reason=None,
        )
        if req.client_index not in client_outputs:
            client_outputs[req.client_index] = []
        client_outputs[req.client_index].append(engine_output)

    request_ids = [req.request_id for req in abort_lists]
    scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    for client_index, outputs in client_outputs.items():
        engine_core_outputs = EngineCoreOutputs(outputs=outputs)
        self.output_queue.put_nowait((client_index, engine_core_outputs))

    success = scheduler.reset_prefix_cache()
    if not success:
        raise RuntimeError(
            f"Prefix cache must be reset to prevent kv cache pollution! Reset: {success}"
        )


def areal_injected_update_weight(self, path):
    self.abort_all_reqs()
    return self.collective_rpc("update_weights", args=(path,))


def areal_injected_update_weight_lora(
    self, lora_model_path, lora_name, lora_int_id, base_model_name
):
    self.abort_all_reqs()
    return self.collective_rpc(
        "update_weights_lora",
        args=(
            lora_model_path,
            lora_name,
            lora_int_id,
            base_model_name,
        ),
    )


def areal_injected_update_weight_xccl(self):
    self.abort_all_reqs()
    return self.collective_rpc("update_weight_xccl")


def areal_injected_update_weight_lora_xccl(self):
    self.abort_all_reqs()
    return self.collective_rpc("update_weight_lora_xccl")


def finish_request(self, req_state: "RequestState"):
    if req_state.lora_name is None:
        return
    lora_stats = self.lora_name_to_stats[req_state.lora_name]
    # Simply added this if-condition
    if req_state.request_id in lora_stats.running_requests:
        lora_stats.running_requests.remove(req_state.request_id)


def register():
    EngineCore.abort_all_reqs = abort_all_reqs
    EngineCore.areal_injected_update_weight = areal_injected_update_weight
    EngineCore.areal_injected_update_weight_lora = areal_injected_update_weight_lora
    EngineCore.areal_injected_update_weight_xccl = areal_injected_update_weight_xccl
    EngineCore.areal_injected_update_weight_lora_xccl = (
        areal_injected_update_weight_lora_xccl
    )
    from areal.utils import pkg_version

    if not pkg_version.is_version_greater_or_equal("vllm", "0.12.0"):
        setattr(
            LoRARequestStates,
            "finish_request",
            finish_request,
        )
