from setuptools import find_packages, setup

setup(
    name="vllm-enginecore-ext-areal",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "vllm.general_plugins": [
            "enginecore_ext = vllm_engine_core_ext_areal:register",
        ],
    },
)
