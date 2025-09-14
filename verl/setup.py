import os
from pathlib import Path

from setuptools import find_packages, setup

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "verl/version/version")) as f:
    __version__ = f.read().strip()

install_requires = [
    "accelerate",
    "codetiming",
    "datasets",
    "dill",
    "hydra-core",
    "numpy<2.0.0",
    "pandas",
    "peft",
    "pyarrow>=19.0.0",
    "pybind11",
    "pylatexenc",
    "ray[default]>=2.41.0",
    "torchdata",
    "tensordict>=0.8.0,<=0.9.1,!=0.9.0",
    "transformers",
    "wandb",
    "packaging>=20.0",
    "tensorboard",
]

TEST_REQUIRES = ["pytest", "pre-commit", "py-spy", "pytest-asyncio"]
PRIME_REQUIRES = ["pyext"]
GEO_REQUIRES = ["mathruler", "torchvision", "qwen_vl_utils"]
GPU_REQUIRES = ["liger-kernel", "flash-attn"]
MATH_REQUIRES = ["math-verify"]  # Add math-verify as an optional dependency
VLLM_REQUIRES = ["tensordict>=0.8.0,<=0.9.1,!=0.9.0", "vllm>=0.7.3,<=0.9.1"]
SGLANG_REQUIRES = [
    "tensordict>=0.8.0,<=0.9.1,!=0.9.0",
    "sglang[srt,openai]==0.4.10.post2",
    "torch==2.7.1",
]
TRL_REQUIRES = ["trl<=0.9.6"]
MCORE_REQUIRES = ["mbridge"]

extras_require = {
    "test": TEST_REQUIRES,
    "prime": PRIME_REQUIRES,
    "geo": GEO_REQUIRES,
    "gpu": GPU_REQUIRES,
    "math": MATH_REQUIRES,
    "vllm": VLLM_REQUIRES,
    "sglang": SGLANG_REQUIRES,
    "trl": TRL_REQUIRES,
    "mcore": MCORE_REQUIRES,
}


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="verl",
    version=__version__,
    package_dir={"": "."},
    packages=find_packages(where="."),
    url="https://github.com/volcengine/verl",
    license="Apache 2.0",
    author="Bytedance - Seed - MLSys",
    author_email="zhangchi.usc1992@bytedance.com, gmsheng@connect.hku.hk",
    description="verl: Volcano Engine Reinforcement Learning for LLM",
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={
        "": ["version/*"],
        "verl": ["trainer/config/*.yaml"],
    },
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
