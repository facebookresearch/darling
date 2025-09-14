import os
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


vllm_package_name = "vllm"
vllm_package_version = get_version(vllm_package_name)
if vllm_package_version is None:
    raise PackageNotFoundError("To use vllm rollout, please ensure the 'vllm' package is properly installed. See https://verl.readthedocs.io/en/latest/start/install.html for more details")

###
# package_version = get_version(package_name)
# [SUPPORT AMD:]
# Do not call any torch.cuda* API here, or ray actor creation import class will fail.
if "ROCM_PATH" in os.environ:
    import re

    match = re.match(r"(\d+\.\d+\.?\d*)", vllm_package_version)
    if match:
        vllm_package_version = match.group(1)
    else:
        raise ValueError(f"Warning: Could not parse version format: {vllm_package_version}")
###

if Version(vllm_package_version) <= Version("0.6.3"):
    vllm_mode = "customized"
    from .fire_vllm_rollout import FIREvLLMRollout  # noqa: F401
    from .vllm_rollout import vLLMRollout  # noqa: F401
else:
    vllm_mode = "spmd"
    from .vllm_rollout_spmd import vLLMAsyncRollout, vLLMRollout  # noqa: F401
