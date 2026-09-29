"""vLLM engine settings shared by training rollouts and evaluation."""

from typing import Any


# passed to every vllm engine as `additional_config`, so training and eval generate the same way
VLLM_ADDITIONAL_CONFIG: dict[str, Any] = {
    # the kernel for the linear-attention (gated deltanet) layers of models like qwen3.5.
    # on hopper gpus (isambard's gh200s) vllm defaults to flashinfer's kernel, which is
    # compiled on first use - the nodes have no cuda compiler, so use vllm's triton kernel
    # instead. models without these layers ignore it.
    "gdn_prefill_backend": "triton",
}
