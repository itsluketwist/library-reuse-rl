#!/usr/bin/env python3
"""Check the locked GPU stack (the pyproject [hpc] extra) actually works on this node.

Run inside a GPU allocation after `uv sync --locked --extra hpc`. Each check prints "ok" or
raises, and they run in order of what is most likely to break on isambard:

    python scripts/check_gpu_stack.py                               # single-gpu checks
    torchrun --nproc-per-node=4 scripts/check_gpu_stack.py --nccl   # multi-gpu comms
"""

import argparse
import os

import torch


# small model, so the checks are quick and fit alongside each other on one gpu
MODEL = "Qwen/Qwen3-0.6B"


def check_cuda() -> None:
    """
    Print the package versions and check CUDA and bf16 are usable.
    """
    import transformers
    import trl
    import vllm

    print(f"torch {torch.__version__} (cuda build {torch.version.cuda})")
    print(
        f"vllm {vllm.__version__}, transformers {transformers.__version__}, trl {trl.__version__}"
    )
    assert torch.cuda.is_available(), "cuda is unavailable - is this a gpu node?"
    print(f"gpu: {torch.cuda.get_device_name(0)} x {torch.cuda.device_count()}")
    assert torch.cuda.is_bf16_supported(), "bf16 is unavailable"
    print("cuda ok")


def check_triton() -> None:
    """
    Compile a small function with torch.compile, which generates triton kernels at runtime.

    This is the check most likely to fail, as the stack is built for cuda 12.9 but isambard's
    driver only supports 12.7 - precompiled kernels are fine, runtime-compiled ones may not be.
    """
    compiled = torch.compile(lambda x: (x.sin() + x.cos()).sum())
    result = compiled(torch.randn(1024, device="cuda"))
    torch.cuda.synchronize()
    assert torch.isfinite(result).item(), "torch.compile gave a non-finite result"
    print("triton ok")


def check_training_step() -> None:
    """
    Run one full-weight bf16 training step (forward, backward, optimizer) with transformers.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-6,
    )

    batch = tokenizer(
        ["def add(a, b):\n    return a + b"],
        return_tensors="pt",
    ).to("cuda")
    loss = model(**batch, labels=batch["input_ids"]).loss
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    assert torch.isfinite(loss).item(), "training step gave a non-finite loss"
    print(f"training step ok (loss {loss.item():.3f})")

    # free the gpu memory before vllm starts, as it reserves a fraction of what is free
    del model, optimizer
    torch.cuda.empty_cache()


def check_vllm() -> None:
    """
    Generate a short completion with vllm, which uses its own precompiled cuda kernels.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.3,
        max_model_len=2048,
    )
    outputs = llm.generate(
        ["Write a python function that adds two numbers."],
        SamplingParams(max_tokens=64),
    )
    text = outputs[0].outputs[0].text
    assert text.strip(), "vllm generated an empty completion"
    print(f"vllm ok: {text.strip()[:80]!r}")


def check_nccl() -> None:
    """
    All-reduce a tensor across every gpu in the job, to check multi-gpu communication.

    Must be launched with torchrun, which starts one process per gpu.
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # each process contributes its rank, so the sum should be 0 + 1 + ... + (world_size - 1)
    tensor = torch.tensor([float(rank)], device="cuda")
    torch.distributed.all_reduce(tensor)
    expected = world_size * (world_size - 1) / 2
    assert tensor.item() == expected, (
        f"all-reduce gave {tensor.item()}, expected {expected}"
    )
    if rank == 0:
        print(f"nccl ok across {world_size} gpus")
    torch.distributed.destroy_process_group()


def main() -> None:
    """
    Run the single-gpu checks, or the multi-gpu check with --nccl.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nccl",
        action="store_true",
        help="run the multi-gpu all-reduce check (launch with torchrun)",
    )
    args = parser.parse_args()

    if args.nccl:
        check_nccl()
        return

    check_cuda()
    check_triton()
    check_training_step()
    check_vllm()
    print("all checks passed")


# vllm starts worker processes, which re-import this file - the guard stops them re-running it
if __name__ == "__main__":
    main()
