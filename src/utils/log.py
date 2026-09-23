"""Logging and timing utilities for experiment scripts."""

import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime


def timestamp() -> str:
    """
    The current local time, in the same format as python's logging lines.

    Returns a string like '2026-09-23 09:59:42'.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str = "") -> None:
    """Print with immediate flush to ensure output is visible during long runs."""
    print(message, flush=True)


def log_header(title: str) -> None:
    """Print a clearly delimited section header, stamped with the time it began."""
    log()
    log("=" * 60)
    log(f"  {title}  [{timestamp()}]")
    log("=" * 60)


def log_step(step: int, total: int, message: str) -> None:
    """Print a numbered progress step."""
    log(f"[{step}/{total}] {message}")


def _fmt_duration(seconds: float) -> str:
    """Format seconds into a human-readable string like '1h 4m 32s'.

    Returns a short string suitable for log output.
    """
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def log_gpu_memory(label: str) -> None:
    """Log gpu memory usage (per device) at a labelled point in a run.

    Reports what torch has allocated/reserved, plus the device-wide free/total
    from the driver (which also accounts for other processes on the gpu), so
    out-of-memory failures can be diagnosed from the job logs.
    """
    # imported here (not at module level) so this logging module stays usable
    # in environments without torch installed
    import torch

    if not torch.cuda.is_available():
        log(f"    [MEMORY] {label}: no cuda device available")
        return

    gib = 1024**3
    for device in range(torch.cuda.device_count()):
        # free/total come from the cuda driver, allocated/reserved from torch
        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        log(
            f"    [MEMORY] {label} (cuda:{device}): "
            f"allocated={allocated / gib:.1f}GiB, "
            f"reserved={reserved / gib:.1f}GiB, "
            f"free={free / gib:.1f}GiB of {total / gib:.1f}GiB total"
        )


@contextmanager
def log_timer(
    label: str,
    enabled: bool = True,
) -> Generator[None, None, None]:
    """Context manager that logs when a code block starts and how long it took.

    Logs lines like:
    ``    [TIMER] generation: started at 2026-09-23 09:59:42``
    ``    [TIMER] generation: 1h 4m 32s (finished at 2026-09-23 11:04:14)``

    `enabled` lets multi-gpu runs log from one process only.
    """
    if enabled:
        log(f"    [TIMER] {label}: started at {timestamp()}")
    start = time.monotonic()
    yield
    if enabled:
        duration = _fmt_duration(time.monotonic() - start)
        log(f"    [TIMER] {label}: {duration} (finished at {timestamp()})")
