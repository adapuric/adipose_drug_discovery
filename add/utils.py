"""Small shared helpers."""

import os

import psutil


def get_physical_cores() -> int:
    """Return usable cores minus one for the parent process."""
    cores = psutil.cpu_count(logical=False)
    allocated_cores_value = os.environ.get("NCPUS") or os.environ.get(
        "PBS_NCPUS"
    )
    if allocated_cores_value is not None:
        allocated_cores = int(allocated_cores_value)
        cores = (
            allocated_cores if cores is None else min(cores, allocated_cores)
        )
    return 1 if cores is None or cores <= 1 else cores - 1
