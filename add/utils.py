"""Small shared helpers."""

import os

import psutil


def get_physical_cores() -> int:
    """Return usable cores minus one for the parent process."""
    cores = psutil.cpu_count(logical=False)
    if "PBS_NCPUS" in os.environ:
        allocated_cores = int(os.environ["PBS_NCPUS"])
        cores = (
            allocated_cores if cores is None else min(cores, allocated_cores)
        )
    return 1 if cores is None or cores <= 1 else cores - 1
