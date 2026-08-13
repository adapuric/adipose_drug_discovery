"""Small shared helpers."""

import psutil


def get_physical_cores() -> int:
    """Return physical cores minus one for the parent process."""
    cores = psutil.cpu_count(logical=False)
    return 1 if cores is None or cores <= 1 else cores - 1
