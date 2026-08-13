"""Tests for shared runtime helpers."""

from __future__ import annotations

import psutil
import pytest

from add.utils import get_physical_cores


def test_get_physical_cores_respects_pbs_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker count stays within the CPUs allocated by PBS."""
    monkeypatch.setenv("PBS_NCPUS", "18")
    monkeypatch.setattr(psutil, "cpu_count", lambda *, logical: 128)

    assert get_physical_cores() == 17
