"""Strict report configuration and reproducible artifact publication."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from add.data import file_identity


def read_report_config(
    path: str | Path,
    *,
    allowed_keys: Sequence[str],
) -> tuple[dict[str, object], Path]:
    """Read version-one YAML and resolve the same project root as add config."""
    resolved = Path(path).expanduser().resolve()
    with resolved.open() as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict) or values.get("schema_version") != 1:
        raise ValueError(
            f"Report config requires schema_version: 1: {resolved}"
        )
    unknown = set(values) - set(allowed_keys) - {"schema_version"}
    if unknown:
        raise ValueError(
            f"Unknown report configuration keys: {sorted(unknown)}"
        )
    return values, resolved.parent.parent


def configured_path(
    values: Mapping[str, object],
    key: str,
    *,
    root: Path,
) -> Path:
    """Resolve a required nonempty configured path without existence
    fallback.
    """
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration {key} must be a nonempty path")
    path = Path(value).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def input_identity(path: str | Path) -> dict[str, object]:
    """Identify a file with SHA256 in addition to its size and modification
    time.
    """
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return file_identity(resolved) | {"sha256": digest.hexdigest()}


def prepare_report_directory(path: str | Path) -> Path:
    """Refuse to overwrite an existing report or mistake partial output for
    completion.
    """
    destination = Path(path).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Report directory is not empty: {destination}; "
            "choose a new output directory"
        )
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def publish_report(
    destination: Path,
    *,
    tables: Mapping[str, pd.DataFrame],
    provenance: Mapping[str, object],
) -> None:
    """Write report tables, then atomically publish completion provenance."""
    for name, table in tables.items():
        path = destination / name
        if path.suffix == ".parquet":
            table.to_parquet(path, index=False)
        else:
            table.to_csv(path, index=False)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scikit-learn", "anndata")
    }
    record = dict(provenance) | {
        "schema_version": 1,
        "status": "complete",
        "versions": versions,
        "artifacts": {
            str(path.relative_to(destination)): input_identity(path)
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        },
    }
    temporary = destination / "provenance.json.tmp"
    temporary.write_text(
        json.dumps(record, indent=2, default=_json_value) + "\n"
    )
    temporary.replace(destination / "provenance.json")


def _json_value(value: object) -> object:
    """Serialize scientific scalar values and explicit paths."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported provenance value: {type(value).__name__}")
