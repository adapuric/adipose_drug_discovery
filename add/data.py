"""Configuration, input validation, and provenance helpers."""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path

import anndata as ad  # type: ignore[import]
import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True, kw_only=True)
class AnalysisConfig:
    """Resolved core settings plus source-specific configuration sections."""

    config_path: Path
    project_root: Path
    adipose_path: Path
    output_path: Path
    count_layer: str
    condition_col: str
    donor_col: str
    state_col: str
    baseline_label: str
    weightloss_label: str
    lean_label: str
    unassigned_label: str
    min_cells: int
    random_seed: int
    metadata_cols: tuple[str, ...]
    tahoe: Mapping[str, object]
    lincs: Mapping[str, object]
    gene_selection: Mapping[str, object]
    pca_ridge: Mapping[str, object]

    def resolve_path(self, value: object, *, name: str) -> Path:
        """Resolve a configured path relative to the repository root."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Configuration value {name!r} must be a non-empty path."
            )
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()


def load_config(path: str | Path = "config/baselines.yaml") -> AnalysisConfig:
    """Load and validate the compact YAML analysis configuration.

    Relative paths are interpreted from the parent of the directory containing
    the configuration file. For the repository `config/baselines.yaml`, this is
    the repository root. When the packaged default is used outside a checkout,
    paths are interpreted from the current working directory.

    Args:
      path: YAML configuration file.

    Returns:
      Validated configuration with resolved core paths.
    """
    requested_path = Path(path).expanduser()
    config_path = requested_path.resolve()
    using_installed_default = False
    if not config_path.is_file() and requested_path == Path(
        "config/baselines.yaml"
    ):
        installed_default = (
            Path(sys.prefix)
            / "share"
            / "adipose_drug_discovery"
            / "config"
            / "baselines.yaml"
        )
        if installed_default.is_file():
            config_path = installed_default.resolve()
            using_installed_default = True
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. Pass --config."
        )

    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")

    project_root = (
        Path.cwd().resolve()
        if using_installed_default
        else config_path.parent.parent
    )
    required_strings = (
        "adipose_path",
        "output_path",
        "count_layer",
        "condition_col",
        "donor_col",
        "state_col",
        "baseline_label",
        "weightloss_label",
        "lean_label",
        "unassigned_label",
    )
    values: dict[str, str] = {}
    for key in required_strings:
        value = loaded.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Configuration value {key!r} must be a string.")
        values[key] = value

    min_cells = loaded.get("min_cells")
    random_seed = loaded.get("random_seed")
    if not isinstance(min_cells, int) or isinstance(min_cells, bool):
        raise ValueError("Configuration value 'min_cells' must be an integer.")
    if min_cells < 1:
        raise ValueError("Configuration value 'min_cells' must be at least 1.")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError(
            "Configuration value 'random_seed' must be an integer."
        )

    metadata_cols = loaded.get("metadata_cols", [])
    if not isinstance(metadata_cols, list) or not all(
        isinstance(column, str) and column for column in metadata_cols
    ):
        raise ValueError(
            "Configuration 'metadata_cols' must be a list of names."
        )

    sections = {
        name: _mapping_section(loaded, name)
        for name in ("tahoe", "lincs", "gene_selection", "pca_ridge")
    }

    def _resolve_core_path(value: str) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
        return candidate.resolve()

    return AnalysisConfig(
        config_path=config_path,
        project_root=project_root,
        adipose_path=_resolve_core_path(values["adipose_path"]),
        output_path=_resolve_core_path(values["output_path"]),
        count_layer=values["count_layer"],
        condition_col=values["condition_col"],
        donor_col=values["donor_col"],
        state_col=values["state_col"],
        baseline_label=values["baseline_label"],
        weightloss_label=values["weightloss_label"],
        lean_label=values["lean_label"],
        unassigned_label=values["unassigned_label"],
        min_cells=min_cells,
        random_seed=random_seed,
        metadata_cols=tuple(metadata_cols),
        tahoe=sections["tahoe"],
        lincs=sections["lincs"],
        gene_selection=sections["gene_selection"],
        pca_ridge=sections["pca_ridge"],
    )


def load_adipose(
    path: str | Path,
    *,
    count_layer: str,
    required_obs: tuple[str, ...],
    backed: bool = True,
) -> ad.AnnData:
    """Open adipose AnnData and validate identifiers and the count source.

    The configured count layer is mandatory. This function never substitutes
    normalized `X` when raw counts are absent.
    """
    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Adipose AnnData not found: {input_path}. Update adipose_path."
        )
    adata = ad.read_h5ad(input_path, backed="r" if backed else None)

    if missing_obs := [
        column for column in required_obs if column not in adata.obs
    ]:
        if adata.isbacked:
            adata.file.close()
        raise KeyError(f"Adipose AnnData is missing obs columns: {missing_obs}")
    if count_layer not in adata.layers:
        if adata.isbacked:
            adata.file.close()
        raise KeyError(
            f"Configured raw-count layer {count_layer!r} is absent from "
            f"{input_path}. No expression fallback was used."
        )
    if not adata.var_names.is_unique:
        if adata.isbacked:
            adata.file.close()
        raise ValueError(
            "Adipose gene identifiers in var_names must be unique."
        )
    return adata


def load_rescue_table(path: str | Path) -> pd.DataFrame:
    """Load a combined rescue table with its scoring contract validated."""
    rescue_path = Path(path).expanduser().resolve()
    if not rescue_path.is_file():
        raise FileNotFoundError(f"Rescue table not found: {rescue_path}")
    rescue = pd.read_csv(rescue_path)
    required = {
        "gene",
        "state",
        "logFC",
        "moderated_t",
        "p_value",
        "adjusted_p_value",
        "tested",
        "n_pairs",
    }
    if missing := sorted(required.difference(rescue.columns)):
        raise ValueError(f"Rescue table is missing columns: {missing}")
    if rescue.duplicated(["state", "gene"]).any():
        raise ValueError("Rescue table must have one row per state and gene.")
    return rescue


def file_identity(path: str | Path) -> dict[str, object]:
    """Return a lightweight identity record for an input artifact."""
    resolved = Path(path).expanduser().resolve()
    record: dict[str, object] = {
        "path": str(resolved),
        "exists": resolved.exists(),
    }
    if resolved.exists():
        stat = resolved.stat()
        record |= {
            "size_bytes": stat.st_size,
            "modified_utc": datetime.fromtimestamp(
                stat.st_mtime,
                tz=UTC,
            ).isoformat(),
        }
    return record


def write_provenance(
    path: str | Path,
    *,
    command: str,
    config: AnalysisConfig,
    inputs: Mapping[str, object],
    parameters: Mapping[str, object],
) -> Path:
    """Write deterministic, human-readable provenance for an analysis stage."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "command": command,
        "config_path": str(config.config_path),
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "inputs": dict(inputs),
        "parameters": dict(parameters),
        "random_seed": config.random_seed,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            record, handle, indent=2, default=_json_default, sort_keys=True
        )
        handle.write("\n")
    return output_path


def _mapping_section(
    config: Mapping[str, object],
    name: str,
) -> dict[str, object]:
    """Return a copied mapping section or raise an actionable error."""
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section {name!r} must be a mapping.")
    return dict(value)


def _json_default(value: object) -> object:
    """Convert common scientific values for provenance JSON output."""
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")
