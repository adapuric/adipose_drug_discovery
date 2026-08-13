"""Load and cache external perturbation signatures."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, cast

import anndata as ad  # type: ignore[import]
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp  # type: ignore[import]


_CACHE_FORMAT_VERSION = 1
_CACHE_INDEX_COLUMN = "__signature_index__"


@dataclass
class PerturbSignatures:
    """A perturbation-by-gene response matrix with aligned metadata.

    `delta` and `control` use rows in exactly the order of `meta`. `control`,
    when present, contains the matched starting or vehicle expression profile
    used to construct each response. Gene columns follow `genes` explicitly;
    callers must never infer gene identity from column position alone.

    Attributes:
      delta: Perturbation responses with shape (signatures, genes).
      genes: Unique gene identifiers aligned to matrix columns.
      meta: One uniquely indexed metadata row per signature.
      control: Optional matched control expression with the same shape as
        `delta`.
      provenance: Input and transformation metadata for the signatures.
    """

    delta: np.ndarray
    genes: Sequence[str]
    meta: pd.DataFrame
    control: np.ndarray | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate alignment and normalize matrix containers."""
        try:
            delta = np.asarray(self.delta, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "delta must be a numeric two-dimensional matrix"
            ) from error
        if delta.ndim != 2:
            raise ValueError(
                f"delta must be two-dimensional; received shape {delta.shape}"
            )
        if bool(np.isinf(delta).any()):
            raise ValueError("delta contains infinite values")

        genes = tuple(str(gene) for gene in self.genes)
        if not genes:
            raise ValueError("genes must contain at least one identifier")
        if any(not gene.strip() for gene in genes):
            raise ValueError("genes contains a blank identifier")
        if len(genes) != len(set(genes)):
            raise ValueError("genes must contain unique identifiers")
        if delta.shape[1] != len(genes):
            raise ValueError(
                "delta columns do not match genes: "
                f"{delta.shape[1]} columns versus {len(genes)} genes"
            )

        if not isinstance(self.meta, pd.DataFrame):
            raise TypeError("meta must be a pandas DataFrame")
        meta = self.meta.copy()
        if delta.shape[0] != len(meta):
            raise ValueError(
                "delta rows do not match metadata: "
                f"{delta.shape[0]} rows versus {len(meta)} metadata records"
            )
        if not meta.index.is_unique:
            raise ValueError("meta index must uniquely identify signatures")
        if meta.index.hasnans:
            raise ValueError(
                "meta index contains missing signature identifiers"
            )

        control = None
        if self.control is not None:
            try:
                control = np.asarray(self.control, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise TypeError("control must be a numeric matrix") from error
            if control.shape != delta.shape:
                raise ValueError(
                    "control must have the same shape as delta: "
                    f"{control.shape} versus {delta.shape}"
                )
            if bool(np.isinf(control).any()):
                raise ValueError("control contains infinite values")

        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping")

        self.delta = delta
        self.genes = genes
        self.meta = meta
        self.control = control
        self.provenance = dict(self.provenance)

    @property
    def control_expression(self) -> np.ndarray | None:
        """Return the matched control matrix, when available."""
        return self.control

    @property
    def n_signatures(self) -> int:
        """Return the number of perturbation signatures."""
        return self.delta.shape[0]

    @property
    def n_genes(self) -> int:
        """Return the number of aligned genes."""
        return self.delta.shape[1]


def save_perturb_signatures(
    signatures: PerturbSignatures,
    prefix: str | Path,
) -> dict[str, Path]:
    """Write a matrix NPZ, aligned metadata CSV, and provenance JSON.

    Args:
      signatures: Validated signatures to cache.
      prefix: Shared output prefix. A trailing ".npz" is accepted and removed.

    Returns:
      Paths keyed by "matrix", "metadata", and "provenance".
    """
    paths = _cache_paths(prefix)
    if _CACHE_INDEX_COLUMN in signatures.meta.columns:
        raise ValueError(
            f"metadata column {_CACHE_INDEX_COLUMN!r} is reserved for caching"
        )
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    control = (
        signatures.control
        if signatures.control is not None
        else np.empty((0, 0), dtype=np.float64)
    )
    np.savez_compressed(
        paths["matrix"],
        delta=signatures.delta,
        genes=np.asarray(signatures.genes, dtype=str),
        control=control,
        has_control=np.asarray(signatures.control is not None),
    )

    signatures.meta.to_csv(
        paths["metadata"],
        index=True,
        index_label=_CACHE_INDEX_COLUMN,
    )

    cache_record = {
        "cache_format": "adipose_drug_discovery.perturb_signatures",
        "cache_format_version": _CACHE_FORMAT_VERSION,
        "n_signatures": signatures.n_signatures,
        "n_genes": signatures.n_genes,
        "has_control": signatures.control is not None,
        "metadata_index_name": signatures.meta.index.name,
        "signature_provenance": _json_compatible(signatures.provenance),
    }
    paths["provenance"].write_text(
        json.dumps(cache_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def load_perturb_signatures(prefix: str | Path) -> PerturbSignatures:
    """Load and validate a cache written by `save_perturb_signatures`.

    Args:
      prefix: Shared cache prefix or the matrix ".npz" path.

    Returns:
      Cached perturbation signatures.
    """
    paths = _cache_paths(prefix)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(
                f"perturbation cache file does not exist: {path}"
            )

    cache_record = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    if cache_record.get("cache_format_version") != _CACHE_FORMAT_VERSION:
        raise ValueError(
            "unsupported perturbation cache format version: "
            f"{cache_record.get('cache_format_version')!r}"
        )

    with np.load(paths["matrix"], allow_pickle=False) as stored:
        required = {"delta", "genes", "control", "has_control"}
        if missing := required.difference(stored.files):
            raise ValueError(
                f"perturbation matrix cache is missing arrays: {missing}"
            )
        delta = np.asarray(stored["delta"], dtype=np.float64)
        genes = tuple(np.asarray(stored["genes"], dtype=str).tolist())
        has_control = bool(np.asarray(stored["has_control"]).item())
        control = (
            np.asarray(stored["control"], dtype=np.float64)
            if has_control
            else None
        )

    meta = pd.read_csv(paths["metadata"], index_col=_CACHE_INDEX_COLUMN)
    meta.index.name = cache_record.get("metadata_index_name")
    provenance = cache_record.get("signature_provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("cached signature provenance must be a JSON object")

    signatures = PerturbSignatures(
        delta=delta,
        genes=genes,
        meta=meta,
        control=control,
        provenance=provenance,
    )
    expected_shape = (
        int(cache_record.get("n_signatures", -1)),
        int(cache_record.get("n_genes", -1)),
    )
    if signatures.delta.shape != expected_shape:
        raise ValueError(
            "cached matrix shape does not match provenance: "
            f"{signatures.delta.shape} versus {expected_shape}"
        )
    return signatures


def load_tahoe_signatures(
    input_path: str | Path,
    *,
    count_layer: str | None = "counts",
    drug_col: str = "drug",
    vehicle_label: str = "DMSO_TF",
    context_cols: Sequence[str] = ("cell_line_id", "plate"),
    sample_col: str | None = "sample",
    dose_col: str | None = None,
    time_col: str | None = None,
    target_col: str | None = "targets",
    mechanism_col: str | None = "moa-fine",
    chunk_size: int = 4_096,
    normalization_target: float = 1_000_000.0,
) -> PerturbSignatures:
    """Build Tahoe drug-minus-matched-vehicle signatures from raw counts.

    The input may be one H5AD or a directory of H5AD shards. Expression is
    opened in backed mode and aggregated in bounded row chunks. Each treated
    condition is converted from summed counts to log1p CPM, then its vehicle
    profile matched on the explicit `context_cols` is subtracted. Vehicle cells
    are never pooled across those contexts.

    Args:
      input_path: H5AD file or directory containing H5AD shards.
      count_layer: Raw-count layer. None explicitly selects `adata.X`.
      drug_col: Observation column containing drug or vehicle identity.
      vehicle_label: Exact vehicle value in `drug_col`.
      context_cols: Columns that must match between treatment and vehicle.
      sample_col: Optional column separating treated experimental samples.
      dose_col: Optional dose column separating treated conditions.
      time_col: Optional time column separating treated conditions.
      target_col: Optional target annotation retained by consensus.
      mechanism_col: Optional mechanism annotation retained by consensus.
      chunk_size: Maximum number of cell rows read from expression at once.
      normalization_target: Library-size target used before `log1p`.

    Returns:
      Tahoe signatures with a matched vehicle expression row for every delta.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if not np.isfinite(normalization_target) or normalization_target <= 0.0:
        raise ValueError("normalization_target must be positive and finite")
    resolved_context = _validated_column_names(
        context_cols,
        parameter_name="context_cols",
    )
    shards = _h5ad_shards(input_path)

    treated_sums: dict[tuple[object, ...], np.ndarray] = {}
    treated_cells: dict[tuple[object, ...], int] = {}
    control_sums: dict[tuple[object, ...], np.ndarray] = {}
    control_cells: dict[tuple[object, ...], int] = {}
    annotation_values: dict[tuple[object, ...], dict[str, set[str]]] = {}
    reference_genes: tuple[str, ...] | None = None
    treatment_group_cols: list[str] | None = None
    retained_annotation_cols: list[str] | None = None
    source_cells = 0
    vehicle_cells = 0

    for shard in shards:
        backed = ad.read_h5ad(shard, backed="r")
        try:
            # read_h5ad returns a DataFrame here, although AnnData's backed-mode
            # annotation allows the broader experimental Dataset2D type.
            obs = cast(pd.DataFrame, backed.obs)
            required_obs = [drug_col, *resolved_context]
            if missing := [
                column for column in required_obs if column not in obs
            ]:
                raise KeyError(
                    f"Tahoe obs is missing required columns: {missing}"
                )
            if bool(obs.loc[:, required_obs].isna().to_numpy().any()):
                raise ValueError(
                    "Tahoe drug and context columns must not contain missing "
                    "values"
                )

            optional_groups = [sample_col, dose_col, time_col]
            shard_group_cols = list(
                dict.fromkeys(
                    [
                        *resolved_context,
                        drug_col,
                        *[
                            column
                            for column in optional_groups
                            if column is not None and column in obs
                        ],
                    ]
                )
            )
            shard_annotations = [
                column
                for column in (target_col, mechanism_col)
                if column is not None and column in obs
            ]
            if treatment_group_cols is None:
                treatment_group_cols = shard_group_cols
                retained_annotation_cols = shard_annotations
            elif treatment_group_cols != shard_group_cols:
                raise ValueError(
                    "Tahoe shards expose inconsistent treatment grouping "
                    f"columns: {treatment_group_cols} versus {shard_group_cols}"
                )
            elif retained_annotation_cols != shard_annotations:
                raise ValueError(
                    "Tahoe shards expose inconsistent annotation columns: "
                    f"{retained_annotation_cols} versus {shard_annotations}"
                )

            shard_genes = tuple(str(gene) for gene in backed.var_names)
            if len(shard_genes) != len(set(shard_genes)):
                raise ValueError(f"Tahoe var names are not unique in {shard}")
            if reference_genes is None:
                reference_genes = shard_genes
                gene_order = None
            else:
                gene_order = _gene_reorder(shard_genes, reference_genes, shard)

            matrix_source = _count_matrix(backed, count_layer=count_layer)
            source_cells += backed.n_obs
            for start in range(0, backed.n_obs, chunk_size):
                stop = min(start + chunk_size, backed.n_obs)
                chunk_obs = obs.iloc[start:stop]
                chunk_counts = _read_count_chunk(
                    matrix_source,
                    start=start,
                    stop=stop,
                    gene_order=gene_order,
                )
                _validate_raw_counts(chunk_counts, source=shard)

                vehicle_mask = (
                    chunk_obs[drug_col]
                    .astype("string")
                    .eq(vehicle_label)
                    .fillna(False)
                    .to_numpy(dtype=bool)
                )
                treated_mask = ~vehicle_mask
                vehicle_cells += int(vehicle_mask.sum())

                _accumulate_count_groups(
                    chunk_counts,
                    chunk_obs,
                    mask=vehicle_mask,
                    group_cols=resolved_context,
                    sums=control_sums,
                    cell_counts=control_cells,
                )
                _accumulate_count_groups(
                    chunk_counts,
                    chunk_obs,
                    mask=treated_mask,
                    group_cols=shard_group_cols,
                    sums=treated_sums,
                    cell_counts=treated_cells,
                    annotation_cols=shard_annotations,
                    annotation_values=annotation_values,
                )
        finally:
            if backed.file is not None:
                backed.file.close()

    if reference_genes is None or treatment_group_cols is None:
        raise ValueError("Tahoe input did not yield a gene matrix")
    if not treated_sums:
        raise ValueError("Tahoe input contains no non-vehicle treatment cells")

    ordered_treatments = sorted(treated_sums, key=_sortable_group_key)
    if missing_contexts := sorted(
        {
            key[: len(resolved_context)]
            for key in ordered_treatments
            if key[: len(resolved_context)] not in control_sums
        },
        key=_sortable_group_key,
    ):
        preview = ", ".join(str(context) for context in missing_contexts[:3])
        raise ValueError(
            f"{len(missing_contexts)} treated Tahoe contexts lack a matched "
            f"{vehicle_label!r} vehicle (for example: {preview})"
        )

    deltas: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    metadata_records: list[dict[str, object]] = []
    for row_index, treatment_key in enumerate(ordered_treatments):
        context_key = treatment_key[: len(resolved_context)]
        treated_expression = _log1p_cpm(
            treated_sums[treatment_key],
            normalization_target=normalization_target,
            label=f"treated group {treatment_key}",
        )
        control_expression = _log1p_cpm(
            control_sums[context_key],
            normalization_target=normalization_target,
            label=f"vehicle context {context_key}",
        )
        deltas.append(treated_expression - control_expression)
        controls.append(control_expression)

        signature_id = f"tahoe_{row_index:08d}"
        record = {
            column: pd.NA if value is None else value
            for column, value in zip(
                treatment_group_cols,
                treatment_key,
                strict=True,
            )
        } | {
            "signature_id": signature_id,
            "source": "tahoe",
            "n_treated_cells": treated_cells[treatment_key],
            "n_control_cells": control_cells[context_key],
        }
        for column in retained_annotation_cols or []:
            candidates = annotation_values.get(treatment_key, {}).get(
                column, set()
            )
            record[column] = (
                next(iter(candidates)) if len(candidates) == 1 else pd.NA
            )
        metadata_records.append(record)

    metadata = pd.DataFrame.from_records(metadata_records).set_index(
        "signature_id",
        drop=False,
    )
    provenance = {
        "source": "tahoe",
        "input_files": [_file_identity(path) for path in shards],
        "count_source": count_layer or "X",
        "drug_column": drug_col,
        "vehicle_label": vehicle_label,
        "context_columns": list(resolved_context),
        "treatment_group_columns": treatment_group_cols,
        "transform": "log1p_cpm_of_summed_raw_counts",
        "normalization_target": normalization_target,
        "chunk_size": chunk_size,
        "n_source_cells": source_cells,
        "n_vehicle_cells": vehicle_cells,
        "n_treated_cells": source_cells - vehicle_cells,
    }
    return PerturbSignatures(
        delta=np.vstack(deltas),
        genes=reference_genes,
        meta=metadata,
        control=np.vstack(controls),
        provenance=provenance,
    )


def load_lincs_signatures(
    matrix_path: str | Path,
    metadata_path: str | Path,
    *,
    signature_id_col: str = "signature_id",
    gene_metadata_path: str | Path | None = None,
    gene_col: str = "gene",
    landmark_col: str = "is_landmark",
    gene_id_col: str | None = None,
    perturbation_type_col: str | None = None,
    compound_type: str | None = None,
    chunk_size: int = 4_096,
) -> PerturbSignatures:
    """Load measured LINCS signatures using explicit signature and gene IDs.

    Native GCTX and wide CSV, TSV, or Parquet matrices are supported. Matrix
    rows are joined to metadata by `signature_id_col` while retaining matrix
    order. Extra metadata rows are allowed, but every measured signature must
    have exactly one metadata row. When gene metadata is supplied, only genes
    explicitly marked as measured landmarks are retained.

    Args:
      matrix_path: Native GCTX or wide tabular signature matrix.
      metadata_path: CSV, TSV, or Parquet signature metadata table.
      signature_id_col: Identifier present in both input tables.
      gene_metadata_path: Optional gene metadata used for landmark filtering.
      gene_col: Gene-symbol column in gene metadata.
      landmark_col: Boolean-like measured-landmark annotation.
      gene_id_col: Gene-metadata identifier matching GCTX row IDs.
      perturbation_type_col: Optional metadata column used to retain compounds.
      compound_type: Exact compound label retained from the type column.
      chunk_size: GCTX signature rows read per bounded chunk.

    Returns:
      Aligned measured LINCS signatures and metadata.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if (perturbation_type_col is None) != (compound_type is None):
        raise ValueError(
            "perturbation_type_col and compound_type must be provided together"
        )

    matrix_file = Path(matrix_path)
    metadata_file = Path(metadata_path)
    metadata = _read_table(metadata_file)
    metadata_ids = _signature_identifiers(
        metadata,
        signature_id_col=signature_id_col,
        table_name="metadata",
    )
    if matrix_file.name.casefold().endswith(".gctx"):
        return _load_lincs_gctx(
            matrix_file,
            metadata_file=metadata_file,
            metadata=metadata,
            metadata_ids=metadata_ids,
            signature_id_col=signature_id_col,
            gene_metadata_path=gene_metadata_path,
            gene_col=gene_col,
            landmark_col=landmark_col,
            gene_id_col=gene_id_col,
            perturbation_type_col=perturbation_type_col,
            compound_type=compound_type,
            chunk_size=chunk_size,
        )

    matrix = _read_table(matrix_file)
    matrix_ids = _signature_identifiers(
        matrix,
        signature_id_col=signature_id_col,
        table_name="matrix",
    )
    aligned_metadata = _align_lincs_metadata(
        matrix_ids,
        metadata,
        metadata_ids=metadata_ids,
        signature_id_col=signature_id_col,
    )
    retained_positions = _retained_lincs_positions(
        aligned_metadata,
        perturbation_type_col=perturbation_type_col,
        compound_type=compound_type,
    )
    aligned_metadata = aligned_metadata.iloc[retained_positions].copy()

    genes = [column for column in matrix.columns if column != signature_id_col]
    if not genes:
        raise ValueError("LINCS matrix contains no gene columns")
    if len(genes) != len(set(genes)):
        raise ValueError("LINCS matrix gene columns must be unique")

    gene_metadata_file = None
    if gene_metadata_path is not None:
        gene_metadata_file = Path(gene_metadata_path)
        gene_metadata = _read_table(gene_metadata_file)
        if gene_col not in gene_metadata or landmark_col not in gene_metadata:
            raise KeyError(
                "LINCS gene metadata must contain "
                f"{gene_col!r} and {landmark_col!r}"
            )
        gene_names = cast(pd.Series, gene_metadata[gene_col]).astype("string")
        if bool(gene_names.isna().any()) or bool(gene_names.duplicated().any()):
            raise ValueError(
                "LINCS gene metadata gene identifiers must be unique"
            )
        landmark_genes = set(
            gene_names.loc[
                _boolean_mask(cast(pd.Series, gene_metadata[landmark_col]))
            ].astype(str)
        )
        genes = [gene for gene in genes if gene in landmark_genes]
        if not genes:
            raise ValueError(
                "LINCS landmark filter retained no measured matrix genes"
            )

    try:
        delta = (
            matrix.iloc[retained_positions]
            .loc[:, genes]
            .apply(pd.to_numeric, errors="raise")
            .to_numpy(dtype=np.float64)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("LINCS matrix gene columns must be numeric") from error
    if bool(np.isinf(delta).any()):
        raise ValueError("LINCS matrix contains infinite response values")

    provenance: dict[str, object] = {
        "source": "lincs",
        "matrix_file": _file_identity(matrix_file),
        "metadata_file": _file_identity(metadata_file),
        "signature_id_column": signature_id_col,
        "representation": (
            "measured_landmark_genes"
            if gene_metadata_file is not None
            else "measured_matrix_genes"
        ),
        "n_matrix_signatures": len(matrix),
        "n_metadata_signatures": len(metadata),
        "n_retained_signatures": len(aligned_metadata),
        "n_retained_genes": len(genes),
    }
    if gene_metadata_file is not None:
        provenance["gene_metadata_file"] = _file_identity(gene_metadata_file)
        provenance["gene_column"] = gene_col
        provenance["landmark_column"] = landmark_col
    if perturbation_type_col is not None:
        provenance["perturbation_type_column"] = perturbation_type_col
        provenance["retained_perturbation_type"] = compound_type

    return PerturbSignatures(
        delta=delta,
        genes=genes,
        meta=aligned_metadata,
        provenance=provenance,
    )


def _load_lincs_gctx(
    matrix_file: Path,
    *,
    metadata_file: Path,
    metadata: pd.DataFrame,
    metadata_ids: pd.Series,
    signature_id_col: str,
    gene_metadata_path: str | Path | None,
    gene_col: str,
    landmark_col: str,
    gene_id_col: str | None,
    perturbation_type_col: str | None,
    compound_type: str | None,
    chunk_size: int,
) -> PerturbSignatures:
    """Load GCTX without materializing all matrix genes."""
    if not matrix_file.is_file():
        raise FileNotFoundError(f"GCTX matrix does not exist: {matrix_file}")
    with h5py.File(matrix_file, "r") as handle:
        required = (
            "0/DATA/0/matrix",
            "0/META/COL/id",
            "0/META/ROW/id",
        )
        if missing := [name for name in required if name not in handle]:
            raise ValueError(f"GCTX is missing required datasets: {missing}")
        matrix_object = handle["0/DATA/0/matrix"]
        signature_id_object = handle["0/META/COL/id"]
        gene_id_object = handle["0/META/ROW/id"]
        if not all(
            isinstance(item, h5py.Dataset)
            for item in (
                matrix_object,
                signature_id_object,
                gene_id_object,
            )
        ):
            raise ValueError("GCTX matrix and IDs must be HDF5 datasets")
        matrix = cast(h5py.Dataset, matrix_object)
        signature_id_dataset = cast(h5py.Dataset, signature_id_object)
        gene_id_dataset = cast(h5py.Dataset, gene_id_object)
        matrix_ids = pd.Series(
            _hdf5_strings(signature_id_dataset),
            dtype="string",
        )
        matrix_gene_ids = _hdf5_strings(gene_id_dataset)
        orientation = _gctx_orientation(
            matrix.shape,
            n_signatures=len(matrix_ids),
            n_genes=len(matrix_gene_ids),
        )
        _validate_identifier_series(matrix_ids, table_name="GCTX matrix")
        aligned_metadata = _align_lincs_metadata(
            matrix_ids,
            metadata,
            metadata_ids=metadata_ids,
            signature_id_col=signature_id_col,
        )
        signature_positions = _retained_lincs_positions(
            aligned_metadata,
            perturbation_type_col=perturbation_type_col,
            compound_type=compound_type,
        )
        aligned_metadata = aligned_metadata.iloc[signature_positions].copy()
        genes, gene_positions, gene_metadata_file = _gctx_landmark_genes(
            matrix_gene_ids,
            gene_metadata_path=gene_metadata_path,
            gene_id_col=gene_id_col,
            gene_col=gene_col,
            landmark_col=landmark_col,
        )
        delta = _read_gctx_matrix(
            matrix,
            orientation=orientation,
            signature_positions=signature_positions,
            gene_positions=gene_positions,
            chunk_size=chunk_size,
        )

    provenance: dict[str, object] = {
        "source": "lincs",
        "matrix_file": _file_identity(matrix_file),
        "metadata_file": _file_identity(metadata_file),
        "signature_id_column": signature_id_col,
        "representation": (
            "measured_landmark_genes"
            if gene_metadata_file is not None
            else "gctx_matrix_genes"
        ),
        "matrix_format": "gctx",
        "matrix_orientation": orientation,
        "n_matrix_signatures": len(matrix_ids),
        "n_metadata_signatures": len(metadata),
        "n_retained_signatures": len(aligned_metadata),
        "n_matrix_genes": len(matrix_gene_ids),
        "n_retained_genes": len(genes),
        "chunk_size": chunk_size,
    }
    if gene_metadata_file is not None:
        provenance |= {
            "gene_metadata_file": _file_identity(gene_metadata_file),
            "gene_id_column": gene_id_col,
            "gene_column": gene_col,
            "landmark_column": landmark_col,
        }
    if perturbation_type_col is not None:
        provenance["perturbation_type_column"] = perturbation_type_col
        provenance["retained_perturbation_type"] = compound_type
    return PerturbSignatures(
        delta=delta,
        genes=genes,
        meta=aligned_metadata,
        provenance=provenance,
    )


def _signature_identifiers(
    table: pd.DataFrame,
    *,
    signature_id_col: str,
    table_name: str,
) -> pd.Series:
    """Return validated string signature identifiers from one table."""
    if signature_id_col not in table:
        raise KeyError(f"LINCS {table_name} is missing {signature_id_col!r}")
    identifiers = cast(pd.Series, table[signature_id_col]).astype("string")
    _validate_identifier_series(identifiers, table_name=table_name)
    return identifiers


def _validate_identifier_series(
    identifiers: pd.Series,
    *,
    table_name: str,
) -> None:
    """Reject missing, blank, or duplicated identifiers."""
    if bool(identifiers.isna().any()) or bool(
        identifiers.str.strip().eq("").any()
    ):
        raise ValueError(
            f"LINCS {table_name} contains missing signature identifiers"
        )
    if bool(identifiers.duplicated().any()):
        raise ValueError(
            f"LINCS {table_name} contains duplicate signature identifiers"
        )


def _align_lincs_metadata(
    matrix_ids: pd.Series,
    metadata: pd.DataFrame,
    *,
    metadata_ids: pd.Series,
    signature_id_col: str,
) -> pd.DataFrame:
    """Align metadata to explicit matrix signature order."""
    string_matrix_ids = matrix_ids.astype(str)
    aligned_source = metadata.assign(
        _signature_key=metadata_ids.astype(str)
    ).set_index("_signature_key", drop=True)
    missing = pd.Index(string_matrix_ids).difference(aligned_source.index)
    if not missing.empty:
        preview = ", ".join(missing[:3].astype(str))
        raise ValueError(
            f"{len(missing)} LINCS matrix signatures lack metadata "
            f"(for example: {preview})"
        )
    aligned = aligned_source.loc[string_matrix_ids.tolist()].copy()
    aligned.index = pd.Index(string_matrix_ids, name=signature_id_col)
    aligned[signature_id_col] = string_matrix_ids.to_numpy()
    aligned["source"] = "lincs"
    return aligned


def _retained_lincs_positions(
    aligned_metadata: pd.DataFrame,
    *,
    perturbation_type_col: str | None,
    compound_type: str | None,
) -> np.ndarray:
    """Return matrix positions retained by an optional exact type filter."""
    if perturbation_type_col is None:
        return np.arange(len(aligned_metadata), dtype=np.int64)
    if perturbation_type_col not in aligned_metadata:
        raise KeyError(
            "LINCS metadata is missing perturbation type column "
            f"{perturbation_type_col!r}"
        )
    retained = (
        aligned_metadata[perturbation_type_col]
        .astype("string")
        .eq(str(compound_type))
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    positions = np.flatnonzero(retained)
    if not positions.size:
        raise ValueError(
            f"LINCS filter retained no {compound_type!r} perturbations"
        )
    return positions


def _gctx_landmark_genes(
    matrix_gene_ids: np.ndarray,
    *,
    gene_metadata_path: str | Path | None,
    gene_id_col: str | None,
    gene_col: str,
    landmark_col: str,
) -> tuple[list[str], np.ndarray, Path | None]:
    """Map ordered GCTX gene IDs to unique measured landmark symbols."""
    if gene_metadata_path is None:
        genes = matrix_gene_ids.astype(str).tolist()
        if len(genes) != len(set(genes)):
            raise ValueError("GCTX gene identifiers must be unique")
        return genes, np.arange(len(genes), dtype=np.int64), None
    if gene_id_col is None:
        raise ValueError(
            "gene_id_col is required when GCTX gene metadata is supplied"
        )

    gene_metadata_file = Path(gene_metadata_path)
    gene_metadata = _read_table(gene_metadata_file)
    required = (gene_id_col, gene_col, landmark_col)
    if missing := [
        column for column in required if column not in gene_metadata
    ]:
        raise KeyError(f"LINCS gene metadata is missing columns: {missing}")
    identifiers = gene_metadata[gene_id_col].astype("string")
    if bool(identifiers.isna().any()) or bool(identifiers.duplicated().any()):
        raise ValueError("LINCS gene metadata identifiers must be unique")
    source = gene_metadata.assign(_gene_key=identifiers.astype(str)).set_index(
        "_gene_key",
        drop=True,
    )
    ordered_ids = pd.Index(matrix_gene_ids.astype(str))
    missing_ids = ordered_ids.difference(source.index)
    if not missing_ids.empty:
        preview = ", ".join(missing_ids[:3].astype(str))
        raise ValueError(
            f"{len(missing_ids)} GCTX genes lack metadata "
            f"(for example: {preview})"
        )
    ordered = source.loc[ordered_ids.tolist()]
    landmark = _boolean_mask(cast(pd.Series, ordered[landmark_col])).to_numpy()
    positions = np.flatnonzero(landmark)
    symbols = ordered.iloc[positions][gene_col].astype("string")
    if bool(symbols.isna().any()) or bool(symbols.str.strip().eq("").any()):
        raise ValueError("LINCS landmark gene symbols must not be missing")
    genes = symbols.astype(str).tolist()
    if len(genes) != len(set(genes)):
        raise ValueError("LINCS landmark gene symbols must be unique")
    if not genes:
        raise ValueError("LINCS landmark filter retained no GCTX genes")
    return genes, positions, gene_metadata_file


def _hdf5_strings(dataset: h5py.Dataset) -> np.ndarray:
    """Decode one-dimensional HDF5 identifiers as Python strings."""
    values = np.asarray(dataset.asstr()[:], dtype=str)
    if values.ndim != 1:
        raise ValueError("GCTX identifier datasets must be one-dimensional")
    return values


def _gctx_orientation(
    shape: tuple[int, ...],
    *,
    n_signatures: int,
    n_genes: int,
) -> str:
    """Resolve matrix axes from explicit GCTX metadata dimensions."""
    if len(shape) != 2:
        raise ValueError(f"GCTX matrix must be two-dimensional: {shape}")
    if shape == (n_signatures, n_genes):
        return "signatures_by_genes"
    if shape == (n_genes, n_signatures):
        return "genes_by_signatures"
    raise ValueError(
        "GCTX matrix dimensions do not match signature and gene IDs: "
        f"{shape} versus ({n_signatures}, {n_genes})"
    )


def _read_gctx_matrix(
    matrix: h5py.Dataset,
    *,
    orientation: str,
    signature_positions: np.ndarray,
    gene_positions: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """Read selected GCTX signatures and genes in bounded source-row chunks."""
    delta = np.empty(
        (len(signature_positions), len(gene_positions)),
        dtype=np.float64,
    )
    n_source_signatures = (
        matrix.shape[0]
        if orientation == "signatures_by_genes"
        else matrix.shape[1]
    )
    output_start = 0
    for start in range(0, n_source_signatures, chunk_size):
        stop = min(start + chunk_size, n_source_signatures)
        selected = signature_positions[
            (signature_positions >= start) & (signature_positions < stop)
        ]
        if not selected.size:
            continue
        if orientation == "signatures_by_genes":
            block = matrix[start:stop, gene_positions]
        else:
            block = matrix[gene_positions, start:stop].T
        retained = np.asarray(block)[selected - start]
        output_stop = output_start + len(selected)
        delta[output_start:output_stop] = retained
        output_start = output_stop
    if output_start != len(signature_positions):
        raise RuntimeError("GCTX chunk reader did not retain every signature")
    if not bool(np.isfinite(delta).all()):
        raise ValueError("LINCS GCTX contains non-finite response values")
    return delta


def _cache_paths(prefix: str | Path) -> dict[str, Path]:
    """Return the three paths owned by one perturbation cache prefix."""
    resolved = Path(prefix)
    if resolved.suffix == ".npz":
        resolved = resolved.with_suffix("")
    return {
        "matrix": Path(f"{resolved}.npz"),
        "metadata": Path(f"{resolved}.metadata.csv.gz"),
        "provenance": Path(f"{resolved}.provenance.json"),
    }


def _json_compatible(value: object) -> object:
    """Return a recursively JSON-compatible provenance value."""
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        numpy_scalar: Any = value
        return numpy_scalar.item()
    return None if value is pd.NA else value


def _validated_column_names(
    columns: Sequence[str],
    *,
    parameter_name: str,
) -> list[str]:
    """Return non-empty, unique column names from a public argument."""
    if isinstance(columns, str):
        raise TypeError(f"{parameter_name} must be a sequence of column names")
    resolved = list(columns)
    if not resolved:
        raise ValueError(f"{parameter_name} must contain at least one column")
    if any(
        not isinstance(column, str) or not column.strip() for column in resolved
    ):
        raise ValueError(f"{parameter_name} must contain non-empty strings")
    if len(resolved) != len(set(resolved)):
        raise ValueError(f"{parameter_name} must not contain duplicate columns")
    return resolved


def _h5ad_shards(input_path: str | Path) -> list[Path]:
    """Return one file or deterministic H5AD shards from a directory."""
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        if shards := sorted(path.glob("*.h5ad")):
            return shards
        raise FileNotFoundError(
            f"Tahoe directory contains no H5AD files: {path}"
        )
    raise FileNotFoundError(f"Tahoe input does not exist: {path}")


def _gene_reorder(
    observed: tuple[str, ...],
    expected: tuple[str, ...],
    source: Path,
) -> np.ndarray | None:
    """Return a column reorder when a shard has the same genes in new order."""
    if observed == expected:
        return None
    observed_index = pd.Index(observed)
    indexer = observed_index.get_indexer(expected)
    if len(observed) != len(expected) or bool((indexer < 0).any()):
        raise ValueError(
            f"Tahoe shard gene space differs from earlier shards: {source}"
        )
    return indexer


def _count_matrix(backed: ad.AnnData, *, count_layer: str | None) -> Any:
    """Return the explicitly configured backed raw-count matrix."""
    if count_layer is None:
        if backed.X is None:
            raise ValueError("Tahoe AnnData has no count matrix in X")
        return backed.X
    if count_layer not in backed.layers:
        raise KeyError(f"Tahoe count layer is absent: {count_layer!r}")
    return backed.layers[count_layer]


def _read_count_chunk(
    matrix: Any,
    *,
    start: int,
    stop: int,
    gene_order: np.ndarray | None,
) -> sp.spmatrix | np.ndarray:
    """Read one bounded row chunk and apply an explicit gene reorder."""
    chunk = matrix[start:stop, :]
    if hasattr(chunk, "to_memory"):
        chunk = chunk.to_memory()
    if gene_order is not None:
        chunk = chunk[:, gene_order]
    return chunk.tocsr() if sp.issparse(chunk) else np.asarray(chunk)


def _validate_raw_counts(
    counts: sp.spmatrix | np.ndarray,
    *,
    source: Path,
) -> None:
    """Reject values incompatible with a declared raw-count source."""
    if sp.issparse(counts):
        sparse_counts: Any = counts
        values = np.asarray(sparse_counts.data)
    else:
        values = np.asarray(counts)
    if not bool(np.isfinite(values).all()):
        raise ValueError(
            f"Tahoe raw counts contain non-finite values: {source}"
        )
    if bool((values < 0).any()):
        raise ValueError(f"Tahoe raw counts contain negative values: {source}")
    if bool((values != np.floor(values)).any()):
        raise ValueError(
            f"Tahoe raw counts contain fractional values: {source}"
        )


def _accumulate_count_groups(
    counts: sp.spmatrix | np.ndarray,
    obs: pd.DataFrame,
    *,
    mask: np.ndarray,
    group_cols: Sequence[str],
    sums: dict[tuple[object, ...], np.ndarray],
    cell_counts: dict[tuple[object, ...], int],
    annotation_cols: Sequence[str] = (),
    annotation_values: dict[tuple[object, ...], dict[str, set[str]]]
    | None = None,
) -> None:
    """Accumulate raw count sums for explicit observation-key tuples."""
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        return

    selected_obs = obs.iloc[positions]
    keys = [
        tuple(_canonical_group_value(value) for value in row)
        for row in selected_obs.loc[:, list(group_cols)].itertuples(
            index=False,
            name=None,
        )
    ]
    key_codes: dict[tuple[object, ...], int] = {}
    unique_keys: list[tuple[object, ...]] = []
    codes = np.empty(len(keys), dtype=np.int64)
    for position, key in enumerate(keys):
        if key not in key_codes:
            key_codes[key] = len(unique_keys)
            unique_keys.append(key)
        codes[position] = key_codes[key]

    indicator = sp.csr_matrix(
        (
            np.ones(len(positions), dtype=np.float64),
            (codes, positions),
        ),
        shape=(len(unique_keys), counts.shape[0]),
    )
    grouped = indicator @ counts
    group_sizes = np.bincount(codes, minlength=len(unique_keys))
    for group_index, key in enumerate(unique_keys):
        if sp.issparse(grouped):
            sparse_grouped: Any = grouped
            group_sum = sparse_grouped.getrow(group_index).toarray().ravel()
        else:
            group_sum = np.asarray(grouped[group_index]).ravel()
        group_sum = group_sum.astype(np.float64, copy=False)
        if key in sums:
            sums[key] += group_sum
            cell_counts[key] += int(group_sizes[group_index])
        else:
            sums[key] = group_sum.copy()
            cell_counts[key] = int(group_sizes[group_index])

        if annotation_values is None:
            continue
        records = selected_obs.iloc[np.flatnonzero(codes == group_index)]
        stored = annotation_values.setdefault(key, {})
        for column in annotation_cols:
            candidates = stored.setdefault(column, set())
            candidates.update(records[column].dropna().astype(str).unique())


def _canonical_group_value(value: object) -> object:
    """Return a stable hashable scalar for a grouping-key tuple."""
    missing: Any = pd.isna(value)
    if bool(missing):
        return None
    if isinstance(value, np.generic):
        numpy_scalar: Any = value
        return numpy_scalar.item()
    return value


def _sortable_group_key(values: tuple[object, ...]) -> tuple[str, ...]:
    """Return a deterministic string key for heterogeneous metadata tuples."""
    return tuple("" if value is None else str(value) for value in values)


def _log1p_cpm(
    counts: np.ndarray,
    *,
    normalization_target: float,
    label: str,
) -> np.ndarray:
    """Convert one summed-count vector to log1p counts per target total."""
    library_size = float(np.sum(counts, dtype=np.float64))
    if not np.isfinite(library_size) or library_size <= 0.0:
        raise ValueError(f"{label} has no positive count library")
    return np.log1p(counts * (normalization_target / library_size))


def _read_table(path: Path) -> pd.DataFrame:
    """Read a supported CSV, tab-delimited text, or Parquet table."""
    if not path.is_file():
        raise FileNotFoundError(f"table does not exist: {path}")
    name = path.name.casefold()
    if name.endswith(".parquet") or name.endswith(".pq"):
        return pd.read_parquet(path)
    if name.endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")):
        return pd.read_csv(path, sep="\t")
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(
        f"unsupported table format for {path}; use CSV, TSV, TXT, or Parquet"
    )


def _boolean_mask(values: pd.Series) -> pd.Series:
    """Interpret common boolean-like landmark annotations explicitly."""
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values.dtype):
        numeric = pd.Series(
            pd.to_numeric(values, errors="coerce"),
            index=values.index,
            dtype=np.float64,
        )
        observed = numeric.dropna()
        invalid = observed.loc[~observed.isin(np.asarray([0.0, 1.0]))]
        if not invalid.empty:
            raise ValueError("numeric landmark annotations must be 0 or 1")
        return numeric.eq(1).fillna(False)

    normalized = values.astype("string").str.strip().str.casefold()
    truthy = {"true", "t", "yes", "y", "1", "landmark"}
    falsy = {"false", "f", "no", "n", "0", ""}
    invalid = normalized.dropna().loc[~normalized.dropna().isin(truthy | falsy)]
    if not invalid.empty:
        raise ValueError(
            "landmark annotations contain unsupported values: "
            f"{sorted(invalid.unique().tolist())}"
        )
    return normalized.isin(truthy).fillna(False)


def _file_identity(path: Path) -> dict[str, object]:
    """Return a reproducible local file identity record."""
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
