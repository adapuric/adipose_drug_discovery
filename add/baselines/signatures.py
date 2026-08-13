"""Signature alignment and aggregation for baseline methods."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp  # type: ignore[import]

from add.perturb import PerturbSignatures


def _average_signature_groups(
    signatures: PerturbSignatures,
    *,
    group_cols: Sequence[str],
    id_prefix: str,
) -> PerturbSignatures:
    """Average signature replicates within explicit metadata groups."""
    if missing := [
        column for column in group_cols if column not in signatures.meta
    ]:
        raise KeyError(
            f"Perturbation metadata lacks grouping columns: {missing}"
        )
    delta = _dense_matrix(signatures.delta)
    _require_finite(delta, name="perturbation deltas")

    output_delta: list[np.ndarray] = []
    output_meta: list[dict[str, object]] = []
    grouped = signatures.meta.groupby(
        list(group_cols),
        observed=True,
        sort=True,
        dropna=False,
    )

    for keys, positions in grouped.indices.items():
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        group_positions = np.asarray(positions, dtype=int)
        output_delta.append(np.mean(delta[group_positions], axis=0))
        group_meta = signatures.meta.iloc[group_positions]
        record: dict[str, object] = dict(
            zip(group_cols, key_tuple, strict=True)
        )
        for column in group_meta.columns:
            if column in record:
                continue
            record[column] = _collapse_metadata(
                pd.Series(group_meta[column], index=group_meta.index)
            )
        record["n_source_signatures"] = len(group_positions)
        output_meta.append(record)

    meta = _with_signature_ids(
        pd.DataFrame(output_meta),
        prefix=id_prefix,
    )

    return PerturbSignatures(
        delta=np.vstack(output_delta),
        genes=list(signatures.genes),
        meta=meta,
        provenance={
            **dict(signatures.provenance),
            "aggregation": "mean_within_metadata_group",
            "group_cols": list(group_cols),
        },
    )


def _average_drug_contexts(
    context_signatures: PerturbSignatures,
    *,
    drug_col: str,
    context_cols: Sequence[str],
) -> PerturbSignatures:
    """Average context-level deltas once per drug and annotate support."""
    delta = _dense_matrix(context_signatures.delta)
    output_delta: list[np.ndarray] = []
    output_meta: list[dict[str, object]] = []
    grouped = context_signatures.meta.groupby(
        drug_col,
        observed=True,
        sort=True,
        dropna=False,
    )
    for drug, positions in grouped.indices.items():
        group_positions = np.asarray(positions, dtype=int)
        group_meta = context_signatures.meta.iloc[group_positions]
        output_delta.append(np.mean(delta[group_positions], axis=0))
        record: dict[str, object] = {
            drug_col: drug,
            "n_external_contexts": int(
                group_meta.loc[:, list(context_cols)].drop_duplicates().shape[0]
            ),
        }
        for column in group_meta.columns:
            if column in {
                drug_col,
                *context_cols,
                "n_source_signatures",
            }:
                continue
            record[column] = _collapse_metadata(
                pd.Series(group_meta[column], index=group_meta.index)
            )
        output_meta.append(record)

    meta = _with_signature_ids(
        pd.DataFrame(output_meta),
        prefix="mean_drug",
    )

    return PerturbSignatures(
        delta=np.vstack(output_delta),
        genes=list(context_signatures.genes),
        meta=meta,
        provenance={
            **dict(context_signatures.provenance),
            "aggregation": "equal_context_mean_by_drug",
        },
    )


def _subset_signatures(
    signatures: PerturbSignatures,
    signature_ids: Sequence[str],
) -> PerturbSignatures:
    """Return signatures selected by stable signature identifiers."""
    positions = _signature_positions(signatures, signature_ids)
    control = (
        None if signatures.control is None else signatures.control[positions]
    )

    return PerturbSignatures(
        delta=signatures.delta[positions],
        genes=list(signatures.genes),
        meta=signatures.meta.iloc[positions].copy(),
        control=control,
        provenance=dict(signatures.provenance),
    )


def _signature_ids(signatures: PerturbSignatures) -> tuple[str, ...]:
    """Return unique signature identifiers aligned to matrix rows."""
    if "signature_id" in signatures.meta:
        values = signatures.meta["signature_id"].astype(str).tolist()
    else:
        values = signatures.meta.index.astype(str).tolist()
    if len(values) != len(set(values)):
        raise ValueError("Perturbation signature identifiers must be unique.")

    return tuple(values)


def _with_signature_ids(
    metadata: pd.DataFrame,
    *,
    prefix: str,
) -> pd.DataFrame:
    """Assign canonical row identifiers after aggregation or prediction."""
    identifiers = pd.Index(
        [f"{prefix}_{index:06d}" for index in range(len(metadata))],
        name="signature_id",
    )
    result = metadata.copy()
    result.index = identifiers

    # Source signature IDs describe different rows and must not survive as the
    # identity of a newly predicted or averaged profile.
    result["signature_id"] = identifiers.to_numpy()
    return result


def _signature_positions(
    signatures: PerturbSignatures,
    signature_ids: Sequence[str] | None,
) -> np.ndarray:
    """Resolve requested signature IDs to matrix row positions."""
    available = _signature_ids(signatures)
    if signature_ids is None:
        return np.arange(len(available), dtype=int)
    requested = [str(identifier) for identifier in signature_ids]
    if not requested:
        raise ValueError("The training signature set must not be empty.")
    if len(requested) != len(set(requested)):
        raise ValueError("Requested signature IDs must be unique.")
    lookup = {identifier: index for index, identifier in enumerate(available)}
    if missing := [
        identifier for identifier in requested if identifier not in lookup
    ]:
        raise KeyError(f"Unknown perturbation signature IDs: {missing}")
    return np.asarray(
        [lookup[identifier] for identifier in requested], dtype=int
    )


def _gene_positions(
    available_genes: Sequence[str],
    selected_genes: Sequence[str],
) -> np.ndarray:
    """Resolve unique gene identifiers without assuming column order."""
    available = [str(gene) for gene in available_genes]
    if len(available) != len(set(available)):
        raise ValueError("Perturbation genes must be unique.")
    lookup = {gene: index for index, gene in enumerate(available)}
    if missing := [
        str(gene) for gene in selected_genes if str(gene) not in lookup
    ]:
        raise ValueError(
            f"Requested model genes are unavailable: {missing[:10]}"
        )
    return np.asarray([lookup[str(gene)] for gene in selected_genes], dtype=int)


def _required_string_values(
    metadata: pd.DataFrame,
    column: str,
) -> np.ndarray:
    """Return non-missing metadata values as strings."""
    if column not in metadata:
        raise KeyError(f"Perturbation metadata lacks {column!r}.")
    values = pd.Series(metadata[column], index=metadata.index)
    if bool(values.isna().any()):
        raise ValueError(f"Perturbation metadata contains missing {column}.")
    return values.astype(str).to_numpy()


def _context_columns(
    context_col: str | Sequence[str],
) -> tuple[str, ...]:
    """Return validated columns jointly defining an external context."""
    columns: tuple[str, ...]
    if isinstance(context_col, str):
        columns = (context_col,)
    else:
        columns = tuple(context_col)
    if not columns or any(
        not isinstance(column, str) or not column.strip() for column in columns
    ):
        raise ValueError("Context columns must be non-empty strings.")
    if len(columns) != len(set(columns)):
        raise ValueError("Context columns must be unique.")
    return columns


def _context_keys(
    metadata: pd.DataFrame,
    context_cols: Sequence[str],
) -> list[tuple[str, ...]]:
    """Return exact, hashable context keys aligned to metadata rows."""
    context_frame = metadata.loc[:, list(context_cols)]
    if context_frame.isna().any(axis=None):
        raise ValueError("Perturbation metadata contains a missing context.")
    return [
        tuple(str(value) for value in row)
        for row in context_frame.itertuples(index=False, name=None)
    ]


def _context_label(
    metadata_row: pd.Series,
    context_cols: Sequence[str],
) -> str:
    """Return a readable label for one exact context key."""
    key = tuple(str(metadata_row[column]) for column in context_cols)
    return _format_context_key(key, context_cols)


def _format_context_key(
    key: Sequence[str],
    context_cols: Sequence[str],
) -> str:
    """Format a context key without using it for grouping semantics."""
    return "|".join(
        f"{column}={value}"
        for column, value in zip(context_cols, key, strict=True)
    )


def _collapse_metadata(values: pd.Series) -> object:
    """Return one value or a deterministic delimiter-separated set."""
    if observed := sorted({str(value) for value in values.dropna()}):
        return observed[0] if len(observed) == 1 else " | ".join(observed)
    else:
        return pd.NA


def _metadata_annotation(
    metadata_row: pd.Series,
    column: str | None,
) -> object:
    """Return one configured annotation or missing when unavailable."""
    if column is None:
        return pd.NA
    return pd.NA if column not in metadata_row.index else metadata_row[column]


def _metadata_annotation_from_frame(
    metadata: pd.DataFrame,
    column: str | None,
) -> object:
    """Collapse a configured annotation across one drug's contexts."""
    if column is None:
        return pd.NA
    if column not in metadata:
        return pd.NA
    return _collapse_metadata(pd.Series(metadata[column], index=metadata.index))


def _dense_matrix(matrix: np.ndarray | sp.csr_matrix) -> np.ndarray:
    """Return a dense float matrix for bounded profile-level calculations."""
    if isinstance(matrix, sp.csr_matrix):
        return np.asarray(matrix.toarray(), dtype=float)
    return np.asarray(matrix, dtype=float)


def _dense_row(
    matrix: np.ndarray | sp.csr_matrix,
    position: int,
) -> np.ndarray:
    """Return one dense profile vector."""
    return _dense_matrix(matrix[position]).reshape(-1)


def _require_finite(values: np.ndarray, *, name: str) -> None:
    """Reject values that would silently change model support."""
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values.")


def _root_mean_squared_error(
    predicted: np.ndarray,
    observed: np.ndarray,
) -> float:
    """Return RMSE over aligned finite gene values."""
    finite = np.isfinite(predicted) & np.isfinite(observed)
    if not finite.any():
        return np.nan
    return float(
        np.sqrt(np.mean(np.square(predicted[finite] - observed[finite])))
    )
