"""Donor-level raw-count pseudobulk aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import anndata as ad  # type: ignore[import]
import numpy as np
import pandas as pd
from scipy import sparse  # type: ignore[import]


def adipose_group_support(
    adata_or_obs: ad.AnnData | pd.DataFrame,
    *,
    donor_col: str = "Donor",
    condition_col: str = "condition",
    state_col: str = "cell_state_t2d",
    min_cells: int = 20,
    unassigned_label: str = "Unassigned",
    pooled_label: str = "AD_ALL",
) -> pd.DataFrame:
    """Count nuclei supporting every candidate pseudobulk group.

    Unlike the expression output, this table retains groups below
    ``min_cells`` so filtering is observable and can be persisted as
    ``group_support.csv``.

    Args:
      adata_or_obs: Adipocyte AnnData or its observation table.
      donor_col: Biological-replicate column.
      condition_col: Experimental-condition column.
      state_col: Adipocyte-state column.
      min_cells: Retention threshold used for pseudobulk profiles.
      unassigned_label: State excluded from state-specific candidates.
      pooled_label: Label used for pooled adipocyte candidates.

    Returns:
      One row per candidate group with ``n_cells`` and ``retained``.
    """
    if min_cells < 1:
        raise ValueError("min_cells must be at least 1.")
    if isinstance(adata_or_obs, pd.DataFrame):
        obs = adata_or_obs
    else:
        obs = cast(pd.DataFrame, adata_or_obs.obs)
    required = [donor_col, condition_col, state_col]
    if missing := [column for column in required if column not in obs]:
        raise KeyError(f"AnnData obs is missing columns: {missing}")

    state_values = obs[state_col].to_numpy(dtype=object)
    assigned_mask = np.fromiter(
        (
            not pd.isna(value) and str(value) != unassigned_label
            for value in state_values
        ),
        dtype=bool,
        count=len(obs),
    )
    if any(str(value) == pooled_label for value in state_values[assigned_mask]):
        raise ValueError(
            f"Pooled label {pooled_label!r} is already an assigned state."
        )
    state_support = _group_support(
        obs,
        group_cols=(donor_col, condition_col, state_col),
        cell_mask=assigned_mask,
        min_cells=min_cells,
    )
    pooled_support = _group_support(
        obs,
        group_cols=(donor_col, condition_col),
        cell_mask=np.ones(len(obs), dtype=bool),
        min_cells=min_cells,
    )
    pooled_support.insert(2, state_col, pooled_label)
    return pd.concat(
        [state_support, pooled_support],
        axis=0,
        ignore_index=True,
        sort=False,
    )


def aggregate_counts(
    adata: ad.AnnData,
    *,
    group_cols: Sequence[str],
    count_layer: str,
    min_cells: int = 1,
    metadata_cols: Sequence[str] = (),
    cell_mask: np.ndarray | None = None,
    chunk_size: int = 4096,
) -> ad.AnnData:
    """Sum a configured raw-count layer into biological groups.

    The count layer is read in contiguous row chunks, so a backed sparse
    AnnData does not need to be materialized in memory. Each selected cell is
    assigned to exactly one group and contributes its raw-count vector once.

    Args:
      adata: Cell-level observations with raw counts in ``count_layer``.
      group_cols: Observation columns defining one pseudobulk sample.
      count_layer: Required raw-count layer. ``adata.X`` is never substituted.
      min_cells: Minimum selected cells required to retain a group.
      metadata_cols: Optional observation fields to retain when their
        non-missing value is unique within a group. Ambiguous values become
        missing in the output.
      cell_mask: Optional Boolean mask selecting cells before grouping.
      chunk_size: Number of consecutive cells read from the count layer.

    Returns:
      AnnData with one row per retained group and summed counts in ``X``.

    Raises:
      KeyError: If a grouping column or the configured count layer is absent.
      ValueError: If grouping values or the configured layer are invalid.
    """
    groups = list(group_cols)
    if not groups or len(groups) != len(set(groups)):
        raise ValueError("group_cols must contain distinct column names.")
    if min_cells < 1:
        raise ValueError("min_cells must be at least 1.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")

    obs = cast(pd.DataFrame, adata.obs)
    if missing_groups := [column for column in groups if column not in obs]:
        raise KeyError(f"AnnData is missing grouping columns: {missing_groups}")
    if count_layer not in adata.layers:
        raise KeyError(
            f"Configured raw-count layer {count_layer!r} is absent. "
            "No expression fallback was used."
        )

    selected = _validate_cell_mask(cell_mask, n_cells=adata.n_obs)
    selected_obs = obs.iloc[np.flatnonzero(selected)]
    if selected_obs.loc[:, groups].isna().any(axis=None):
        raise ValueError(
            "Selected cells contain missing donor/condition grouping values."
        )

    selected_keys = pd.MultiIndex.from_frame(selected_obs.loc[:, groups])
    selected_codes, factorized_keys = pd.factorize(selected_keys, sort=False)
    unique_keys = cast(pd.MultiIndex, factorized_keys)
    group_sizes = np.bincount(
        selected_codes,
        minlength=len(unique_keys),
    )
    retained_old_codes = np.flatnonzero(group_sizes >= min_cells)

    old_to_new = np.full(len(unique_keys), -1, dtype=np.int64)
    old_to_new[retained_old_codes] = np.arange(
        len(retained_old_codes),
        dtype=np.int64,
    )
    cell_group_codes = np.full(adata.n_obs, -1, dtype=np.int64)
    cell_group_codes[selected] = old_to_new[selected_codes]

    count_matrix = adata.layers[count_layer]
    count_dtype = np.dtype(count_matrix.dtype)  # type: ignore[attr-defined]
    sum_dtype = np.result_type(count_dtype, np.int64)
    summed_counts = _sum_count_chunks(
        count_matrix,
        cell_group_codes=cell_group_codes,
        n_groups=len(retained_old_codes),
        n_genes=adata.n_vars,
        dtype=sum_dtype,
        chunk_size=chunk_size,
    )

    group_obs = unique_keys.to_frame(index=False).iloc[retained_old_codes]
    group_obs = group_obs.reset_index(drop=True)
    # pandas factorization can discard MultiIndex level names; restore the
    # biological grouping contract explicitly rather than relying on it.
    group_obs.columns = groups
    group_obs["n_cells"] = group_sizes[retained_old_codes]

    optional_metadata = [
        column
        for column in dict.fromkeys(metadata_cols)
        if column not in groups and column in obs
    ]
    for column in optional_metadata:
        values = [
            _unique_non_missing_value(
                obs.iloc[
                    np.flatnonzero(selected & (cell_group_codes == new_code))
                ][column]
            )
            for new_code in range(len(retained_old_codes))
        ]
        group_obs[column] = _persistable_metadata(
            values,
            source=cast(pd.Series, obs[column]),
        )

    group_obs.index = pd.Index(
        [f"pseudobulk_{index:06d}" for index in range(len(group_obs))],
        name="pseudobulk_id",
    )
    result = ad.AnnData(
        X=summed_counts,
        obs=group_obs,
        var=cast(pd.DataFrame, adata.var).copy(),
    )
    result.uns["pseudobulk"] = {
        "aggregation": "sum_raw_counts",
        "count_layer": count_layer,
        "group_cols": groups,
        "min_cells": min_cells,
        "chunk_size": chunk_size,
        "missing_metadata_cols": [
            column for column in metadata_cols if column not in obs
        ],
    }
    return result


def build_adipose_pseudobulk(
    adata: ad.AnnData,
    *,
    count_layer: str,
    donor_col: str = "Donor",
    condition_col: str = "condition",
    state_col: str = "cell_state_t2d",
    min_cells: int = 20,
    metadata_cols: Sequence[str] = (),
    unassigned_label: str = "Unassigned",
    pooled_label: str = "AD_ALL",
    chunk_size: int = 4096,
) -> ad.AnnData:
    """Build state-specific and pooled adipocyte pseudobulk profiles.

    State-specific rows aggregate ``Donor x condition x state`` after removing
    unassigned cells. Pooled rows aggregate ``Donor x condition`` across all
    adipocytes, including cells lacking a specific state assignment, and are
    labelled ``AD_ALL``.

    Args:
      adata: Adipocyte-level AnnData.
      count_layer: Required layer containing raw nuclear counts.
      donor_col: Biological-replicate column.
      condition_col: Experimental-condition column.
      state_col: Adipocyte-state column.
      min_cells: Minimum nuclei required independently for each output row.
      metadata_cols: Optional sample metadata carried when uniquely defined.
      unassigned_label: State excluded from state-specific profiles.
      pooled_label: Label assigned to pooled adipocyte profiles.
      chunk_size: Number of nuclei read per count-matrix chunk.

    Returns:
      AnnData containing state-level rows followed by pooled ``AD_ALL`` rows.
    """
    obs = cast(pd.DataFrame, adata.obs)
    if state_col not in obs:
        raise KeyError(f"AnnData is missing state column {state_col!r}.")
    if pooled_label == unassigned_label:
        raise ValueError("pooled_label and unassigned_label must differ.")

    state_values = obs[state_col].to_numpy(dtype=object)
    assigned_mask = np.fromiter(
        (
            not pd.isna(value) and str(value) != unassigned_label
            for value in state_values
        ),
        dtype=bool,
        count=adata.n_obs,
    )
    if any(str(value) == pooled_label for value in state_values[assigned_mask]):
        raise ValueError(
            f"Pooled label {pooled_label!r} is already an assigned state."
        )

    state_profiles = aggregate_counts(
        adata,
        group_cols=(donor_col, condition_col, state_col),
        count_layer=count_layer,
        min_cells=min_cells,
        metadata_cols=metadata_cols,
        cell_mask=assigned_mask,
        chunk_size=chunk_size,
    )
    pooled_profiles = aggregate_counts(
        adata,
        group_cols=(donor_col, condition_col),
        count_layer=count_layer,
        min_cells=min_cells,
        metadata_cols=metadata_cols,
        chunk_size=chunk_size,
    )
    state_profile_obs = cast(pd.DataFrame, state_profiles.obs)
    pooled_profile_obs = cast(pd.DataFrame, pooled_profiles.obs)
    pooled_profile_obs.insert(2, state_col, pooled_label)

    combined_obs = pd.concat(
        [state_profile_obs, pooled_profile_obs],
        axis=0,
        ignore_index=True,
        sort=False,
    )
    combined_obs.index = pd.Index(
        [f"pseudobulk_{index:06d}" for index in range(len(combined_obs))],
        name="pseudobulk_id",
    )
    combined_counts = sparse.vstack(
        [state_profiles.X, pooled_profiles.X],
        format="csr",
    )
    result = ad.AnnData(
        X=combined_counts,
        obs=combined_obs,
        var=cast(pd.DataFrame, adata.var).copy(),
    )
    result.uns["pseudobulk"] = {
        "aggregation": "sum_raw_counts",
        "count_layer": count_layer,
        "state_group_cols": [donor_col, condition_col, state_col],
        "pooled_group_cols": [donor_col, condition_col],
        "min_cells": min_cells,
        "unassigned_label": unassigned_label,
        "pooled_label": pooled_label,
        "chunk_size": chunk_size,
    }
    return result


def _validate_cell_mask(
    cell_mask: np.ndarray | None,
    *,
    n_cells: int,
) -> np.ndarray:
    """Return a validated Boolean observation mask."""
    if cell_mask is None:
        return np.ones(n_cells, dtype=bool)
    mask = np.asarray(cell_mask)
    if mask.dtype != np.bool_ or mask.ndim != 1 or len(mask) != n_cells:
        raise ValueError("cell_mask must be one Boolean value per observation.")
    return mask


def _group_support(
    obs: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    cell_mask: np.ndarray,
    min_cells: int,
) -> pd.DataFrame:
    """Count all observed groups without applying the support filter."""
    selected_obs = obs.iloc[np.flatnonzero(cell_mask)]
    if selected_obs.loc[:, list(group_cols)].isna().any(axis=None):
        raise ValueError(
            "Selected cells contain missing donor/condition grouping values."
        )
    support = (
        selected_obs.groupby(
            list(group_cols),
            observed=True,
            sort=False,
            dropna=False,
        )
        .size()
        .rename("n_cells")
        .reset_index()
    )
    support["retained"] = support["n_cells"] >= min_cells
    return support


def _sum_count_chunks(
    count_matrix: object,
    *,
    cell_group_codes: np.ndarray,
    n_groups: int,
    n_genes: int,
    dtype: np.dtype,
    chunk_size: int,
) -> sparse.csr_matrix:
    """Accumulate row chunks with a sparse group-membership matrix."""
    result = sparse.csr_matrix((n_groups, n_genes), dtype=dtype)
    for start in range(0, len(cell_group_codes), chunk_size):
        stop = min(start + chunk_size, len(cell_group_codes))
        chunk_codes = cell_group_codes[start:stop]
        retained = chunk_codes >= 0
        if not retained.any():
            continue

        raw_chunk = count_matrix[start:stop]  # type: ignore[index]
        count_chunk = sparse.csr_matrix(raw_chunk[retained], dtype=dtype)
        _validate_raw_counts(count_chunk, start=start)

        retained_codes = chunk_codes[retained]
        membership = sparse.csr_matrix(
            (
                np.ones(len(retained_codes), dtype=dtype),
                (
                    retained_codes,
                    np.arange(len(retained_codes), dtype=np.int64),
                ),
            ),
            shape=(n_groups, len(retained_codes)),
        )
        result = result + membership @ count_chunk

    result.eliminate_zeros()
    return result.tocsr()


def _validate_raw_counts(
    count_chunk: sparse.csr_matrix,
    *,
    start: int,
) -> None:
    """Reject normalized, negative, or non-finite configured count values."""
    values = count_chunk.data
    if values.size == 0:
        return
    if not np.isfinite(values).all():
        raise ValueError(
            "Configured count layer contains non-finite values near "
            f"row {start}."
        )
    if (values < 0).any():
        raise ValueError(
            f"Configured count layer contains negative values near row {start}."
        )
    if (
        np.issubdtype(values.dtype, np.floating)
        and not np.equal(
            values,
            np.floor(values),
        ).all()
    ):
        raise ValueError(
            "Configured count layer contains non-integer values; expected raw "
            f"counts (first offending chunk starts at row {start})."
        )


def _unique_non_missing_value(values: pd.Series) -> object:
    """Return a group value only when it is uniquely defined."""
    observed = pd.unique(values.dropna())
    return observed[0] if len(observed) == 1 else pd.NA


def _persistable_metadata(
    values: list[object],
    *,
    source: pd.Series,
) -> pd.Series | pd.Categorical:
    """Preserve metadata semantics in an AnnData-writable dtype."""
    if pd.api.types.is_numeric_dtype(source.dtype):
        return pd.Series(
            [
                np.nan if cast(bool, pd.isna(value)) else value
                for value in values
            ],
            dtype=np.float64,
        )
    if isinstance(source.dtype, pd.CategoricalDtype):
        return pd.Categorical(
            values,
            categories=source.cat.categories,
            ordered=source.cat.ordered,
        )
    return pd.Categorical(values)
