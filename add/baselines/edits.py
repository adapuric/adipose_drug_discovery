"""Naive single-gene edit predictors for comparison with model sensitivity.

Neither predictor uses an experimental edit outcome. The co-expression
contrast asks whether a model edit only reproduces how nuclei lacking the
edited gene differ from those expressing it. The generic-edit mean ignores
edited-gene identity, analogous to the train-mean drug baseline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import anndata as ad
import anndata.abc as ad_abc
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from add.scoring import score_mimicry


def coexpression_edit_effects(
    path: Path,
    *,
    edited_genes: Mapping[str, Sequence[str]],
    condition: str,
    donor_col: str = "Donor",
    state_col: str = "cell_states_adipocytes",
    condition_col: str = "condition",
    minimum_nuclei: int = 20,
    target_sum: float = 1_000_000,
    chunk_size: int = 4096,
) -> pd.DataFrame:
    """Contrast nuclei lacking each edit's genes with nuclei expressing them.

    Raw counts in `X` are summed separately for nuclei with and without any
    detected edited gene within each donor and state of `condition`. Each sum
    is normalized to log1p CPM by its own library size before taking
    undetected minus detected. The matrix is read in row chunks; memory scales
    with edits x donor/state groups x genes.

    Args:
      path: H5AD holding the raw counts the model was trained on.
      edited_genes: Edit ID mapped to the gene IDs removed by that edit.
      condition: Condition label of the nuclei whose sequence was edited.
      donor_col: Independent biological-replicate column.
      state_col: Adipocyte-state column.
      condition_col: Condition column containing `condition`.
      minimum_nuclei: Minimum nuclei required in each stratum.
      target_sum: Library size used for CPM normalization.
      chunk_size: Number of nuclei read per chunk.

    Returns:
      Long table keyed by edit, donor, state and gene. `delta` is missing with
      status "insufficient_nuclei" when either stratum lacks support.
    """
    if not edited_genes or chunk_size < 1 or minimum_nuclei < 1:
        raise ValueError(
            "Edits, positive chunk_size and minimum_nuclei are required"
        )
    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError("target_sum must be positive and finite")

    with h5py.File(path, "r") as handle:
        obs = cast(
            pd.DataFrame, ad.io.read_elem(cast(h5py.Group, handle["obs"]))
        )
        var = cast(
            pd.DataFrame, ad.io.read_elem(cast(h5py.Group, handle["var"]))
        )
        genes = pd.Index(var.index.astype(str), name="gene_id")
        codes, groups = _condition_groups(
            obs,
            condition=condition,
            donor_col=donor_col,
            state_col=state_col,
            condition_col=condition_col,
        )
        positions = _edited_positions(edited_genes, genes=genes, path=path)
        matrix = _count_matrix(handle, shape=(len(obs), len(genes)))
        totals, detected, n_total, n_detected = _sum_detection_strata(
            matrix,
            codes=codes,
            n_groups=len(groups),
            positions=positions,
            chunk_size=chunk_size,
        )

    tables: list[pd.DataFrame] = []
    for edit_position, edit_id in enumerate(edited_genes):
        absent = totals - detected[edit_position]
        n_absent = n_total - n_detected[edit_position]
        supported = (n_absent >= minimum_nuclei) & (
            n_detected[edit_position] >= minimum_nuclei
        )
        delta = _log1p_cpm(absent, target_sum=target_sum) - _log1p_cpm(
            detected[edit_position], target_sum=target_sum
        )
        delta[~supported] = np.nan

        tables.append(
            pd.DataFrame(
                {
                    "edit_id": edit_id,
                    "donor_id": np.repeat(groups[donor_col], len(genes)),
                    "state": np.repeat(groups[state_col], len(genes)),
                    "gene_id": np.tile(genes, len(groups)),
                    "delta": delta.reshape(-1),
                    "is_edited_gene": np.tile(
                        genes.isin(edited_genes[edit_id]), len(groups)
                    ),
                    "n_detected_nuclei": np.repeat(
                        n_detected[edit_position], len(genes)
                    ),
                    "n_undetected_nuclei": np.repeat(n_absent, len(genes)),
                    "status": np.repeat(
                        np.where(supported, "ok", "insufficient_nuclei"),
                        len(genes),
                    ),
                }
            )
        )

    result = pd.concat(tables, ignore_index=True)
    result["model"] = "coexpression-contrast"
    result["scale"] = "log1p_cpm_undetected_minus_detected"
    result["contrast_condition"] = condition
    return result


def generic_edit_effects(
    effects: pd.DataFrame,
    *,
    minimum_reference_edits: int = 2,
) -> pd.DataFrame:
    """Predict each edit by the mean model effect of the other edits.

    Use one model, edit mode, edited sequence and effect scale per call.
    Values for a gene that another edit removed directly are not used as
    downstream reference effects.

    Args:
      effects: Model rows with `edit_id`, `donor_id`, `state`, `gene_id`,
        `delta` and `is_edited_gene`.
      minimum_reference_edits: Minimum other edits required per value.

    Returns:
      Leave-one-edit-out mean effects with `n_reference_edits` and status.
    """
    if minimum_reference_edits < 1:
        raise ValueError("minimum_reference_edits must be positive")

    keys = ["donor_id", "state", "gene_id"]
    if effects.duplicated(["edit_id", *keys]).any():
        raise ValueError("Model effects require unique edit/donor/state/gene")

    usable = np.isfinite(effects["delta"]) & ~effects["is_edited_gene"]
    own = effects["delta"].where(usable, 0.0)
    totals = (
        effects.assign(own=own, usable=usable.astype(int))
        .groupby(keys, sort=False)[["own", "usable"]]
        .transform("sum")
    )
    n_reference = totals["usable"] - usable.astype(int)
    supported = n_reference >= minimum_reference_edits
    reference_sum = totals["own"] - own

    result = effects[["edit_id", *keys, "is_edited_gene"]].copy()
    result["delta"] = (reference_sum / n_reference).where(supported)
    result["n_reference_edits"] = n_reference
    result["status"] = np.where(supported, "ok", "insufficient_reference_edits")
    result["model"] = "generic-edit-mean"
    return result


def edit_effect_agreement(
    model: pd.DataFrame,
    comparator: pd.DataFrame,
    *,
    minimum_nuclei: int = 20,
    minimum_reference_cpm: float = 1.0,
    minimum_shared_genes: int = 50,
) -> pd.DataFrame:
    """Compare model and naive downstream edit effects per donor and state.

    Genes edited directly are excluded. Evaluated genes must reach
    `minimum_reference_cpm` in the model's unedited reference profile for that
    donor and state. Correlations describe agreement, not accuracy.

    Args:
      model: Model rows with `edit_id`, `donor_id`, `state`, `gene_id`,
        `delta`, `is_edited_gene`, `n_nuclei` and `reference_count`.
      comparator: Naive rows with the same keys, `delta` and `status`.
      minimum_nuclei: Minimum model nuclei required per donor and state.
      minimum_reference_cpm: Minimum unedited reference expression.
      minimum_shared_genes: Minimum finite genes required for correlation.

    Returns:
      One row per model edit, donor and state with Pearson and Spearman
      agreement, root-mean-square effect sizes and an estimability status.
    """
    keys = ["edit_id", "donor_id", "state"]
    aligned = _aligned_downstream_effects(
        model,
        comparator,
        keys=keys,
        minimum_reference_cpm=minimum_reference_cpm,
    )
    comparator_status = (
        comparator.groupby(keys, sort=False)["status"]
        .agg(lambda status: "ok" if status.eq("ok").any() else status.iloc[0])
        .to_dict()
    )
    groups = dict(tuple(aligned.groupby(keys, sort=False)))

    rows: list[dict[str, object]] = []
    for group_key, n_nuclei in (
        model.groupby(keys, sort=True)["n_nuclei"].first().items()
    ):
        key = cast(tuple[str, str, str], group_key)
        rows.append(
            _agreement_row(
                groups.get(key, aligned.iloc[:0]),
                key=key,
                n_nuclei=int(n_nuclei),
                support=str(comparator_status.get(key, "missing_comparator")),
                minimum_nuclei=minimum_nuclei,
                minimum_shared_genes=minimum_shared_genes,
            )
        )

    return pd.DataFrame(rows)


def _aligned_downstream_effects(
    model: pd.DataFrame,
    comparator: pd.DataFrame,
    *,
    keys: list[str],
    minimum_reference_cpm: float,
) -> pd.DataFrame:
    """Pair model and comparator deltas on expressed, unedited genes."""
    library = model.groupby(keys, sort=False)["reference_count"].transform(
        "sum"
    )
    reference_cpm = model["reference_count"] / library.where(library > 0) * 1e6
    evaluated = model.loc[
        ~model["is_edited_gene"] & (reference_cpm >= minimum_reference_cpm)
    ]

    aligned = evaluated[[*keys, "gene_id", "delta"]].merge(
        comparator[[*keys, "gene_id", "delta"]],
        on=[*keys, "gene_id"],
        how="left",
        suffixes=("", "_comparator"),
        validate="one_to_one",
    )
    aligned["delta_comparator"] = aligned["delta_comparator"].astype(float)
    return aligned


def _agreement_row(
    group: pd.DataFrame,
    *,
    key: tuple[str, str, str],
    n_nuclei: int,
    support: str,
    minimum_nuclei: int,
    minimum_shared_genes: int,
) -> dict[str, object]:
    """Correlate one edit's model and naive effects in one donor and state.

    Insufficient model nuclei take precedence over comparator support, which
    takes precedence over the correlation's own status.
    """
    finite = np.isfinite(group["delta"]) & np.isfinite(
        group["delta_comparator"]
    )
    predicted = group.loc[finite, "delta"].to_numpy(dtype=float)
    naive = group.loc[finite, "delta_comparator"].to_numpy(dtype=float)
    genes = group.loc[finite, "gene_id"].astype(str).tolist()
    score = score_mimicry(
        predicted,
        genes,
        naive,
        genes,
        minimum_shared_genes=minimum_shared_genes,
    )

    status = score.status
    if n_nuclei < minimum_nuclei:
        status = "insufficient_model_nuclei"
    elif support != "ok":
        status = support
    estimable = status == "ok"

    return {
        "edit_id": key[0],
        "donor_id": key[1],
        "state": key[2],
        "n_nuclei": n_nuclei,
        "n_genes": len(genes),
        "agreement_pearson": score.pearson if estimable else np.nan,
        "agreement_spearman": score.spearman if estimable else np.nan,
        "model_rms": _root_mean_square(predicted),
        "comparator_rms": _root_mean_square(naive),
        "comparator_status": support,
        "status": status,
    }


def _condition_groups(
    obs: pd.DataFrame,
    *,
    condition: str,
    donor_col: str,
    state_col: str,
    condition_col: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Assign nuclei in the edited condition to donor/state groups."""
    if missing := {donor_col, state_col, condition_col} - set(obs):
        raise KeyError(f"Expression H5AD lacks obs columns: {sorted(missing)}")
    labels = obs[condition_col].astype(str)
    if condition not in set(labels):
        raise ValueError(
            f"Condition {condition!r} is absent from {condition_col}; "
            f"available: {sorted(set(labels))}"
        )
    selected = labels.eq(condition).to_numpy()
    if obs.loc[selected, [donor_col, state_col]].isna().any(axis=None):
        raise ValueError("Edited-condition nuclei lack donor or state labels")

    keys = pd.MultiIndex.from_frame(
        obs.loc[selected, [donor_col, state_col]].astype(str)
    )
    selected_codes, unique_keys = pd.factorize(keys, sort=True)
    codes = np.full(len(obs), -1, dtype=np.int64)
    codes[selected] = selected_codes
    groups = cast(pd.MultiIndex, unique_keys).to_frame(index=False)
    groups.columns = [donor_col, state_col]
    return codes, groups


def _edited_positions(
    edited_genes: Mapping[str, Sequence[str]],
    *,
    genes: pd.Index,
    path: Path,
) -> list[np.ndarray]:
    """Locate each edit's genes in the expression gene axis."""
    if not genes.is_unique:
        raise ValueError(f"Expression gene IDs are not unique: {path}")
    positions: list[np.ndarray] = []
    for edit_id, edit_genes in edited_genes.items():
        missing = set(edit_genes) - set(genes)
        if not edit_genes or missing:
            raise ValueError(
                f"Edit {edit_id} genes are absent from {path}: "
                f"{sorted(missing) or 'none listed'}"
            )
        positions.append(genes.get_indexer(pd.Index(list(edit_genes))))
    return positions


def _count_matrix(
    handle: h5py.File,
    *,
    shape: tuple[int, int],
) -> h5py.Dataset | ad_abc.CSRDataset | ad_abc.CSCDataset:
    """Open `X` without loading it, as sparse or dense storage."""
    node = handle["X"]
    if isinstance(node, h5py.Group):
        matrix = ad.io.sparse_dataset(node)
    elif isinstance(node, h5py.Dataset):
        matrix = node
    else:
        raise TypeError("Unsupported X encoding in expression H5AD")
    if matrix.shape != shape:
        raise ValueError("Expression counts and metadata are misaligned")
    return matrix


def _sum_detection_strata(
    matrix: h5py.Dataset | ad_abc.CSRDataset | ad_abc.CSCDataset,
    *,
    codes: np.ndarray,
    n_groups: int,
    positions: Sequence[np.ndarray],
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sum raw counts per group, overall and for nuclei detecting each edit."""
    n_genes = matrix.shape[1]
    totals = np.zeros((n_groups, n_genes))
    detected = np.zeros((len(positions), n_groups, n_genes))
    n_total = np.zeros(n_groups, dtype=np.int64)
    n_detected = np.zeros((len(positions), n_groups), dtype=np.int64)
    for start in range(0, len(codes), chunk_size):
        chunk_codes = codes[start : start + chunk_size]
        retained = chunk_codes >= 0
        if not retained.any():
            continue
        block = sp.csr_matrix(matrix[start : start + chunk_size])[retained]
        _validate_raw_counts(block.data, start=start)
        chunk_codes = chunk_codes[retained]

        membership = sp.csr_matrix(
            (
                np.ones(len(chunk_codes)),
                (chunk_codes, np.arange(len(chunk_codes))),
            ),
            shape=(n_groups, len(chunk_codes)),
        )
        totals += (membership @ block).toarray()
        n_total += np.bincount(chunk_codes, minlength=n_groups)

        for edit_position, columns in enumerate(positions):
            hits = np.asarray((block[:, columns] > 0).sum(axis=1)).ravel() > 0
            detected[edit_position] += (
                membership[:, hits] @ block[hits]
            ).toarray()
            n_detected[edit_position] += np.bincount(
                chunk_codes[hits], minlength=n_groups
            )
    return totals, detected, n_total, n_detected


def _validate_raw_counts(values: np.ndarray, *, start: int) -> None:
    """Reject normalized, regressed, negative or non-finite counts."""
    if values.size and (
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.equal(values, np.floor(values)).all()
    ):
        raise ValueError(
            "Expression X must hold raw integer counts; found other values "
            f"near row {start}. Use an unregressed input with the same nuclei."
        )


def _log1p_cpm(counts: np.ndarray, *, target_sum: float) -> np.ndarray:
    """Normalize each profile to log1p CPM, keeping empty libraries missing."""
    library = counts.sum(axis=1, keepdims=True)
    scale = np.divide(
        target_sum,
        library,
        out=np.full_like(library, np.nan, dtype=float),
        where=library > 0,
    )
    return np.log1p(counts * scale)


def _root_mean_square(values: np.ndarray) -> float:
    """Return effect magnitude, missing when no value is available."""
    return float(np.sqrt(np.mean(values**2))) if values.size else np.nan
