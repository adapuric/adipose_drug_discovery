"""Paired donor-level adipose rescue signatures."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import anndata as ad  # type: ignore[import]
import numpy as np
import pandas as pd
import pylimma  # type: ignore[import]
from scipy import sparse  # type: ignore[import]
from scipy import stats  # type: ignore[import]


def rescue_vector(
    rescue_results: pd.DataFrame,
    *,
    state: str,
    value_col: str = "moderated_t",
) -> pd.Series:
    """Return the finite tested rescue vector for one adipocyte state.

    The default is the moderated t statistic, ``delta_rescue(c)``. Genes that
    did not pass ``filterByExpr`` or have a non-finite statistic are omitted,
    so downstream similarity calculations never silently consume missing
    values.

    Args:
      rescue_results: Combined per-gene rescue results.
      state: Adipocyte state to extract.
      value_col: Statistic used as the vector value.

    Returns:
      Series indexed by unique gene identifiers in source order.

    Raises:
      KeyError: If a required result column is absent.
      ValueError: If the state is absent or its genes are not unique.
    """
    required = ["gene", "state", "tested", value_col]
    if missing := [
        column for column in required if column not in rescue_results
    ]:
        raise KeyError(f"Rescue results are missing columns: {missing}")

    state_mask = rescue_results["state"].eq(state).fillna(False)
    state_results = rescue_results.loc[state_mask].copy()
    if state_results.empty:
        raise ValueError(f"Rescue results contain no state {state!r}.")
    if state_results["gene"].isna().any():
        raise ValueError(f"State {state!r} contains missing gene identifiers.")
    if state_results["gene"].duplicated().any():
        raise ValueError(f"State {state!r} contains duplicate genes.")

    tested_values = cast(pd.Series, state_results["tested"])
    if pd.api.types.is_bool_dtype(tested_values.dtype):
        tested = tested_values.astype(bool)
    else:
        tested = (
            tested_values.astype(str)
            .str.upper()
            .map({"TRUE": True, "FALSE": False})
        )
        if tested.isna().any():
            raise ValueError("Rescue results contain invalid tested flags.")

    values = cast(
        pd.Series,
        pd.to_numeric(state_results[value_col], errors="coerce"),
    )
    keep = tested.to_numpy(dtype=bool) & np.isfinite(values.to_numpy())
    return pd.Series(
        values.loc[keep].to_numpy(dtype=float),
        index=pd.Index(
            state_results.loc[keep, "gene"].astype(str),
            name="gene",
        ),
        name=value_col,
    )


def select_matched_pairs(
    pseudobulk: ad.AnnData,
    *,
    state: str,
    donor_col: str = "Donor",
    condition_col: str = "condition",
    state_col: str = "cell_state_t2d",
    baseline_label: str = "baseline",
    weightloss_label: str = "weightloss",
) -> ad.AnnData:
    """Select donors represented exactly once in each paired condition.

    Ambiguous donors with duplicate rows in either condition are excluded
    rather than averaged. Rows are returned as baseline then weightloss for
    each donor, which makes the sign and sample order passed to R explicit.

    Args:
      pseudobulk: Donor-level pseudobulk counts in ``X``.
      state: Adipocyte state to analyze, including ``AD_ALL`` when pooled.
      donor_col: Biological-replicate column in ``pseudobulk.obs``.
      condition_col: Condition column in ``pseudobulk.obs``.
      state_col: Adipocyte-state column in ``pseudobulk.obs``.
      baseline_label: Reference condition for the paired contrast.
      weightloss_label: Post-surgery condition for the paired contrast.

    Returns:
      A copied AnnData ordered donor-by-donor as baseline, weightloss.
    """
    if baseline_label == weightloss_label:
        raise ValueError("Baseline and weightloss labels must differ.")
    required = [donor_col, condition_col, state_col]
    if missing := [
        column for column in required if column not in pseudobulk.obs
    ]:
        raise KeyError(f"Pseudobulk obs is missing columns: {missing}")

    obs = cast(pd.DataFrame, pseudobulk.obs)
    state_mask = obs[state_col].eq(state).fillna(False).to_numpy(dtype=bool)
    condition_mask = (
        obs[condition_col]
        .isin([baseline_label, weightloss_label])
        .to_numpy(dtype=bool)
    )
    donor_mask = obs[donor_col].notna().to_numpy(dtype=bool)
    candidate_positions = np.flatnonzero(
        state_mask & condition_mask & donor_mask
    )
    candidates = obs.iloc[candidate_positions]

    eligible_donors: list[object] = []
    for donor in pd.unique(candidates[donor_col]):
        donor_conditions = candidates.loc[
            candidates[donor_col] == donor,
            condition_col,
        ]
        if (
            int((donor_conditions == baseline_label).sum()) == 1
            and int((donor_conditions == weightloss_label).sum()) == 1
            and len(donor_conditions) == 2
        ):
            eligible_donors.append(donor)

    eligible_donors.sort(key=lambda value: str(value))
    ordered_positions: list[int] = []
    for donor in eligible_donors:
        for condition in (baseline_label, weightloss_label):
            match = (
                (obs[donor_col] == donor)
                & (obs[condition_col] == condition)
                & (obs[state_col] == state)
            ).fillna(False)
            positions = np.flatnonzero(match.to_numpy(dtype=bool))
            if len(positions) != 1:
                raise RuntimeError(
                    "Pair selection invariant failed for "
                    f"donor={donor!r}, condition={condition!r}, "
                    f"state={state!r}."
                )
            ordered_positions.append(int(positions[0]))

    paired = pseudobulk[ordered_positions].copy()
    paired.uns["paired_rescue"] = {
        "state": state,
        "baseline_label": baseline_label,
        "weightloss_label": weightloss_label,
        "n_pairs": len(eligible_donors),
        "donors": [str(donor) for donor in eligible_donors],
    }
    return paired


def run_paired_limma(
    paired_pseudobulk: ad.AnnData,
    *,
    donor_col: str = "Donor",
    condition_col: str = "condition",
    baseline_label: str = "baseline",
    weightloss_label: str = "weightloss",
) -> pd.DataFrame:
    """Run filtering, TMM, and paired voom/limma in native Python.

    The result retains every input gene. Genes that do not pass expression
    filtering have `tested=False` and missing model statistics.

    Args:
      paired_pseudobulk: Explicitly ordered donor pairs with raw sums in ``X``.
      donor_col: Donor column in ``paired_pseudobulk.obs``.
      condition_col: Condition column in ``paired_pseudobulk.obs``.
      baseline_label: Reference level for the limma coefficient.
      weightloss_label: Contrast level; positive values mean
        weightloss-minus-baseline.

    Returns:
      Per-gene limma statistics in original gene order.
    """
    paired_obs = cast(pd.DataFrame, paired_pseudobulk.obs)
    n_pairs = _validate_explicit_pairs(
        paired_obs,
        donor_col=donor_col,
        condition_col=condition_col,
        baseline_label=baseline_label,
        weightloss_label=weightloss_label,
    )
    if n_pairs < 2:
        raise ValueError(
            "Paired voom/limma requires at least two matched donors; "
            f"received {n_pairs}."
        )
    if paired_pseudobulk.n_vars == 0:
        raise ValueError("Paired pseudobulk contains no genes.")
    if not paired_pseudobulk.var_names.is_unique:
        raise ValueError("Gene identifiers must be unique for paired limma.")
    if not paired_pseudobulk.obs_names.is_unique:
        raise ValueError("Pseudobulk sample identifiers must be unique.")

    counts = _dense_raw_counts(paired_pseudobulk.X)
    if counts.shape != paired_pseudobulk.shape:
        raise ValueError("Pseudobulk count matrix shape is inconsistent.")

    genes = paired_pseudobulk.var_names.astype(str).tolist()
    gene_counts = counts.T.astype(np.float64, copy=False)
    design = _paired_design(n_pairs)
    tested = _filter_by_expression(gene_counts, design)
    result = pd.DataFrame(
        {
            "gene": genes,
            "tested": tested,
            "logFC": np.nan,
            "moderated_t": np.nan,
            "p_value": np.nan,
            "adjusted_p_value": np.nan,
        }
    )
    if not tested.any():
        return result

    tested_counts = gene_counts[tested]
    effective_library_sizes = _tmm_effective_library_sizes(tested_counts)
    voom_result = pylimma.voom(
        tested_counts,
        design=design,
        lib_size=effective_library_sizes,
        normalize_method="none",
        span=0.5,
        adaptive_span=False,
        plot=False,
    )
    if voom_result is None:
        raise RuntimeError("Native voom returned no result for array input.")
    voom_expression = np.asarray(voom_result["E"], dtype=np.float64)
    voom_weights = np.asarray(voom_result["weights"], dtype=np.float64)
    fit = pylimma.lm_fit(
        voom_expression,
        design=design,
        weights=voom_weights,
    )
    if fit is None:
        raise RuntimeError("Native limma returned no fit for array input.")
    # Exact within-pair equality has a zero condition estimand. Stabilize QR
    # roundoff before empirical-Bayes moderation can amplify it.
    unchanged = np.all(
        voom_expression[:, 0::2] == voom_expression[:, 1::2],
        axis=1,
    )
    coefficients = np.asarray(fit["coefficients"], dtype=np.float64)
    coefficients[unchanged, -1] = 0.0
    fit["coefficients"] = coefficients
    moderated_fit = pylimma.e_bayes(fit)
    if moderated_fit is None:
        raise RuntimeError("Native empirical Bayes returned no fit.")
    table = pylimma.top_table(
        moderated_fit,
        coef=design.shape[1] - 1,
        number=tested_counts.shape[0],
        adjust_method="BH",
        sort_by="none",
    )
    if len(table) != int(tested.sum()):
        raise RuntimeError(
            "Paired voom/limma did not return every tested gene."
        )
    result.loc[tested, "logFC"] = table["log_fc"].to_numpy(dtype=float)
    result.loc[tested, "moderated_t"] = table["t"].to_numpy(dtype=float)
    result.loc[tested, "p_value"] = table["p_value"].to_numpy(dtype=float)
    result.loc[tested, "adjusted_p_value"] = table["adj_p_value"].to_numpy(
        dtype=float
    )
    return result


def estimate_rescue(
    pseudobulk: ad.AnnData,
    *,
    state: str,
    donor_col: str = "Donor",
    condition_col: str = "condition",
    state_col: str = "cell_state_t2d",
    baseline_label: str = "baseline",
    weightloss_label: str = "weightloss",
) -> pd.DataFrame:
    """Estimate one matched baseline-to-weightloss rescue vector.

    ``delta_rescue`` is the moderated t statistic by default. Positive
    ``delta_rescue`` and ``logFC`` values denote higher expression after
    weight loss than at baseline.

    Args:
      pseudobulk: State and pooled donor-level raw-count profiles.
      state: State label to estimate.
      donor_col: Biological-replicate column.
      condition_col: Condition column.
      state_col: State column.
      baseline_label: Reference condition.
      weightloss_label: Post-surgery contrast condition.

    Returns:
      Per-gene rescue table with state and matched-donor count.
    """
    paired = select_matched_pairs(
        pseudobulk,
        state=state,
        donor_col=donor_col,
        condition_col=condition_col,
        state_col=state_col,
        baseline_label=baseline_label,
        weightloss_label=weightloss_label,
    )
    n_pairs = paired.n_obs // 2
    if n_pairs < 2:
        raise ValueError(
            f"State {state!r} has {n_pairs} valid donor pair(s); "
            "at least two are required for paired voom/limma."
        )

    result = run_paired_limma(
        paired,
        donor_col=donor_col,
        condition_col=condition_col,
        baseline_label=baseline_label,
        weightloss_label=weightloss_label,
    )
    result.insert(1, "state", state)
    result.insert(2, "n_pairs", n_pairs)
    moderated_t = cast(pd.Series, result["moderated_t"])
    result.insert(6, "delta_rescue", moderated_t)
    return result


def estimate_all_rescue_vectors(
    pseudobulk: ad.AnnData,
    *,
    states: Sequence[str] | None = None,
    donor_col: str = "Donor",
    condition_col: str = "condition",
    state_col: str = "cell_state_t2d",
    baseline_label: str = "baseline",
    weightloss_label: str = "weightloss",
) -> pd.DataFrame:
    """Estimate paired rescue vectors for every requested adipocyte state."""
    obs = cast(pd.DataFrame, pseudobulk.obs)
    if state_col not in obs:
        raise KeyError(f"Pseudobulk obs is missing state column {state_col!r}.")
    if states is None:
        state_series = cast(pd.Series, obs[state_col])
        state_values = [
            str(value) for value in pd.unique(state_series.dropna())
        ]
    else:
        state_values = list(states)
    if not state_values:
        raise ValueError(
            "No adipocyte states were provided for rescue analysis."
        )

    results = [
        estimate_rescue(
            pseudobulk,
            state=state,
            donor_col=donor_col,
            condition_col=condition_col,
            state_col=state_col,
            baseline_label=baseline_label,
            weightloss_label=weightloss_label,
        )
        for state in state_values
    ]
    return pd.concat(results, axis=0, ignore_index=True)


def _validate_explicit_pairs(
    obs: pd.DataFrame,
    *,
    donor_col: str,
    condition_col: str,
    baseline_label: str,
    weightloss_label: str,
) -> int:
    """Validate one ordered baseline/weightloss row per donor."""
    if missing := [
        column for column in (donor_col, condition_col) if column not in obs
    ]:
        raise KeyError(f"Paired pseudobulk obs is missing columns: {missing}")
    if len(obs) % 2:
        raise ValueError("Paired pseudobulk must contain two rows per donor.")

    expected_conditions = [baseline_label, weightloss_label] * (len(obs) // 2)
    observed_conditions = obs[condition_col].astype(str).tolist()
    if observed_conditions != expected_conditions:
        raise ValueError(
            "Paired rows must be explicitly ordered baseline then weightloss "
            "for every donor."
        )
    donors = obs[donor_col].astype(str).tolist()
    for index in range(0, len(donors), 2):
        if donors[index] != donors[index + 1]:
            raise ValueError(
                "Adjacent baseline and weightloss rows must share a donor."
            )
    if len(set(donors[::2])) != len(donors) // 2:
        raise ValueError("Each biological donor must appear in one pair only.")
    return len(donors) // 2


def _dense_raw_counts(matrix: object) -> np.ndarray:
    """Materialize the small donor-level matrix and validate raw sums."""
    # scipy.sparse.issparse is not typed as a TypeGuard.
    counts = (
        matrix.toarray()  # type: ignore[attr-defined]
        if sparse.issparse(matrix)
        else np.asarray(matrix)
    )

    if not np.issubdtype(counts.dtype, np.number):
        raise ValueError("Pseudobulk counts must be numeric.")
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Pseudobulk counts must be finite and non-negative.")
    if (
        np.issubdtype(counts.dtype, np.floating)
        and not np.equal(
            counts,
            np.floor(counts),
        ).all()
    ):
        raise ValueError("Pseudobulk input must contain summed raw counts.")
    if (counts.sum(axis=1) == 0).any():
        raise ValueError(
            "Every pseudobulk sample must have a positive library."
        )
    return counts


def _paired_design(n_pairs: int) -> np.ndarray:
    """Return the donor-blocked weightloss-minus-baseline design matrix."""
    n_samples = n_pairs * 2
    design = np.zeros((n_samples, n_pairs + 1), dtype=np.float64)
    design[:, 0] = 1.0
    for donor_index in range(1, n_pairs):
        design[2 * donor_index : 2 * donor_index + 2, donor_index] = 1.0
    design[1::2, -1] = 1.0
    return design


def _filter_by_expression(
    counts: np.ndarray,
    design: np.ndarray,
    *,
    min_count: float = 10.0,
    min_total_count: float = 15.0,
    large_n: float = 10.0,
    min_prop: float = 0.7,
) -> np.ndarray:
    """Return the edgeR-compatible expression filter for a design matrix."""
    library_sizes = counts.sum(axis=0)
    design_q, _ = np.linalg.qr(design, mode="reduced")
    leverages = np.square(design_q).sum(axis=1)
    minimum_sample_size = 1.0 / leverages.max()
    if minimum_sample_size > large_n:
        minimum_sample_size = (
            large_n + (minimum_sample_size - large_n) * min_prop
        )

    median_library_size = float(np.median(library_sizes))
    cpm_cutoff = min_count / median_library_size * 1e6
    cpm = counts / library_sizes[np.newaxis, :] * 1e6
    tolerance = 1e-14
    keep_cpm = (
        np.sum(cpm >= cpm_cutoff, axis=1) >= minimum_sample_size - tolerance
    )
    keep_total = counts.sum(axis=1) >= min_total_count - tolerance
    return keep_cpm & keep_total


def _tmm_effective_library_sizes(counts: np.ndarray) -> np.ndarray:
    """Return edgeR-compatible TMM effective library sizes."""
    library_sizes = counts.sum(axis=0)
    nonzero_counts = counts[np.any(counts > 0, axis=1)]
    if nonzero_counts.shape[0] == 0 or nonzero_counts.shape[1] == 1:
        return library_sizes.astype(np.float64, copy=False)

    quantiles = np.quantile(nonzero_counts, 0.75, axis=0)
    relative_quantiles = quantiles / library_sizes
    if float(np.median(relative_quantiles)) < 1e-20:
        reference_index = int(np.argmax(np.sqrt(nonzero_counts).sum(axis=0)))
    else:
        reference_index = int(
            np.argmin(np.abs(relative_quantiles - relative_quantiles.mean()))
        )

    reference = nonzero_counts[:, reference_index]
    factors = np.array(
        [
            _tmm_factor(
                nonzero_counts[:, sample_index],
                reference,
                observed_library_size=float(library_sizes[sample_index]),
                reference_library_size=float(library_sizes[reference_index]),
            )
            for sample_index in range(nonzero_counts.shape[1])
        ]
    )
    factors /= np.exp(np.mean(np.log(factors)))
    return library_sizes * factors


def _tmm_factor(
    observed: np.ndarray,
    reference: np.ndarray,
    *,
    observed_library_size: float,
    reference_library_size: float,
    logratio_trim: float = 0.3,
    sum_trim: float = 0.05,
    a_cutoff: float = -1e10,
) -> float:
    """Calculate one weighted trimmed mean of M-values factor."""
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log2(
            (observed / observed_library_size)
            / (reference / reference_library_size)
        )
        average_expression = (
            np.log2(observed / observed_library_size)
            + np.log2(reference / reference_library_size)
        ) / 2.0
        variance = (
            observed_library_size - observed
        ) / observed_library_size / observed + (
            reference_library_size - reference
        ) / reference_library_size / reference

    finite = (
        np.isfinite(log_ratio)
        & np.isfinite(average_expression)
        & (average_expression > a_cutoff)
    )
    log_ratio = log_ratio[finite]
    average_expression = average_expression[finite]
    variance = variance[finite]
    if log_ratio.size == 0 or np.max(np.abs(log_ratio)) < 1e-6:
        return 1.0

    n_values = log_ratio.size
    lower_logratio_rank = np.floor(n_values * logratio_trim) + 1
    upper_logratio_rank = n_values + 1 - lower_logratio_rank
    lower_sum_rank = np.floor(n_values * sum_trim) + 1
    upper_sum_rank = n_values + 1 - lower_sum_rank
    logratio_ranks = stats.rankdata(log_ratio, method="average")
    sum_ranks = stats.rankdata(average_expression, method="average")
    keep = (
        (logratio_ranks >= lower_logratio_rank)
        & (logratio_ranks <= upper_logratio_rank)
        & (sum_ranks >= lower_sum_rank)
        & (sum_ranks <= upper_sum_rank)
    )
    weights = 1.0 / variance[keep]
    with np.errstate(divide="ignore", invalid="ignore"):
        weighted_mean = np.sum(log_ratio[keep] * weights) / np.sum(weights)
    if not np.isfinite(weighted_mean):
        weighted_mean = 0.0
    return float(2.0**weighted_mean)
