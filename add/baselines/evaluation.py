"""Held-out context splitting, prediction accuracy and summaries."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil

import numpy as np
import pandas as pd

from add.baselines.scoring import _mimic_fields
from add.baselines.signatures import _context_columns
from add.baselines.signatures import _context_keys
from add.baselines.signatures import _root_mean_squared_error
from add.baselines.signatures import _signature_ids
from add.perturb import PerturbSignatures


def split_signature_contexts(
    signatures: PerturbSignatures,
    *,
    context_col: str | Sequence[str],
    test_fraction: float,
    random_seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split signature IDs while keeping each context on only one side."""
    context_cols = _context_columns(context_col)
    if missing := [
        column for column in context_cols if column not in signatures.meta
    ]:
        raise KeyError(
            f"Perturbation metadata lacks context columns: {missing}"
        )
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1.")

    context_values = _context_keys(signatures.meta, context_cols)
    unique_contexts = sorted(set(context_values))
    if len(unique_contexts) < 2:
        raise ValueError(
            "At least two contexts are required for a held-out split."
        )

    n_test = min(
        len(unique_contexts) - 1,
        max(1, ceil(len(unique_contexts) * test_fraction)),
    )
    generator = np.random.default_rng(random_seed)
    shuffled = generator.permutation(len(unique_contexts))
    test_contexts = {
        unique_contexts[index] for index in shuffled[:n_test].tolist()
    }
    signature_ids = _signature_ids(signatures)

    train_ids = tuple(
        signature_id
        for signature_id, context in zip(
            signature_ids,
            context_values,
            strict=True,
        )
        if context not in test_contexts
    )
    test_ids = tuple(
        signature_id
        for signature_id, context in zip(
            signature_ids,
            context_values,
            strict=True,
        )
        if context in test_contexts
    )

    return train_ids, test_ids


def summarize_heldout_comparisons(
    evaluation: pd.DataFrame,
    *,
    comparators: Sequence[str] = ("train_mean", "drug_mean"),
    random_seed: int = 0,
    bootstrap_draws: int = 2000,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Summarize model accuracy gains over comparators by held-out group.

    Signatures from one held-out evaluation group, such as a cell line, are
    not independent, so each group contributes the mean of its paired
    differences and the bootstrap resamples groups. Positive values favour
    the model: comparator RMSE minus model RMSE, and model Pearson minus
    comparator Pearson. A signature is paired only when both predictions
    are estimable.

    Args:
      evaluation: Rows with `evaluation_group`, model `prediction_*` and
        `score_status`, and `<comparator>_rmse`, `_pearson` and `_status`.
      comparators: Column prefixes of the comparator predictions.
      random_seed: Seed for the group bootstrap; comparator `i` uses child
        seed `[random_seed, i]`.
      bootstrap_draws: Number of group resamples.
      confidence_level: Coverage of the percentile interval.

    Returns:
      One row per comparator and metric with the mean over groups, support,
      the fraction of groups favouring the model and a conditional interval.
    """
    if bootstrap_draws < 1 or not 0 < confidence_level < 1:
        raise ValueError("Invalid bootstrap draws or confidence level")

    rows: list[dict[str, object]] = []
    for index, comparator in enumerate(comparators):
        paired = evaluation.loc[
            evaluation["score_status"].eq("ok")
            & evaluation[f"{comparator}_status"].eq("ok")
        ]
        by_group = _group_improvements(paired, comparator=comparator)
        draws = np.random.default_rng([random_seed, index]).integers(
            0, max(len(by_group), 1), size=(bootstrap_draws, len(by_group))
        )

        for metric in ("rmse_improvement", "pearson_improvement"):
            values = by_group[metric].to_numpy(dtype=float)
            lower, upper, interval_status = _group_bootstrap_interval(
                values,
                draws,
                confidence_level=confidence_level,
            )
            rows.append(
                {
                    "comparator": comparator,
                    "metric": metric,
                    "mean": float(values.mean()) if values.size else np.nan,
                    "n_groups": int(values.size),
                    "n_signatures": len(paired),
                    "fraction_groups_favouring_model": (
                        float((values > 0).mean()) if values.size else np.nan
                    ),
                    "lower": lower,
                    "upper": upper,
                    "interval_status": interval_status,
                    "confidence_level": confidence_level,
                }
            )

    return pd.DataFrame(rows)


def _group_improvements(
    paired: pd.DataFrame,
    *,
    comparator: str,
) -> pd.DataFrame:
    """Average model-minus-comparator gains within each held-out group."""
    return (
        pd.DataFrame(
            {
                "evaluation_group": paired["evaluation_group"],
                "rmse_improvement": paired[f"{comparator}_rmse"]
                - paired["prediction_rmse"],
                "pearson_improvement": paired["prediction_pearson"]
                - paired[f"{comparator}_pearson"],
            }
        )
        .groupby("evaluation_group", sort=True)
        .mean()
    )


def _group_bootstrap_interval(
    values: np.ndarray,
    draws: np.ndarray,
    *,
    confidence_level: float,
) -> tuple[float, float, str]:
    """Return a percentile interval of group-resampled means and its status.

    Fewer than three groups, or identical group values, give no interval.
    """
    if values.size < 3:
        return np.nan, np.nan, "insufficient_groups_for_interval"
    if not np.ptp(values) > 0:
        return np.nan, np.nan, "degenerate_group_values"

    tail = (1 - confidence_level) / 2
    lower, upper = np.quantile(values[draws].mean(axis=1), [tail, 1 - tail])
    return lower, upper, "conditional_group_bootstrap"


def _prediction_fields(
    predicted: np.ndarray,
    observed: np.ndarray,
    genes: Sequence[str],
    *,
    minimum_shared_genes: int,
) -> dict[str, object]:
    """Measure one held-out signature prediction against its observation."""
    score = _mimic_fields(
        predicted,
        genes,
        observed,
        genes,
        minimum_shared_genes=minimum_shared_genes,
    )
    return {
        "prediction_pearson": score["score_mimic"],
        "prediction_spearman": score["score_spearman"],
        "prediction_rmse": _root_mean_squared_error(predicted, observed),
        "n_shared_genes": score["n_shared_genes"],
        "score_status": score["score_status"],
    }


def _comparator_fields(
    name: str,
    predicted: np.ndarray,
    observed: np.ndarray,
    genes: Sequence[str],
    *,
    minimum_shared_genes: int,
) -> dict[str, object]:
    """Name one comparator's accuracy against a held-out signature."""
    fields = _prediction_fields(
        predicted,
        observed,
        genes,
        minimum_shared_genes=minimum_shared_genes,
    )
    return {
        f"{name}_pearson": fields["prediction_pearson"],
        f"{name}_spearman": fields["prediction_spearman"],
        f"{name}_status": fields["score_status"],
        f"{name}_rmse": fields["prediction_rmse"],
    }


def _context_score_summary(context_scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize context variability without replacing context-level rows."""
    rows: list[dict[str, object]] = []
    for (state, drug), group in context_scores.groupby(
        ["state", "drug"],
        observed=True,
        sort=True,
        dropna=False,
    ):
        finite = group.loc[np.isfinite(group["score_mimic"]), "score_mimic"]
        rows.append(
            {
                "state": str(state),
                "drug": str(drug),
                "context_score_median": np.nan
                if finite.empty
                else float(finite.median()),
                "context_score_sd": np.nan
                if finite.empty
                else float(finite.std(ddof=0)),
                "context_fraction_positive": np.nan
                if finite.empty
                else float((finite > 0).mean()),
            }
        )
    return pd.DataFrame(rows)
