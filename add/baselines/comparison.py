"""Compare drug rankings from different baselines on matched gene support.

These comparisons are descriptive. They show whether baselines order the
same drugs similarly and whether a drug mimics the rescue more than the
drug-agnostic mean perturbation response; they do not test drug efficacy.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping

import numpy as np
import pandas as pd


def add_train_mean_reference(
    rankings: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Attach each state's drug-agnostic reference score from the same genes.

    `score_minus_train_mean` is positive when a drug mimics the rescue more
    than the drug-agnostic mean response. It is missing when either score is
    non-estimable. Rankings keep their row order.

    Args:
      rankings: Drug rows with `state`, `score` and `score_status`.
      reference: One row per state from the same baseline without drug
        identity, scored on the same genes as `rankings`: `score_train_mean`
        for measured signatures, `score_pca_ridge_reference` for PCA-ridge
        predictions.

    Returns:
      A copy of `rankings` with `train_mean_score` and
      `score_minus_train_mean`.
    """
    if reference["state"].duplicated().any():
        raise ValueError("The train-mean reference needs one row per state")

    states = reference["state"].astype(str)
    reference_score = pd.DataFrame(
        {
            "state": states,
            "train_mean_score": reference["score"].where(
                reference["score_status"].eq("ok")
            ),
        }
    )
    result = rankings.merge(
        reference_score, on="state", how="left", validate="many_to_one"
    )
    drug_score = result["score"].where(result["score_status"].eq("ok"))
    result["score_minus_train_mean"] = drug_score - result["train_mean_score"]
    return result


def compare_drug_rankings(
    rankings: Mapping[str, pd.DataFrame],
    *,
    top_n: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join baselines' drug scores per state and summarize their agreement.

    Rankings are compared as given; scoring them on the same genes and drug
    identities is the caller's responsibility. Agreement uses drugs that both
    baselines score in a state. Ties in the top-`top_n` lists break by drug
    name, and the overlap is divided by the list length actually compared.

    Args:
      rankings: Baseline name mapped to rows with `state`, `drug`, `score`,
        `rank`, `score_status` and `score_minus_train_mean`.
      top_n: Number of top drugs compared per state.

    Returns:
      A wide state-by-drug table with each baseline's score, rank, status
      and train-mean difference, and one agreement row per baseline pair and
      state.
    """
    if len(rankings) < 2 or top_n < 1:
        raise ValueError("Compare at least two rankings with a positive top_n")

    columns = ["score", "rank", "score_status", "score_minus_train_mean"]
    frames: list[pd.DataFrame] = []
    for name, ranking in rankings.items():
        if ranking.duplicated(["state", "drug"]).any():
            raise ValueError(f"{name} ranks a drug twice within a state")
        frames.append(
            ranking[["state", "drug", *columns]].rename(
                columns={column: f"{name}_{column}" for column in columns}
            )
        )

    matched = frames[0]
    for frame in frames[1:]:
        matched = matched.merge(
            frame, on=["state", "drug"], how="outer", validate="1:1"
        )
    matched = matched.sort_values(["state", "drug"], ignore_index=True)

    rows: list[dict[str, object]] = []
    for first, second in itertools.combinations(rankings, 2):
        for state, group in matched.groupby("state", sort=True):
            rows.append(
                {"state": str(state), "baseline_a": first, "baseline_b": second}
                | _pair_agreement(group, first, second, top_n=top_n)
            )
    return matched, pd.DataFrame(rows)


def _pair_agreement(
    matched: pd.DataFrame,
    first: str,
    second: str,
    *,
    top_n: int,
) -> dict[str, object]:
    """Measure how two baselines order the drugs both score in one state."""
    both = matched.loc[
        matched[f"{first}_score_status"].eq("ok")
        & matched[f"{second}_score_status"].eq("ok")
    ]
    status = "ok"
    spearman = np.nan
    if len(both) < 3:
        status = "insufficient_drugs"
    elif (
        both[f"{first}_score"].nunique() < 2
        or both[f"{second}_score"].nunique() < 2
    ):
        status = "constant_scores"
    else:
        spearman = float(
            both[f"{first}_score"].corr(
                both[f"{second}_score"], method="spearman"
            )
        )

    compared = min(top_n, len(both))
    top = [
        set(
            both.sort_values(
                [f"{name}_score", "drug"],
                ascending=[False, True],
                kind="stable",
            )["drug"].head(compared)
        )
        for name in (first, second)
    ]
    return {
        "n_drugs": len(both),
        "score_spearman": spearman,
        "top_n": compared,
        "top_n_overlap": len(top[0] & top[1]) / compared
        if compared
        else np.nan,
        "fraction_a_above_train_mean": _fraction_above(matched, first),
        "fraction_b_above_train_mean": _fraction_above(matched, second),
        "status": status,
    }


def _fraction_above(matched: pd.DataFrame, name: str) -> float:
    """Return the share of a baseline's scored drugs above the reference."""
    difference = matched[f"{name}_score_minus_train_mean"].dropna()
    return float((difference > 0).mean()) if len(difference) else np.nan
