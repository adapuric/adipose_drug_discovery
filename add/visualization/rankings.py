"""Drug-ranking and external-context visualizations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from add.visualization.tables import _drug_labels
from add.visualization.tables import _ordered_values
from add.visualization.tables import _require_columns
from add.visualization.tables import _score_label
from add.visualization.tables import _state_column


def plot_top_rankings(
    rankings: pd.DataFrame,
    *,
    adipocyte_state: str,
    top_n: int = 15,
    score_col: str = "score",
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Plot top drug scores as a horizontal lollipop ranking.

    Example Usage:
      >>> figure, axis = plot_top_rankings(
      ...     rankings,
      ...     adipocyte_state="AD_ALL",
      ... )
    """
    state_col = _state_column(rankings)
    _require_columns(
        rankings,
        (state_col, score_col),
        table_name="rankings",
    )
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    selected = rankings.loc[rankings[state_col].astype(str) == adipocyte_state]
    numeric_scores = cast(
        pd.Series,
        pd.to_numeric(selected[score_col], errors="coerce"),
    )
    finite_mask = np.isfinite(numeric_scores.to_numpy(dtype=float))
    selected = selected.loc[finite_mask].copy()
    if selected.empty:
        raise ValueError(
            f"No finite rankings found for state {adipocyte_state!r}"
        )
    selected[score_col] = selected[score_col].astype(float)
    selected = selected.nlargest(top_n, score_col).sort_values(score_col)
    labels = _drug_labels(selected)

    height = max(2.0, 0.22 * len(selected) + 0.7)
    figure, axis = plt.subplots(figsize=figsize or (3.5, height))
    positions = np.arange(len(selected))
    scores = selected[score_col].to_numpy(dtype=float)
    colors = np.where(scores >= 0, "#b2182b", "#2166ac")
    axis.hlines(positions, 0.0, scores, color="#bdbdbd", linewidth=0.8)
    axis.scatter(scores, positions, c=colors, s=18, zorder=3)
    axis.axvline(0.0, color="#555555", linewidth=0.5)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.set_xlabel(_score_label(score_col))
    axis.set_ylabel("Candidate drug")
    axis.set_title(f"Top rescue-mimicking candidates — {adipocyte_state}")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure, axis


def plot_drug_state_scores(
    rankings: pd.DataFrame,
    *,
    top_drugs: int = 20,
    score_col: str = "score",
    state_order: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Plot a drug-by-state score matrix with context support as dot area.

    Example Usage:
      >>> figure, axis = plot_drug_state_scores(rankings, top_drugs=12)
    """
    state_col = _state_column(rankings)
    _require_columns(
        rankings,
        (state_col, score_col),
        table_name="rankings",
    )
    if top_drugs < 1:
        raise ValueError("top_drugs must be at least 1")

    table = rankings.copy()
    table["_drug_label"] = _drug_labels(table)
    table[score_col] = pd.to_numeric(table[score_col], errors="coerce")
    finite = table.loc[np.isfinite(table[score_col])]
    if finite.empty:
        raise ValueError("rankings contains no finite scores")
    selected_drugs = (
        finite.groupby("_drug_label", sort=False)[score_col]
        .max()
        .nlargest(top_drugs)
        .index.tolist()
    )
    state_values = pd.Series(table[state_col])
    states = _ordered_values(state_values, requested=state_order)
    table = table.loc[table["_drug_label"].isin(selected_drugs)]
    if table.duplicated(["_drug_label", state_col]).any():
        raise ValueError("rankings must have one row per drug and state")

    score_matrix = table.pivot(
        index="_drug_label",
        columns=state_col,
        values=score_col,
    ).reindex(index=selected_drugs, columns=states)
    support_col = "n_external_contexts"
    if support_col in table:
        table[support_col] = pd.to_numeric(
            table[support_col],
            errors="coerce",
        )
        support_matrix = table.pivot(
            index="_drug_label",
            columns=state_col,
            values=support_col,
        ).reindex(index=selected_drugs, columns=states)
    else:
        support_matrix = pd.DataFrame(
            1.0, score_matrix.index, score_matrix.columns
        )

    width = max(2.6, 0.5 * len(states) + 1.7)
    height = max(2.2, 0.23 * len(selected_drugs) + 0.8)
    figure, axis = plt.subplots(figsize=figsize or (width, height))
    x_grid, y_grid = np.meshgrid(
        np.arange(len(states)),
        np.arange(len(selected_drugs)),
    )
    score_values = score_matrix.to_numpy(dtype=float)
    support_values = support_matrix.to_numpy(dtype=float, na_value=np.nan)
    finite_scores = np.isfinite(score_values)
    finite_support = support_values[np.isfinite(support_values)]
    maximum_support = (
        float(np.max(finite_support)) if finite_support.size else 1.0
    )
    sizes = 12.0 + 48.0 * np.nan_to_num(support_values / maximum_support)
    dots = axis.scatter(
        x_grid[finite_scores],
        y_grid[finite_scores],
        c=score_values[finite_scores],
        s=sizes[finite_scores],
        cmap="RdBu_r",
        norm=mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
        edgecolors="#333333",
        linewidths=0.25,
    )
    missing = ~finite_scores
    axis.scatter(
        x_grid[missing],
        y_grid[missing],
        marker="x",
        c="#bdbdbd",
        s=10,
        linewidths=0.5,
    )
    axis.set_xticks(np.arange(len(states)))
    axis.set_xticklabels(states, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(selected_drugs)))
    axis.set_yticklabels(selected_drugs)
    axis.invert_yaxis()
    axis.set_xlabel("Adipocyte state")
    axis.set_ylabel("Candidate drug")
    axis.set_title("Drug mimicry across adipocyte states")
    colorbar = figure.colorbar(dots, ax=axis, fraction=0.04, pad=0.03)
    colorbar.set_label(_score_label(score_col))
    figure.tight_layout()
    return figure, axis


def plot_context_variability(
    context_scores: pd.DataFrame,
    *,
    adipocyte_state: str,
    drugs: Sequence[str] | None = None,
    score_col: str = "score_mimic",
    random_seed: int = 42,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Plot individual external-context scores for selected drugs.

    Example Usage:
      >>> figure, axis = plot_context_variability(
      ...     context_scores,
      ...     adipocyte_state="AD_ALL",
      ... )
    """
    state_col = _state_column(context_scores)
    _require_columns(
        context_scores,
        (state_col, score_col),
        table_name="context_scores",
    )
    selected = context_scores.loc[
        context_scores[state_col].astype(str) == adipocyte_state
    ].copy()
    selected["_drug_label"] = _drug_labels(selected)
    selected[score_col] = pd.to_numeric(selected[score_col], errors="coerce")
    selected = selected.loc[np.isfinite(selected[score_col])]
    if drugs is None:
        selected_drugs = (
            selected.groupby("_drug_label", sort=False)[score_col]
            .median()
            .nlargest(10)
            .index.tolist()
        )
    else:
        selected_drugs = list(drugs)
    selected = selected.loc[selected["_drug_label"].isin(selected_drugs)]
    if selected.empty:
        raise ValueError(
            "No finite context scores match the requested state/drugs"
        )

    height = max(2.0, 0.28 * len(selected_drugs) + 0.7)
    figure, axis = plt.subplots(figsize=figsize or (3.5, height))
    rng = np.random.default_rng(random_seed)
    for position, drug in enumerate(selected_drugs):
        values = selected.loc[
            selected["_drug_label"] == drug,
            score_col,
        ].to_numpy(dtype=float)
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        axis.scatter(
            values,
            position + jitter,
            s=10,
            alpha=0.65,
            color="#4c78a8",
            edgecolors="none",
        )
        median = float(np.median(values))
        axis.plot(
            [median, median],
            [position - 0.16, position + 0.16],
            color="#111111",
            linewidth=1.2,
        )
    axis.axvline(0.0, color="#777777", linewidth=0.5)
    axis.set_yticks(np.arange(len(selected_drugs)))
    axis.set_yticklabels(selected_drugs)
    axis.invert_yaxis()
    axis.set_xlabel(_score_label(score_col))
    axis.set_ylabel("Candidate drug")
    axis.set_title(f"External-context variability — {adipocyte_state}")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure, axis
