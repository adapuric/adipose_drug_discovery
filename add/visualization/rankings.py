"""Drug-ranking and external-context visualizations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from add.visualization.style import _color_limit
from add.visualization.style import _fit_row_labels
from add.visualization.style import _style_row_axis
from add.visualization.style import set_matplotlib_publication_parameters
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
    reference_col: str | None = None,
    figsize: tuple[float, float] | None = None,
    width: float = 1.4,
    row_height: float = 0.105,
    marker_size: float = 7.0,
) -> tuple[Figure, Axes]:
    """Plot top drug scores in one state as a lollipop ranking.

    Rank 1 is the top row. A dashed line marks the state's drug-agnostic
    reference score when `reference_col` names one, such as the train-mean
    score on the same genes: drugs left of it mimic the rescue less than the
    generic perturbation response. The comparison is descriptive.

    Args:
      rankings: Drug rows with a state column and `score_col`.
      adipocyte_state: State to plot.
      top_n: Number of highest-scoring drugs shown.
      score_col: Score column plotted on the x axis.
      reference_col: Column holding one reference score per state, on the
        scale of `score_col`; None draws no reference.
      figsize: Overall inches; overrides `width` and `row_height`.
      width: Inches of the data area when `figsize` is None.
      row_height: Inches per drug row when `figsize` is None.
      marker_size: Marker area in points squared.

    Example Usage:
      >>> figure, axis = plot_top_rankings(
      ...     rankings,
      ...     adipocyte_state="AD_ALL",
      ...     reference_col="train_mean_score",
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
    n_scored = len(selected)
    selected[score_col] = selected[score_col].astype(float)
    selected = selected.nlargest(top_n, score_col)
    reference = _state_reference(selected, reference_col)

    dimensions = figsize or (0.9 + width, 0.55 + row_height * len(selected))
    with plt.rc_context():
        set_matplotlib_publication_parameters()
        figure, axis = plt.subplots(figsize=dimensions, layout="constrained")

        positions = np.arange(len(selected))
        scores = selected[score_col].to_numpy(dtype=float)
        axis.hlines(
            positions, 0.0, scores, color="#AAAAAA", linewidth=0.5, zorder=1
        )
        axis.scatter(
            scores,
            positions,
            s=marker_size,
            color="#0072B2",
            edgecolors="white",
            linewidths=0.2,
            zorder=3,
        )
        axis.axvline(0.0, color="#AAAAAA", linewidth=0.4, zorder=0)
        if reference is not None:
            reference_color = "#D55E00"
            axis.axvline(
                reference,
                color=reference_color,
                linestyle="--",
                linewidth=0.5,
                zorder=2,
            )
            figure.legend(
                handles=[
                    Line2D(
                        [],
                        [],
                        color=reference_color,
                        linestyle="--",
                        linewidth=0.5,
                        label="Train-mean reference (drug-agnostic)",
                    )
                ],
                loc="outside lower center",
                frameon=False,
            )

        axis.set_yticks(positions, _drug_labels(selected))
        axis.set_ylim(len(selected) - 0.4, -0.6)
        axis.set_xlabel(_score_label(score_col), labelpad=2)
        _style_row_axis(axis)
        figure.suptitle(
            f"{adipocyte_state}: top {len(selected)} of {n_scored} "
            "scored drugs",
            x=0.01,
            ha="left",
            fontsize=plt.rcParams["axes.titlesize"],
        )
        if figsize is None:
            _fit_row_labels(figure, axis, content_width=width)

    return figure, axis


def plot_drug_state_scores(
    rankings: pd.DataFrame,
    *,
    top_drugs: int = 20,
    score_col: str = "score",
    state_order: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
    cell_size: float = 0.13,
    max_dot_size: float = 30.0,
    color_quantile: float = 0.95,
) -> tuple[Figure, Axes]:
    """Plot a drug-by-state score matrix with context support as dot area.

    Rows are the `top_drugs` drugs with the highest score in any state, in
    that order. Colour is the score on a diverging scale centred on zero and
    limited by the `color_quantile` of absolute displayed scores; the colour
    bar is extended when values are clipped, and `rankings` keeps them. Dot
    area is proportional to the drug's external contexts. Unscored
    drug-state cells are grey crosses, never zero.

    Args:
      rankings: Drug rows with a state column, `score_col` and optionally
        `n_external_contexts`.
      top_drugs: Number of drugs shown.
      score_col: Score column used for selection and colour.
      state_order: States listed first, in this order.
      figsize: Overall inches; overrides `cell_size`.
      cell_size: Inches per drug-state cell when `figsize` is None.
      max_dot_size: Dot area in points squared at the most contexts.
      color_quantile: Quantile of absolute scores used as the colour limit
        (0.5 to 1).

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

    score_matrix, support_matrix = _score_support_matrices(
        table,
        state_col=state_col,
        score_col=score_col,
        drugs=selected_drugs,
        states=states,
    )
    score_values = score_matrix.to_numpy(dtype=float)
    limit = _color_limit(score_values, color_quantile)

    dimensions = figsize or (
        1.2 + cell_size * len(states),
        0.9 + cell_size * len(selected_drugs),
    )
    with plt.rc_context():
        set_matplotlib_publication_parameters()
        figure, axis = plt.subplots(figsize=dimensions, layout="constrained")

        dots = _draw_score_dots(
            axis,
            score_values,
            support_matrix.to_numpy(dtype=float, na_value=np.nan),
            limit=limit,
            max_dot_size=max_dot_size,
        )
        axis.set_xticks(np.arange(len(states)), states, rotation=90)
        axis.set_yticks(np.arange(len(selected_drugs)), selected_drugs)
        axis.set_xlim(-0.6, len(states) - 0.4)
        axis.set_ylim(len(selected_drugs) - 0.4, -0.6)
        axis.tick_params(length=0, pad=1.5)
        axis.spines[:].set_visible(False)

        finite_values = score_values[np.isfinite(score_values)]
        colorbar = figure.colorbar(
            dots,
            ax=axis,
            shrink=0.35,
            aspect=10,
            pad=0.02,
            extend="both" if (np.abs(finite_values) > limit).any() else None,
        )
        colorbar.set_label(_score_label(score_col))
        _support_legend(figure, support_matrix, max_dot_size=max_dot_size)
        figure.suptitle(
            f"Top {len(selected_drugs)} drugs across adipocyte states",
            x=0.01,
            ha="left",
            fontsize=plt.rcParams["axes.titlesize"],
        )
        if figsize is None:
            _fit_row_labels(
                figure,
                axis,
                content_width=cell_size * len(states),
                margin=0.9,
            )

    return figure, axis


def plot_context_variability(
    context_scores: pd.DataFrame,
    *,
    adipocyte_state: str,
    drugs: Sequence[str] | None = None,
    score_col: str = "score_mimic",
    random_seed: int = 42,
    figsize: tuple[float, float] | None = None,
    width: float = 1.4,
    row_height: float = 0.13,
    marker_size: float = 3.0,
) -> tuple[Figure, Axes]:
    """Plot individual external-context scores for selected drugs.

    Each point is one measured context, such as a cell line, jittered with a
    seeded generator; the black bar is the median context score. Row labels
    give the number of scored contexts. `drugs` keeps its order, so passing
    the top-ranked drugs shows how consistently the ranking holds across
    contexts; by default the 10 highest medians are shown.

    Args:
      context_scores: Context rows with a state column and `score_col`.
      adipocyte_state: State to plot.
      drugs: Drug labels in display order; None selects by median.
      score_col: Score column plotted on the x axis.
      random_seed: Seed for the vertical jitter.
      figsize: Overall inches; overrides `width` and `row_height`.
      width: Inches of the data area when `figsize` is None.
      row_height: Inches per drug row when `figsize` is None.
      marker_size: Context marker area in points squared.

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

    dimensions = figsize or (
        0.9 + width,
        0.55 + row_height * len(selected_drugs),
    )
    with plt.rc_context():
        set_matplotlib_publication_parameters()
        figure, axis = plt.subplots(figsize=dimensions, layout="constrained")

        labels = _draw_context_rows(
            axis,
            selected,
            drugs=selected_drugs,
            score_col=score_col,
            marker_size=marker_size,
            random_seed=random_seed,
        )
        axis.axvline(0.0, color="#AAAAAA", linewidth=0.4, zorder=0)
        axis.set_yticks(np.arange(len(selected_drugs)), labels)
        axis.set_ylim(len(selected_drugs) - 0.4, -0.6)
        axis.set_xlabel(_score_label(score_col), labelpad=2)
        _style_row_axis(axis)
        figure.suptitle(
            f"{adipocyte_state}: score per external context (dot) and "
            "median (bar)",
            x=0.01,
            ha="left",
            fontsize=plt.rcParams["axes.titlesize"],
        )
        if figsize is None:
            _fit_row_labels(figure, axis, content_width=width)

    return figure, axis


def _state_reference(rows: pd.DataFrame, column: str | None) -> float | None:
    """Return the one finite reference score shared by a state's rows."""
    if column is None:
        return None
    if column not in rows:
        raise ValueError(f"rankings is missing reference column {column!r}")

    values = pd.to_numeric(rows[column], errors="coerce").dropna().unique()
    if len(values) > 1:
        raise ValueError(f"{column} must hold one value per state")
    return float(values[0]) if len(values) else None


def _score_support_matrices(
    table: pd.DataFrame,
    *,
    state_col: str,
    score_col: str,
    drugs: list[str],
    states: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot scores and context support to drug-by-state matrices.

    Support is 1 for every cell when `n_external_contexts` is absent.
    """
    score_matrix = table.pivot(
        index="_drug_label",
        columns=state_col,
        values=score_col,
    ).reindex(index=drugs, columns=states)

    support_col = "n_external_contexts"
    if support_col not in table:
        support_matrix = pd.DataFrame(
            1.0, score_matrix.index, score_matrix.columns
        )
        return score_matrix, support_matrix

    support = table.assign(
        **{support_col: pd.to_numeric(table[support_col], errors="coerce")}
    )
    support_matrix = support.pivot(
        index="_drug_label",
        columns=state_col,
        values=support_col,
    ).reindex(index=drugs, columns=states)
    return score_matrix, support_matrix


def _draw_score_dots(
    axis: Axes,
    scores: np.ndarray,
    support: np.ndarray,
    *,
    limit: float,
    max_dot_size: float,
) -> PathCollection:
    """Draw clipped score dots sized by support and crosses for missing."""
    rows, columns = np.meshgrid(
        np.arange(scores.shape[0]),
        np.arange(scores.shape[1]),
        indexing="ij",
    )
    scored = np.isfinite(scores)
    maximum_support = (
        float(np.nanmax(support)) if np.isfinite(support).any() else 1.0
    )
    sizes = max_dot_size * np.clip(
        np.nan_to_num(support / maximum_support), 0.15, 1.0
    )

    dots = axis.scatter(
        columns[scored],
        rows[scored],
        c=np.clip(scores[scored], -limit, limit),
        s=sizes[scored],
        cmap="RdBu_r",
        norm=mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        edgecolors="#555555",
        linewidths=0.2,
    )
    axis.scatter(
        columns[~scored],
        rows[~scored],
        marker="x",
        s=6,
        c="#999999",
        linewidths=0.4,
    )
    return dots


def _support_legend(
    figure: Figure,
    support: pd.DataFrame,
    *,
    max_dot_size: float,
) -> None:
    """Explain dot area with the smallest, middle and largest support."""
    values = support.to_numpy(dtype=float, na_value=np.nan)
    finite = np.unique(values[np.isfinite(values)])
    if finite.size < 2:
        return

    maximum = float(finite.max())
    shown = sorted({int(finite.min()), int(np.median(finite)), int(maximum)})
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=np.sqrt(max_dot_size * max(value / maximum, 0.15)),
            markerfacecolor="white",
            markeredgecolor="#555555",
            markeredgewidth=0.2,
            label=str(value),
        )
        for value in shown
    ]
    figure.legend(
        handles=handles,
        title="External contexts",
        loc="outside lower center",
        ncol=len(handles),
        frameon=False,
        handletextpad=0.2,
        columnspacing=0.8,
    )


def _draw_context_rows(
    axis: Axes,
    selected: pd.DataFrame,
    *,
    drugs: list[str],
    score_col: str,
    marker_size: float,
    random_seed: int,
) -> list[str]:
    """Draw jittered context scores and medians; return the row labels."""
    rng = np.random.default_rng(random_seed)
    labels: list[str] = []
    for position, drug in enumerate(drugs):
        values = selected.loc[
            selected["_drug_label"] == drug,
            score_col,
        ].to_numpy(dtype=float)
        labels.append(f"{drug} ({len(values)})")
        if not values.size:
            continue

        jitter = rng.uniform(-0.15, 0.15, size=len(values))
        axis.scatter(
            values,
            position + jitter,
            s=marker_size,
            alpha=0.6,
            color="#0072B2",
            edgecolors="none",
        )
        median = float(np.median(values))
        axis.plot(
            [median, median],
            [position - 0.3, position + 0.3],
            color="black",
            linewidth=0.75,
        )

    return labels
