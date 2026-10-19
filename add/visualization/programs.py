"""Visualizations of rescue gene-set programs and drug score decomposition."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from add.visualization.style import _color_limit
from add.visualization.style import _fit_row_labels
from add.visualization.style import _style_row_axis
from add.visualization.style import set_matplotlib_publication_parameters
from add.visualization.tables import _require_columns


def plot_program_matrix(
    matrix: pd.DataFrame,
    *,
    color_label: str,
    title: str,
    figsize: tuple[float, float] | None = None,
    cell_size: float = 0.13,
    color_quantile: float = 1.0,
) -> tuple[Figure, Axes]:
    """Plot a labelled rows-by-columns matrix on a diverging scale.

    The scale is centred on zero and limited by the `color_quantile` of
    absolute values; the colour bar is extended when values are clipped.
    Missing cells are grey crosses, never zero. Rows and columns keep the
    order of `matrix`.

    Args:
      matrix: Numeric table whose index and columns label the cells.
      color_label: Colour-bar label naming the quantity and its units.
      title: Figure title stating what is shown.
      figsize: Overall inches; overrides `cell_size`.
      cell_size: Inches per cell when `figsize` is None.
      color_quantile: Quantile of absolute values used as the colour limit
        (0.5 to 1).

    Example Usage:
      >>> figure, axis = plot_program_matrix(
      ...     shifts,
      ...     color_label="Mean moderated t shift",
      ...     title="Hallmark programs in the rescue vector",
      ...     cell_size=0.15,
      ... )
    """
    if matrix.empty:
        raise ValueError("matrix has no cells to plot")
    if not np.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("cell_size must be positive")

    values = matrix.to_numpy(dtype=float)
    limit = _color_limit(values, color_quantile)
    finite = values[np.isfinite(values)]
    dimensions = figsize or (
        1.2 + cell_size * matrix.shape[1],
        0.8 + cell_size * matrix.shape[0],
    )
    with plt.rc_context():
        set_matplotlib_publication_parameters()
        figure, axis = plt.subplots(figsize=dimensions, layout="constrained")

        color_map = mpl.colormaps["RdBu_r"].copy()
        color_map.set_bad("#e6e6e6")
        image = axis.imshow(
            np.ma.masked_invalid(values),
            aspect="auto",
            cmap=color_map,
            norm=mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        )
        missing_y, missing_x = np.nonzero(~np.isfinite(values))
        axis.scatter(
            missing_x, missing_y, marker="x", s=6, c="#999999", linewidths=0.4
        )
        axis.set_xticks(
            np.arange(matrix.shape[1]),
            [str(column) for column in matrix.columns],
            rotation=90,
        )
        axis.set_yticks(
            np.arange(matrix.shape[0]), [str(row) for row in matrix.index]
        )
        axis.tick_params(length=0, pad=1.5)
        axis.spines[:].set_visible(False)

        colorbar = figure.colorbar(
            image,
            ax=axis,
            shrink=min(1.0, 4.0 / max(matrix.shape[0], 1) + 0.25),
            aspect=10,
            pad=0.02,
            extend="both" if (np.abs(finite) > limit).any() else "neither",
        )
        colorbar.set_label(color_label)
        figure.suptitle(
            title, x=0.01, ha="left", fontsize=plt.rcParams["axes.titlesize"]
        )
        if figsize is None:
            _fit_row_labels(
                figure,
                axis,
                content_width=cell_size * matrix.shape[1],
                margin=0.9,
            )

    return figure, axis


def plot_donor_program_scores(
    scores: pd.DataFrame,
    *,
    programs: Sequence[str],
    condition_order: Sequence[str],
    group_col: str | None = None,
    figsize: tuple[float, float] | None = None,
    panel_width: float = 1.3,
    height: float = 1.6,
    point_size: float = 4.0,
) -> tuple[Figure, np.ndarray]:
    """Show each donor's program score by condition, joining paired visits.

    One panel per program. Columns are conditions within each group (for
    example study); grey lines join one donor's scores, so the paired
    change is visible, and the black bar is the median over donors. Column
    labels give the donors shown. Scores are mean gene z scores across all
    profiles, so they compare profiles, not programs.

    Args:
      scores: Rows with `donor_id`, `condition`, `program` and `score`, and
        `group_col` when given.
      programs: Programs to show, one panel each, in order.
      condition_order: Conditions shown, in order; others are omitted.
      group_col: Optional column splitting columns into groups.
      figsize: Overall inches; overrides `panel_width` and `height`.
      panel_width: Inches per program panel when `figsize` is None.
      height: Figure height in inches when `figsize` is None.
      point_size: Donor marker area in points squared.

    Example Usage:
      >>> figure, axes = plot_donor_program_scores(
      ...     scores,
      ...     programs=["stress", "metabolic"],
      ...     condition_order=["baseline", "weightloss", "Lean"],
      ...     group_col="study",
      ...     panel_width=1.5,
      ... )
    """
    _require_columns(
        scores,
        ("donor_id", "condition", "program", "score"),
        table_name="scores",
    )
    if not programs:
        raise ValueError("programs must name at least one program")
    if not np.isfinite(point_size) or point_size <= 0:
        raise ValueError("point_size must be positive")

    table = scores.loc[scores["condition"].isin(list(condition_order))].copy()
    table["_group"] = (
        table[group_col].astype(str) if group_col is not None else "all"
    )
    columns = [
        (group, condition)
        for group in sorted(table["_group"].unique())
        for condition in condition_order
        if (
            (table["_group"] == group) & (table["condition"] == condition)
        ).any()
    ]
    position = {column: index for index, column in enumerate(columns)}
    dimensions = figsize or (0.5 + panel_width * len(programs), height)

    with plt.rc_context():
        set_matplotlib_publication_parameters()
        figure, axes = plt.subplots(
            1,
            len(programs),
            figsize=dimensions,
            layout="constrained",
            squeeze=False,
        )
        for axis, program in zip(axes[0], programs, strict=True):
            rows = table.loc[table["program"].eq(program)]
            _draw_donor_columns(
                axis, rows, position=position, point_size=point_size
            )
            counts = (
                rows.dropna(subset=["score"])
                .groupby(["_group", "condition"])["donor_id"]
                .nunique()
            )
            axis.set_xticks(
                np.arange(len(columns)),
                [
                    f"{condition} ({int(counts.get((group, condition), 0))})"
                    + (f"\n{group}" if group_col is not None else "")
                    for group, condition in columns
                ],
                rotation=90,
            )
            axis.set_title(program, fontsize=plt.rcParams["axes.titlesize"])
            axis.axhline(0.0, color="#AAAAAA", linewidth=0.3, zorder=0)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0, 0].set_ylabel("Program score (mean gene z)")
        figure.suptitle(
            "Program scores per donor; lines join a donor's paired visits",
            x=0.01,
            ha="left",
            fontsize=plt.rcParams["axes.titlesize"],
        )

    return figure, axes


def plot_score_contributions(
    decomposition: pd.DataFrame,
    *,
    state: str,
    part_order: Sequence[str],
    drugs: Sequence[str] | None = None,
    top_n: int = 15,
    label_col: str = "drug",
    figsize: tuple[float, float] | None = None,
    width: float = 1.6,
    row_height: float = 0.11,
) -> tuple[Figure, Axes]:
    """Show how each gene group adds up to a drug's rescue-mimicry score.

    Bars stack each group's contribution, positive to the right and
    negative to the left of zero; the black dot is the full Pearson score,
    which the contributions sum to. Rows are `drugs` in order or, when None,
    the `top_n` drugs by full score. Descriptive only.

    Args:
      decomposition: Rows from `decompose_program_mimicry` for one source.
      state: Adipocyte state shown.
      part_order: Contribution groups in stacking and legend order.
      drugs: Drug labels to show, in order; None selects by full score.
      top_n: Number of drugs when `drugs` is None.
      label_col: Column holding the drug labels.
      figsize: Overall inches; overrides `width` and `row_height`.
      width: Inches of the data area when `figsize` is None.
      row_height: Inches per drug row when `figsize` is None.

    Example Usage:
      >>> figure, axis = plot_score_contributions(
      ...     decomposition,
      ...     state="AD_ALL",
      ...     part_order=["stress", "metabolic", "shared", "other"],
      ...     top_n=12,
      ... )
    """
    _require_columns(
        decomposition,
        (label_col, "state", "measure", "part", "value", "status"),
        table_name="decomposition",
    )
    rows = decomposition.loc[
        decomposition["state"].astype(str).eq(state)
        & decomposition["status"].eq("ok")
    ]
    full = rows.loc[rows["measure"].eq("full")].set_index(label_col)["value"]
    if full.empty:
        raise ValueError(f"No estimable scores for state {state!r}")
    selected = (
        [drug for drug in drugs if drug in full.index]
        if drugs is not None
        else full.nlargest(top_n).index.tolist()
    )
    if not selected:
        raise ValueError("None of the requested drugs has an estimable score")
    contributions = (
        rows.loc[rows["measure"].eq("contribution")]
        .pivot(index=label_col, columns="part", values="value")
        .reindex(index=selected, columns=list(part_order))
        .fillna(0.0)
    )
    colors = _part_colors(part_order)

    dimensions = figsize or (1.0 + width, 0.75 + row_height * len(selected))
    with plt.rc_context():
        set_matplotlib_publication_parameters()
        figure, axis = plt.subplots(figsize=dimensions, layout="constrained")

        positions = np.arange(len(selected))
        right = np.zeros(len(selected))
        left = np.zeros(len(selected))
        for part in part_order:
            values = contributions[part].to_numpy(dtype=float)
            start = np.where(values >= 0, right, left)
            axis.barh(
                positions,
                values,
                left=start,
                height=0.7,
                color=colors[part],
                linewidth=0,
            )
            right += np.clip(values, 0, None)
            left += np.clip(values, None, 0)
        axis.scatter(
            full.reindex(selected).to_numpy(dtype=float),
            positions,
            s=6,
            color="black",
            zorder=3,
        )
        axis.axvline(0.0, color="#777777", linewidth=0.4, zorder=2)
        axis.set_yticks(positions, selected)
        axis.set_ylim(len(selected) - 0.4, -0.6)
        axis.set_xlabel("Contribution to Pearson mimicry (dot: full score)")
        _style_row_axis(axis)
        figure.legend(
            handles=[
                Patch(color=colors[part], label=part) for part in part_order
            ],
            loc="outside lower center",
            ncols=min(len(part_order), 4),
            frameon=False,
        )
        figure.suptitle(
            f"{state}: what each gene group adds to the score",
            x=0.01,
            ha="left",
            fontsize=plt.rcParams["axes.titlesize"],
        )
        if figsize is None:
            _fit_row_labels(figure, axis, content_width=width)

    return figure, axis


def _draw_donor_columns(
    axis: Axes,
    rows: pd.DataFrame,
    *,
    position: dict[tuple[str, str], int],
    point_size: float,
) -> None:
    """Draw donor scores per column, joined within donor, with medians."""
    rows = rows.assign(
        _x=[
            position.get((group, condition), np.nan)
            for group, condition in zip(
                rows["_group"], rows["condition"], strict=True
            )
        ]
    ).dropna(subset=["_x", "score"])
    for _, donor in rows.groupby(["_group", "donor_id"]):
        ordered = donor.sort_values("_x")
        if len(ordered) > 1:
            axis.plot(
                ordered["_x"],
                ordered["score"],
                color="#CCCCCC",
                linewidth=0.3,
                zorder=1,
            )
    axis.scatter(
        rows["_x"],
        rows["score"],
        s=point_size,
        color="#0072B2",
        edgecolors="white",
        linewidths=0.2,
        zorder=2,
    )
    medians = rows.groupby("_x")["score"].median()
    for x_position, median in zip(
        medians.index.to_numpy(dtype=float),
        medians.to_numpy(dtype=float),
        strict=True,
    ):
        axis.plot(
            [x_position - 0.3, x_position + 0.3],
            [median, median],
            color="black",
            linewidth=0.75,
            zorder=3,
        )


def _part_colors(part_order: Sequence[str]) -> dict[str, str]:
    """Give programs Okabe-Ito colours in order; shared/other stay fixed."""
    palette = ("#D55E00", "#009E73", "#0072B2", "#E69F00", "#56B4E9")
    fixed = {"shared": "#CC79A7", "other": "#BBBBBB"}
    programs = [part for part in part_order if part not in fixed]
    colors = {
        part: palette[index % len(palette)]
        for index, part in enumerate(programs)
    }
    return colors | {part: fixed[part] for part in part_order if part in fixed}
