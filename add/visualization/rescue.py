"""Visualizations of paired adipose rescue statistics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from add.visualization.style import _fit_row_labels
from add.visualization.style import set_matplotlib_publication_parameters
from add.visualization.tables import _ordered_values
from add.visualization.tables import _require_columns


def plot_rescue_summary(
    rescue_results: pd.DataFrame,
    *,
    top_genes: int = 24,
    state_order: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
    cell_size: float = 0.13,
) -> tuple[Figure, Axes]:
    """Plot the strongest rescue statistics across adipocyte states.

    Genes are selected by their largest absolute moderated t statistic in any
    state, the statistic the drug baselines score against. Empty state-gene
    combinations are grey crosses, never zero. Row labels give the paired
    donors per state.

    Args:
      rescue_results: Per-gene rows with `gene`, `state`, `moderated_t` and
        `n_pairs`.
      top_genes: Number of genes shown.
      state_order: States listed first, in this order.
      figsize: Overall inches; overrides `cell_size`.
      cell_size: Inches per state-gene cell when `figsize` is None.

    Example Usage:
      >>> figure, axis = plot_rescue_summary(rescue_results, top_genes=20)
    """
    _require_columns(
        rescue_results,
        ("gene", "state", "moderated_t", "n_pairs"),
        table_name="rescue_results",
    )
    if top_genes < 1:
        raise ValueError("top_genes must be at least 1")

    moderated_t = cast(
        pd.Series,
        pd.to_numeric(
            rescue_results["moderated_t"],
            errors="coerce",
        ),
    )
    finite_mask = np.isfinite(moderated_t.to_numpy(dtype=float))
    finite = rescue_results.loc[finite_mask].copy()
    if finite.empty:
        raise ValueError("rescue_results contains no finite moderated t values")
    finite["moderated_t"] = finite["moderated_t"].astype(float)

    gene_strength = finite.groupby("gene", sort=False)["moderated_t"].apply(
        lambda values: float(np.max(np.abs(values)))
    )
    selected_genes = gene_strength.nlargest(top_genes).index.tolist()
    states = _ordered_values(finite["state"], requested=state_order)
    matrix = (
        finite.loc[finite["gene"].isin(selected_genes)]
        .pivot(index="state", columns="gene", values="moderated_t")
        .reindex(index=states, columns=selected_genes)
    )
    values = matrix.to_numpy(dtype=float)
    limit = float(np.nanmax(np.abs(values)))
    support = finite.groupby("state", sort=False)["n_pairs"].max()

    dimensions = figsize or (
        1.4 + cell_size * len(selected_genes),
        0.75 + cell_size * len(states),
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
        axis.set_box_aspect(len(states) / len(selected_genes))
        missing_y, missing_x = np.nonzero(~np.isfinite(values))
        axis.scatter(
            missing_x,
            missing_y,
            marker="x",
            s=6,
            c="#999999",
            linewidths=0.4,
        )

        axis.set_yticks(
            np.arange(len(states)),
            [
                f"{state} ({int(support.get(state, 0))} donors)"
                for state in states
            ],
        )
        axis.set_xticks(
            np.arange(len(selected_genes)), selected_genes, rotation=90
        )
        for label in axis.get_xticklabels():
            label.set_fontstyle("italic")
        axis.tick_params(length=0, pad=1.5)
        axis.spines[:].set_visible(False)

        colorbar = figure.colorbar(
            image,
            ax=axis,
            shrink=0.35,
            aspect=10,
            pad=0.02,
            anchor=(0.0, 1.0),
        )
        colorbar.set_label("Moderated t")
        # Top-anchored cells leave layout slack at the bottom, where saving
        # crops it; the colour bar would otherwise centre them.
        axis.set_anchor("N")
        figure.suptitle(
            "Paired rescue vector (weight loss - baseline): top genes by "
            "largest |t| in any state",
            x=0.01,
            ha="left",
            fontsize=plt.rcParams["axes.titlesize"],
        )
        if figsize is None:
            _fit_row_labels(
                figure,
                axis,
                content_width=cell_size * len(selected_genes),
                margin=0.9,
            )

    return figure, axis
