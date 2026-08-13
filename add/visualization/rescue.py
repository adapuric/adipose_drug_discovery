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

from add.visualization.tables import _ordered_values
from add.visualization.tables import _require_columns


def plot_rescue_summary(
    rescue_results: pd.DataFrame,
    *,
    top_genes: int = 24,
    state_order: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Plot the strongest rescue statistics across adipocyte states.

    Genes are selected by their largest absolute moderated t statistic in any
    state. Empty state-gene combinations remain visibly missing.

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

    width = max(3.4, 0.18 * len(selected_genes))
    height = max(1.8, 0.28 * len(states))
    figure, axis = plt.subplots(figsize=figsize or (width, height))
    limit = float(np.nanmax(np.abs(matrix.to_numpy(dtype=float))))
    color_map = mpl.colormaps["RdBu_r"].copy()
    color_map.set_bad("#e6e6e6")
    image = axis.imshow(
        np.ma.masked_invalid(matrix.to_numpy(dtype=float)),
        aspect="auto",
        cmap=color_map,
        norm=mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )

    support = finite.groupby("state", sort=False)["n_pairs"].max()
    axis.set_yticks(np.arange(len(states)))
    axis.set_yticklabels(
        [f"{state} (donors={int(support.get(state, 0))})" for state in states]
    )
    axis.set_xticks(np.arange(len(selected_genes)))
    axis.set_xticklabels(selected_genes, rotation=60, ha="right")
    axis.set_xlabel("Gene")
    axis.set_ylabel("Adipocyte state")
    axis.set_title("Paired baseline → weight-loss rescue vector")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("Moderated t (weight loss - baseline)")
    figure.tight_layout()
    return figure, axis
