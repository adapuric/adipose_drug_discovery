"""Candidate-versus-rescue signature visualizations."""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from add.visualization.style import set_matplotlib_publication_parameters


def plot_signature_scatter(
    candidate_delta: pd.Series,
    rescue_delta: pd.Series,
    *,
    candidate_label: str,
    adipocyte_state: str,
    figsize: tuple[float, float] = (1.8, 1.9),
    marker_size: float = 1.5,
) -> tuple[Figure, Axes]:
    """Plot aligned candidate and rescue values for their shared genes.

    Each point is one gene; the title gives the shared genes and the Pearson
    correlation that the mimicry score summarizes. Points are rasterized in
    vector output, and long label lines wrap at context separators.

    Args:
      candidate_delta: Drug signature indexed by gene.
      rescue_delta: Rescue vector indexed by gene.
      candidate_label: Name of the plotted signature.
      adipocyte_state: State of the rescue vector.
      figsize: Overall inches.
      marker_size: Marker area in points squared.

    Example Usage:
      >>> figure, axis = plot_signature_scatter(
      ...     drug_delta,
      ...     rescue_delta,
      ...     candidate_label="Drug A",
      ...     adipocyte_state="AD_ALL",
      ... )
    """
    if not candidate_delta.index.is_unique or not rescue_delta.index.is_unique:
        raise ValueError("Candidate and rescue gene identifiers must be unique")
    aligned = pd.concat(
        [
            candidate_delta.rename("candidate"),
            rescue_delta.rename("rescue"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    finite = np.isfinite(aligned["candidate"]) & np.isfinite(aligned["rescue"])
    aligned = aligned.loc[finite]
    if len(aligned) < 3:
        raise ValueError("At least three finite shared genes are required")
    correlation = float(aligned["candidate"].corr(aligned["rescue"]))

    with plt.rc_context():
        set_matplotlib_publication_parameters()
        figure, axis = plt.subplots(figsize=figsize, layout="constrained")

        axis.scatter(
            aligned["candidate"],
            aligned["rescue"],
            s=marker_size,
            alpha=0.4,
            color="#0072B2",
            edgecolors="none",
            rasterized=True,
        )
        axis.axhline(0.0, color="#AAAAAA", linewidth=0.4, zorder=0)
        axis.axvline(0.0, color="#AAAAAA", linewidth=0.4, zorder=0)
        axis.set_box_aspect(1)
        label = "\n".join(
            textwrap.fill(line.replace("|", " | "), width=40)
            for line in candidate_label.splitlines()
        )
        axis.set_xlabel(f"{label}\n(treated - vehicle)", labelpad=2)
        axis.set_ylabel(
            f"{adipocyte_state}: weight loss - baseline\n(moderated t)",
            labelpad=2,
        )
        axis.tick_params(length=2, pad=1.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(
            f"{len(aligned):,} shared genes; Pearson r {correlation:.2f}",
            loc="left",
            pad=3,
        )

    return figure, axis
