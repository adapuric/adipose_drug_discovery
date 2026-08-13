"""Candidate-versus-rescue signature visualizations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_signature_scatter(
    candidate_delta: pd.Series,
    rescue_delta: pd.Series,
    *,
    candidate_label: str,
    adipocyte_state: str,
    figsize: tuple[float, float] = (2.7, 2.5),
) -> tuple[Figure, Axes]:
    """Plot aligned candidate and rescue values for their shared genes.

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

    figure, axis = plt.subplots(figsize=figsize)
    axis.scatter(
        aligned["candidate"],
        aligned["rescue"],
        s=5,
        alpha=0.45,
        color="#4c78a8",
        edgecolors="none",
    )
    axis.axhline(0.0, color="#aaaaaa", linewidth=0.4)
    axis.axvline(0.0, color="#aaaaaa", linewidth=0.4)
    axis.set_xlabel(f"{candidate_label}: treated - vehicle")
    axis.set_ylabel(f"{adipocyte_state}: weight loss - baseline")
    axis.set_title(
        f"Shared genes={len(aligned):,}; Pearson r={correlation:.2f}"
    )
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure, axis
