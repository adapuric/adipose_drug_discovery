"""Descriptive model-edit effects and donor-level prediction comparisons."""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_edit_effect_heatmap(
    effects: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    color_limit: float | None = None,
    label_width: int = 24,
) -> tuple[Figure, Axes]:
    """Show equal-donor sensitivity effects with missing combinations masked.

    Example Usage:
      >>> figure, axis = plot_edit_effect_heatmap(effects, color_limit=0.5)
    """
    _validate_effects(effects)
    table = (
        effects.groupby(["edit_id", "state", "gene_id"], observed=True)["delta"]
        .mean()
        .unstack("gene_id")
    )
    values = table.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    limit = (
        color_limit
        if color_limit is not None
        else (float(np.max(np.abs(finite))) if finite.size else 1.0)
    )
    if limit == 0 and color_limit is None:
        limit = 1.0
    if not np.isfinite(limit) or limit <= 0 or label_width < 1:
        raise ValueError("color_limit and label_width must be positive")
    figure, axis = plt.subplots(
        figsize=_figure_size(
            figsize,
            (max(5, len(table.columns) * 0.45), max(3, len(table) * 0.4)),
        )
    )
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#dddddd")
    artist = axis.imshow(
        np.ma.masked_invalid(values),
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    labels = _gene_labels(effects)
    axis.set_xticks(
        np.arange(len(table.columns)),
        [
            textwrap.fill(labels.get(str(x), str(x)), label_width)
            for x in table.columns
        ],
        rotation=60,
        ha="right",
    )
    axis.set_yticks(
        np.arange(len(table)),
        [
            textwrap.fill(f"{edit} | {state}", label_width)
            for edit, state in table.index
        ],
    )
    axis.set_title("Model sensitivity: equal-donor mean\nGrey = missing")
    figure.colorbar(artist, ax=axis, label=_effect_label(effects))
    figure.tight_layout()
    return figure, axis


def plot_donor_edit_effects(
    effects: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    point_size: float = 18,
    label_width: int = 34,
    intervals: pd.DataFrame | None = None,
) -> tuple[Figure, Axes]:
    """Show independent donor effects and means without invented intervals.

    Example Usage:
      >>> figure, axis = plot_donor_edit_effects(effects, point_size=24)
    """
    _validate_effects(effects)
    table = effects.copy()
    labels = _gene_labels(effects)
    table["gene_label"] = table["gene_id"].map(labels)
    table["label"] = (
        table[["edit_id", "state", "gene_label"]]
        .astype(str)
        .agg(" | ".join, axis=1)
    )
    figure, axis = _donor_points(
        table,
        value_col="delta",
        label_col="label",
        figsize=figsize,
        point_size=point_size,
        label_width=label_width,
    )
    if intervals is not None:
        interval_table = intervals.copy()
        interval_table["gene_label"] = interval_table["gene_id"].map(labels)
        interval_table["label"] = (
            interval_table[["edit_id", "state", "gene_label"]]
            .astype(str)
            .agg(" | ".join, axis=1)
        )
        positions = {
            label: i for i, label in enumerate(sorted(table["label"].unique()))
        }
        for _, row in interval_table.iterrows():
            if (
                row["label"] in positions
                and np.isfinite([row["lower"], row["upper"]]).all()
            ):
                axis.plot(
                    [row["lower"], row["upper"]],
                    [positions[row["label"]]] * 2,
                    color="black",
                    linewidth=1.2,
                )
        axis.set_title("Donor effects; conditional donor-bootstrap intervals")
    axis.set_xlabel(_effect_label(effects))
    if intervals is None:
        axis.set_title("Model sensitivity: donor points and equal-donor means")
    return figure, axis


def plot_downstream_magnitude(
    magnitudes: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    point_size: float = 18,
) -> tuple[Figure, Axes]:
    """Compare downstream effect sizes with the exact no-effect reference.

    Example Usage:
      >>> figure, axis = plot_downstream_magnitude(magnitudes, point_size=22)
    """
    required = {"edit_id", "state", "model", "donor_id", "mean_absolute_delta"}
    if not required <= set(magnitudes):
        raise ValueError("Magnitude table lacks independent-donor effects")
    table = magnitudes.copy()
    table["label"] = (
        table[["edit_id", "state", "model"]].astype(str).agg(" | ".join, axis=1)
    )
    figure, axis = _donor_points(
        table,
        value_col="mean_absolute_delta",
        label_col="label",
        figsize=figsize,
        point_size=point_size,
        label_width=40,
    )
    axis.set_xlabel("Mean absolute downstream effect (edited genes excluded)")
    axis.set_title(
        "Model sensitivity magnitude; zero is the no-effect reference"
    )
    return figure, axis


def plot_transition_errors(
    metrics: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    point_size: float = 18,
) -> tuple[Figure, Axes]:
    """Show donor-level change prediction errors with explicit model support.

    Example Usage:
      >>> figure, axis = plot_transition_errors(metrics, figsize=(7, 4))
    """
    if not {"model", "state", "donor_id", "delta_rmse"} <= set(metrics):
        raise ValueError("Transition table lacks donor-level error columns")
    table = metrics.copy()
    table["label"] = (
        table[["model", "state"]].astype(str).agg(" | ".join, axis=1)
    )
    figure, axis = _donor_points(
        table,
        value_col="delta_rmse",
        label_col="label",
        figsize=figsize,
        point_size=point_size,
        label_width=40,
    )
    axis.set_xlabel("RMSE of paired log1p-CPM change (lower is better)")
    transductive = (
        "comparison_class" in metrics
        and metrics["comparison_class"]
        .eq("donor_heldout_transductive_preprocessing")
        .any()
    )
    axis.set_title(
        "Donor-held-out transition; transductive preprocessing"
        if transductive
        else "Donor-held-out surgery transition: individual errors"
    )
    return figure, axis


def plot_transition_delta_scatter(
    predictions: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (5, 5),
    maximum_points: int = 5000,
    random_seed: int = 42,
) -> tuple[Figure, Axes]:
    """Plot declared paired changes without selecting favorable genes.

    Example Usage:
      >>> figure, axis = plot_transition_delta_scatter(
      ...     predictions, maximum_points=1000,
      ... )
    """
    if not {"observed_delta", "predicted_delta", "model", "state"} <= set(
        predictions
    ):
        raise ValueError(
            "Scatter requires named observed and predicted changes"
        )
    if (
        predictions["model"].nunique() != 1
        or predictions["state"].nunique() != 1
    ):
        raise ValueError(
            "Select one model and state explicitly for the scatter"
        )
    if maximum_points < 1:
        raise ValueError("maximum_points must be positive")
    table = predictions.loc[
        np.isfinite(predictions["observed_delta"])
        & np.isfinite(predictions["predicted_delta"])
    ]
    if table.empty:
        raise ValueError("No finite change pairs are available")
    if len(table) > maximum_points:
        table = table.sample(maximum_points, random_state=random_seed)
    figure, axis = plt.subplots(figsize=_figure_size(figsize, (5, 5)))
    axis.scatter(
        table["observed_delta"],
        table["predicted_delta"],
        s=5,
        alpha=0.3,
        rasterized=True,
    )
    bounds = table[["observed_delta", "predicted_delta"]].to_numpy()
    low, high = float(bounds.min()), float(bounds.max())
    axis.plot([low, high], [low, high], color="grey", linewidth=0.8)
    axis.set(
        xlabel="Observed weightloss - baseline (log1p-CPM)",
        ylabel="Predicted change (log1p-CPM)",
        title=(
            f"{table['model'].iloc[0]} | {table['state'].iloc[0]}\n"
            "Descriptive donor/gene pairs"
        ),
    )
    figure.tight_layout()
    return figure, axis


def plot_support_qc(
    qc: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Display nuclei coverage without treating nuclei as independent donors.

    Example Usage:
      >>> figure, axis = plot_support_qc(qc, figsize=(8, 4))
    """
    if not {"donor_id", "state", "edit_id", "n_nuclei"} <= set(qc):
        raise ValueError(
            "QC table requires donor/state/edit and nucleus counts"
        )
    table = qc.copy()
    table["label"] = (
        table[["donor_id", "state", "edit_id"]]
        .astype(str)
        .agg(" | ".join, axis=1)
    )
    if table["label"].duplicated().any():
        raise ValueError("Select one run for the support QC plot")
    figure, axis = plt.subplots(
        figsize=_figure_size(figsize, (7, max(3, len(table) * 0.2)))
    )
    axis.barh(np.arange(len(table)), table["n_nuclei"])
    axis.set_yticks(np.arange(len(table)), table["label"])
    axis.set(
        xlabel="Nuclei observed within each donor/state",
        title="Coverage QC; biological replicate = donor",
    )
    figure.tight_layout()
    return figure, axis


def _validate_effects(effects: pd.DataFrame) -> None:
    """Require one effect per donor/edit/state/gene on one declared scale."""
    keys = ["edit_id", "state", "donor_id", "gene_id"]
    if not {*keys, "delta", "scale"} <= set(effects):
        raise ValueError("Effect table lacks required named columns")
    if (
        effects.empty
        or effects.duplicated(keys).any()
        or effects["scale"].nunique() != 1
    ):
        raise ValueError("Select one scale/run with unique donor effect rows")


def _effect_label(effects: pd.DataFrame) -> str:
    """Keep historical mean-count ratios distinct from normalized count
    effects.
    """
    scale = str(effects["scale"].iloc[0])
    if scale == "log2_ratio_of_nucleus_means_plus1":
        return "log2((edited nucleus mean + 1)/(unedited nucleus mean + 1))"
    if scale == "shared_unedited_log1p_cpm":
        return "Edited - unedited log1p-CPM (shared unedited size factor)"
    return "Edited - unedited summed model counts"


def _figure_size(
    size: tuple[float, float] | None,
    default: tuple[float, float],
) -> tuple[float, float]:
    """Validate presentation dimensions in inches."""
    resolved = size or default
    if len(resolved) != 2 or not all(
        np.isfinite(x) and x > 0 for x in resolved
    ):
        raise ValueError("figsize requires two finite positive dimensions")
    return resolved


def _donor_points(
    table: pd.DataFrame,
    *,
    value_col: str,
    label_col: str,
    figsize: tuple[float, float] | None,
    point_size: float,
    label_width: int,
) -> tuple[Figure, Axes]:
    """Render complete donor values and means with independent support
    labels.
    """
    if not np.isfinite(point_size) or point_size <= 0 or label_width < 1:
        raise ValueError("point_size and label_width must be positive")
    if table.duplicated([label_col, "donor_id"]).any():
        raise ValueError("Duplicate independent donor values in a plot group")
    groups = sorted(table[label_col].unique())
    figure, axis = plt.subplots(
        figsize=_figure_size(figsize, (8, max(3, len(groups) * 0.4)))
    )
    labels: list[str] = []
    for position, label in enumerate(groups):
        group = table.loc[table[label_col].eq(label)].sort_values("donor_id")
        values = group[value_col].to_numpy(dtype=float)
        finite = np.isfinite(values)
        offsets = (
            np.linspace(-0.12, 0.12, len(group))
            if len(group) > 1
            else np.zeros(1)
        )
        axis.scatter(
            values[finite], position + offsets[finite], s=point_size, alpha=0.75
        )
        if finite.any():
            axis.scatter(
                [values[finite].mean()],
                [position],
                marker="|",
                s=point_size * 5,
                color="black",
            )
        labels.append(
            textwrap.fill(
                f"{label} (donors {finite.sum()}/{len(group)})", label_width
            )
        )
    axis.axvline(0, color="grey", linewidth=0.8)
    axis.set_yticks(np.arange(len(groups)), labels)
    figure.tight_layout()
    return figure, axis


def _gene_labels(effects: pd.DataFrame) -> dict[str, str]:
    """Use display symbols without collapsing canonical gene identities."""
    genes = effects[["gene_id"]].drop_duplicates()
    if "symbol" not in effects:
        return {str(gene): str(gene) for gene in genes["gene_id"]}
    mapping = effects[["gene_id", "symbol"]].drop_duplicates()
    if mapping["gene_id"].duplicated().any():
        raise ValueError("Each canonical gene must have one display symbol")
    mapping["ambiguous_symbol"] = mapping["symbol"].duplicated(keep=False)
    labels: dict[str, str] = {}
    for _, row in mapping.iterrows():
        gene, symbol = str(row["gene_id"]), row["symbol"]
        labels[gene] = (
            gene
            if pd.isna(symbol) or not str(symbol).strip()
            else f"{symbol} ({gene})"
            if row["ambiguous_symbol"]
            else str(symbol)
        )
    return labels
