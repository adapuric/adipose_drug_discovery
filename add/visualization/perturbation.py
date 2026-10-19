"""Descriptive model-edit effects and donor-level prediction comparisons."""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import blended_transform_factory

from add.visualization.style import set_matplotlib_publication_parameters


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
    point_size: float = 5.0,
    column_width: float = 0.16,
    height: float = 2.0,
) -> tuple[Figure, Axes]:
    """Show each held-out donor's change-prediction error under every model.

    Columns are models, grouped by state. Grey lines join one held-out
    donor's errors across models, so the paired comparison is visible, and
    the black bar is the mean over donors with a finite error. Column labels
    give those donors. Errors are the RMSE of the paired log1p-CPM change on
    each fold's evaluation genes; lower is better.

    Args:
      metrics: Rows with `model`, `state`, `donor_id` and `delta_rmse`, and
        optionally `comparison_class`.
      figsize: Overall inches; overrides `column_width` and `height`.
      point_size: Donor marker area in points squared.
      column_width: Inches per model column when `figsize` is None.
      height: Figure height in inches when `figsize` is None.

    Example Usage:
      >>> figure, axis = plot_transition_errors(metrics, column_width=0.2)
    """
    if not {"model", "state", "donor_id", "delta_rmse"} <= set(metrics):
        raise ValueError("Transition table lacks donor-level error columns")
    if not np.isfinite(point_size) or point_size <= 0:
        raise ValueError("point_size must be positive")
    if metrics.duplicated(["model", "state", "donor_id"]).any():
        raise ValueError("Duplicate independent donor values in a plot group")

    table = metrics.assign(
        delta_rmse=pd.to_numeric(metrics["delta_rmse"], errors="coerce")
    )
    states = sorted(table["state"].astype(str).unique())
    colors = _transition_model_colors(table["model"])
    models = list(colors)
    n_columns = len(states) * (len(models) + 1) - 1
    dimensions = figsize or (0.7 + column_width * n_columns, height)
    transductive = (
        "comparison_class" in metrics
        and metrics["comparison_class"]
        .eq("donor_heldout_transductive_preprocessing")
        .any()
    )

    with plt.rc_context():
        set_matplotlib_publication_parameters()
        figure, axis = _figure(figsize, dimensions)

        ticks: list[float] = []
        labels: list[str] = []
        for index, state in enumerate(states):
            positions, state_labels = _draw_paired_donors(
                axis,
                table.loc[table["state"].astype(str).eq(state)],
                colors=colors,
                start=index * (len(models) + 1),
                point_size=point_size,
            )
            ticks += positions
            labels += state_labels
            axis.text(
                float(np.mean(positions)),
                1.0,
                state,
                transform=blended_transform_factory(
                    axis.transData, axis.transAxes
                ),
                ha="center",
                va="bottom",
            )

        axis.set_xticks(ticks, labels, rotation=90)
        axis.set_xlim(-0.6, n_columns - 0.4)
        axis.set_ylim(bottom=0)
        axis.set_ylabel("Change RMSE (log1p-CPM)", labelpad=2)
        axis.tick_params(axis="x", length=0, pad=1.5)
        axis.tick_params(axis="y", length=2, pad=1.5)
        axis.spines[["top", "right"]].set_visible(False)
        figure.suptitle(
            "Donor-held-out surgery change: prediction RMSE per held-out "
            "donor (lower is better)"
            + (
                "\nTransductive preprocessing (transductive_hvg policy)"
                if transductive
                else ""
            ),
            x=0.01,
            ha="left",
            fontsize=plt.rcParams["axes.titlesize"],
        )

    return figure, axis


def plot_transition_delta_scatter(
    predictions: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (1.8, 1.9),
    maximum_points: int = 5000,
    random_seed: int = 42,
    marker_size: float = 1.5,
) -> tuple[Figure, Axes]:
    """Plot declared paired changes without selecting favorable genes.

    Each point is one held-out donor and gene; up to `maximum_points` are
    sampled with `random_seed`. Pairs share donors and genes, so the plot is
    descriptive. The dashed line is equality.

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

    with plt.rc_context():
        set_matplotlib_publication_parameters()
        figure, axis = _figure(figsize, (1.8, 1.9))

        axis.scatter(
            table["observed_delta"],
            table["predicted_delta"],
            s=marker_size,
            alpha=0.3,
            color="#0072B2",
            edgecolors="none",
            rasterized=True,
        )
        bounds = table[["observed_delta", "predicted_delta"]].to_numpy()
        low, high = float(bounds.min()), float(bounds.max())
        axis.plot(
            [low, high],
            [low, high],
            color="#777777",
            linestyle="--",
            linewidth=0.5,
        )
        axis.axhline(0.0, color="#AAAAAA", linewidth=0.4, zorder=0)
        axis.axvline(0.0, color="#AAAAAA", linewidth=0.4, zorder=0)
        axis.set_box_aspect(1)
        axis.set_xlabel(
            "Observed weight loss - baseline (log1p-CPM)", labelpad=2
        )
        axis.set_ylabel("Predicted change (log1p-CPM)", labelpad=2)
        axis.tick_params(length=2, pad=1.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(
            f"{table['model'].iloc[0]} | {table['state'].iloc[0]}: "
            f"{len(table):,} donor-gene pairs",
            loc="left",
            pad=3,
        )

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


def _transition_model_colors(models: pd.Series) -> dict[str, str]:
    """Map each model to its colour, in display order.

    The no-change, mean-shift and target-mean references come first, then
    the adipose ridge, then other exports sorted by name.
    """
    reference_colors = {
        "adipose-no-change": "#999999",
        "adipose-mean-shift": "#E69F00",
        "adipose-target-mean": "#D55E00",
        "adipose-ridge": "#0072B2",
    }
    export_colors = ("#009E73", "#CC79A7", "#56B4E9")
    observed = set(models.astype(str))

    colors = {
        model: color
        for model, color in reference_colors.items()
        if model in observed
    }
    for index, model in enumerate(sorted(observed - set(reference_colors))):
        colors[model] = export_colors[index % len(export_colors)]
    return colors


def _draw_paired_donors(
    axis: Axes,
    state_rows: pd.DataFrame,
    *,
    colors: dict[str, str],
    start: int,
    point_size: float,
) -> tuple[list[float], list[str]]:
    """Draw one state's donor errors joined across models.

    Returns the column positions and their labels with finite donor counts.
    """
    present = [model for model in colors if model in set(state_rows["model"])]
    positions = np.arange(start, start + len(present), dtype=float)
    errors = state_rows.pivot(
        index="donor_id", columns="model", values="delta_rmse"
    ).reindex(columns=present)

    for donor_errors in errors.to_numpy(dtype=float):
        finite = np.isfinite(donor_errors)
        axis.plot(
            positions[finite],
            donor_errors[finite],
            color="#CCCCCC",
            linewidth=0.3,
            zorder=1,
        )

    labels: list[str] = []
    for position, model in zip(positions, present, strict=True):
        values = errors[model].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        labels.append(f"{model} ({len(values)})")
        if not values.size:
            continue

        axis.scatter(
            np.full(len(values), position),
            values,
            s=point_size,
            color=colors[model],
            edgecolors="white",
            linewidths=0.2,
            zorder=2,
        )
        axis.plot(
            [position - 0.3, position + 0.3],
            [values.mean()] * 2,
            color="black",
            linewidth=0.75,
            zorder=3,
        )

    return positions.tolist(), labels


def _figure(
    size: tuple[float, float] | None,
    default: tuple[float, float],
) -> tuple[Figure, Axes]:
    """Create one constrained-layout axes of validated size in inches."""
    return plt.subplots(
        figsize=_figure_size(size, default), layout="constrained"
    )


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
