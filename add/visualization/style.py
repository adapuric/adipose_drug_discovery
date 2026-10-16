"""Shared publication styling for adipose discovery figures.

Figures follow the NASP atlas design: 5 pt type, 0.25 pt axes, geometry set
in inches per row or cell, constrained layout, Okabe-Ito colours and grey
for missing values.
"""

from pathlib import Path

import numpy as np
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def set_matplotlib_publication_parameters() -> None:
    """Set publication parameters with Graphik and sans-serif fallback."""
    font_families = _publication_font_families()

    plt.rcParams.update(
        {
            "font.size": 5,
            "axes.titlesize": 5,
            "axes.labelsize": 5,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.fontsize": 5,
            "figure.titlesize": 5,
            "figure.dpi": 450,
            "font.family": font_families,
            "font.sans-serif": [
                "Arial",
                "Nimbus Sans",
                "DejaVu Sans",
            ],
            "axes.linewidth": 0.25,
            "xtick.major.width": 0.25,
            "ytick.major.width": 0.25,
            "xtick.minor.width": 0.25,
            "ytick.minor.width": 0.25,
        }
    )


def _publication_font_families() -> list[str]:
    """Return an unambiguous Graphik family with sans-serif fallback."""
    alias = "Graphik NASP"
    manager = font_manager.fontManager
    if any(font.name == alias for font in manager.ttflist):
        return [alias, "sans-serif"]

    system_graphik = [
        path
        for path in font_manager.findSystemFonts()
        if "graphik" in Path(path).stem.casefold()
    ]
    registered_paths = {font.fname for font in manager.ttflist}
    for path in system_graphik:
        if path in registered_paths:
            continue
        try:
            manager.addfont(path)
        except (OSError, RuntimeError):
            continue

    graphik_fonts = [font for font in manager.ttflist if font.name == "Graphik"]
    otf_fonts = [
        font
        for font in graphik_fonts
        if Path(font.fname).suffix.casefold() == ".otf"
    ]
    selected_fonts = otf_fonts or graphik_fonts
    if not selected_fonts:
        return ["sans-serif"]

    manager.ttflist.extend(
        font_manager.FontEntry(
            fname=font.fname,
            name=alias,
            style=font.style,
            variant=font.variant,
            weight=font.weight,
            stretch=font.stretch,
            size=font.size,
        )
        for font in selected_fonts
    )
    return [alias, "sans-serif"]


def _style_row_axis(axis: Axes) -> None:
    """Style a panel of labelled rows: open frame, tickless rows."""
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0, pad=2)
    axis.tick_params(axis="x", length=2, pad=1.5)


def _fit_row_labels(
    figure: Figure,
    axis: Axes,
    *,
    content_width: float,
    margin: float = 0.3,
) -> None:
    """Widen the figure so the longest row label fits beside its content.

    `content_width` is the data area, and `margin` the colour bar or padding,
    in inches.
    """
    labels = axis.get_yticklabels()
    if not labels:
        return

    label_width = max(text.get_window_extent().width for text in labels)
    needed = label_width / figure.dpi + content_width + margin
    if needed > figure.get_figwidth():
        figure.set_figwidth(needed)


def _color_limit(values: np.ndarray, quantile: float) -> float:
    """Symmetric colour limit: a quantile of absolute finite values, or 1."""
    if not 0.5 <= quantile <= 1:
        raise ValueError("color_quantile must lie between 0.5 and 1")

    finite = np.abs(values[np.isfinite(values)])
    limit = float(np.quantile(finite, quantile)) if finite.size else 1.0
    return limit or 1.0
