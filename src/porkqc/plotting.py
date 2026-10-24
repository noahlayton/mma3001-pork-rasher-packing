"""Shared Matplotlib style so every figure in the report looks consistent.

Colours come from a colour-vision-deficiency-validated categorical palette and
are assigned to classes/series in a *fixed* order, so a given class has the
same colour in every figure.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed (works on servers/CI)
import matplotlib.pyplot as plt  # noqa: E402

#: Categorical colours in fixed order (blue, orange, aqua, yellow, magenta, ...).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"


def apply_style() -> None:
    """Set global rcParams: light surface, recessive grid/axes, readable fonts."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_MUTED,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
