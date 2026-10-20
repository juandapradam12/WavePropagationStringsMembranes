"""Shared Matplotlib style for coherent storytelling figures."""

from __future__ import annotations

import matplotlib as mpl


PALETTE = {
    "ink": "#1a1f2e",
    "muted": "#5c6578",
    "accent": "#0b6e4f",
    "accent_soft": "#3d9a7a",
    "warm": "#c45c26",
    "cool": "#2c5f8a",
    "bg": "#f7f5f1",
    "grid": "#d9d4cb",
    "cmap": "YlGnBu_r",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["bg"],
            "axes.edgecolor": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.titlecolor": PALETTE["ink"],
            "axes.titlesize": 12,
            "axes.titlepad": 12,
            "axes.labelsize": 11,
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "text.color": PALETTE["ink"],
            "font.family": "DejaVu Sans",
            "axes.grid": False,
            "legend.frameon": False,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.25,
            "savefig.facecolor": PALETTE["bg"],
        }
    )
