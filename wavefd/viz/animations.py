"""Animated GIFs of string and membrane temporal evolution."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import TwoSlopeNorm

from wavefd.solvers.membrane_2d import MembraneSolution
from wavefd.solvers.string_1d import StringSolution
from wavefd.viz.style import PALETTE, apply_style


def _frame_indices(n_t: int, n_frames: int) -> np.ndarray:
    """Evenly spaced time indices, always including the first and last step."""
    if n_t <= n_frames:
        return np.arange(n_t, dtype=int)
    return np.unique(np.linspace(0, n_t - 1, n_frames, dtype=int))


def animate_string_gif(
    sol: StringSolution,
    out_path: str | Path,
    *,
    n_frames: int = 60,
    fps: int = 12,
) -> Path:
    """GIF of the fixed-end string over the simulated interval."""
    apply_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    idxs = _frame_indices(sol.t.size, n_frames)
    ymin = float(np.min(sol.u))
    ymax = float(np.max(sol.u))
    pad = 0.08 * max(ymax - ymin, 1e-12)
    T1 = sol.config.fundamental_period(float(sol.x[-1] - sol.x[0]))

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    (line,) = ax.plot([], [], color=PALETTE["accent"], lw=2.2)
    ax.axhline(0.0, color=PALETTE["grid"], lw=1)
    ax.set_xlim(sol.x[0], sol.x[-1])
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$u(x,t)$ [m]")
    title = ax.set_title("")
    fig.tight_layout()

    def _init():
        line.set_data([], [])
        title.set_text("")
        return line, title

    def _update(frame: int):
        i = int(idxs[frame])
        line.set_data(sol.x, sol.u[i])
        title.set_text(
            rf"Fixed string — $t={sol.t[i]*1e3:.2f}$ ms "
            rf"($t/T_1={sol.t[i]/T1:.2f}$)"
        )
        return line, title

    anim = animation.FuncAnimation(
        fig,
        _update,
        init_func=_init,
        frames=len(idxs),
        interval=1000 / fps,
        blit=True,
    )
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def animate_membrane_gif(
    sol: MembraneSolution,
    out_path: str | Path,
    *,
    n_frames: int = 50,
    fps: int = 10,
) -> Path:
    """GIF of the clamped membrane displacement field."""
    apply_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    idxs = _frame_indices(sol.t.size, n_frames)
    vmax = float(np.max(np.abs(sol.u))) or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    extent = [sol.x[0], sol.x[-1], sol.y[0], sol.y[-1]]
    T11 = sol.config.fundamental_period()

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    im = ax.imshow(
        sol.u[0],
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        norm=norm,
        aspect="equal",
        animated=True,
    )
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    title = ax.set_title("")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$u$ [m]")
    fig.tight_layout()

    def _update(frame: int):
        i = int(idxs[frame])
        im.set_data(sol.u[i])
        title.set_text(
            rf"Clamped membrane — $t={sol.t[i]*1e3:.2f}$ ms "
            rf"($t/T_{{11}}={sol.t[i]/T11:.2f}$)"
        )
        return im, title

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(idxs),
        interval=1000 / fps,
        blit=False,
    )
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out_path
