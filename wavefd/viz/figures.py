"""Generate the figure suite used by the README narrative."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from wavefd.analysis.diagnostics import (
    cfl_report,
    relative_energy_drift,
    standing_wave_period_estimate,
)
from wavefd.solvers.membrane_2d import MembraneSolution
from wavefd.solvers.string_1d import StringSolution
from wavefd.viz.animations import animate_membrane_gif, animate_string_gif
from wavefd.viz.style import PALETTE, apply_style
from wavefd.viz.validation_figures import generate_validation_figures


def generate_all_figures(
    string_fixed: StringSolution,
    string_driven: StringSolution,
    membrane: MembraneSolution,
    string_x0: np.ndarray,
    string_u0: np.ndarray,
    membrane_u0: np.ndarray,
    out_dir: str | Path,
) -> tuple[dict[str, Path], dict]:
    """Write every storytelling figure and return paths plus validation summary."""
    apply_style()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["ic_overview"] = _fig_ic_overview(string_x0, string_u0, membrane_u0, membrane, out)
    paths["cfl"] = _fig_cfl(string_fixed, membrane, out)
    paths["string_evolution"] = _fig_string_evolution(string_fixed, out)
    paths["string_gif"] = animate_string_gif(string_fixed, out / "string_evolution.gif")
    paths["string_spacetime"] = _fig_string_spacetime(string_fixed, out)
    paths["string_energy"] = _fig_energy(string_fixed, "string_energy.png", "Fixed-end string", out)
    paths["string_period"] = _fig_period_check(string_fixed, out)
    paths["driven_evolution"] = _fig_driven(string_driven, out)
    paths["membrane_evolution"] = _fig_membrane_evolution(membrane, out)
    paths["membrane_gif"] = animate_membrane_gif(membrane, out / "membrane_evolution.gif")
    paths["membrane_energy"] = _fig_energy(membrane, "membrane_energy.png", "Clamped membrane", out)
    paths["workflow"] = _fig_workflow_summary(string_fixed, string_driven, membrane, out)

    validation_paths, validation_summary = generate_validation_figures(string_fixed, out)
    paths.update(validation_paths)
    return paths, validation_summary


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path)
    plt.close(fig)
    return path


def _fig_ic_overview(
    x: np.ndarray,
    u0: np.ndarray,
    membrane_u0: np.ndarray,
    membrane: MembraneSolution,
    out: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})

    axes[0].fill_between(x, 0.0, u0, color=PALETTE["accent"], alpha=0.25)
    axes[0].plot(x, u0, color=PALETTE["accent"], lw=2.2)
    axes[0].axhline(0.0, color=PALETTE["grid"], lw=1)
    axes[0].set_xlabel(r"$x$ [m]")
    axes[0].set_ylabel(r"$u(x,0)$ [m]")
    axes[0].set_title("Plucked string — triangular initial displacement")
    peak_i = int(np.argmax(u0))
    axes[0].annotate(
        f"pluck at $x={x[peak_i]:.2f}$ m\n$A={u0[peak_i]*1e3:.1f}$ mm",
        xy=(x[peak_i], u0[peak_i]),
        xytext=(x[peak_i] + 0.08, u0[peak_i] * 0.7),
        fontsize=9,
        color=PALETTE["muted"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["muted"]),
    )

    extent = [membrane.x[0], membrane.x[-1], membrane.y[0], membrane.y[-1]]
    im = axes[1].imshow(
        membrane_u0,
        origin="lower",
        extent=extent,
        cmap=PALETTE["cmap"],
        aspect="equal",
    )
    axes[1].set_xlabel(r"$x$ [m]")
    axes[1].set_ylabel(r"$y$ [m]")
    axes[1].set_title("Drumhead — localized central strike")
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$u(x,y,0)$ [m]")

    fig.suptitle("Initial data for the two wave problems", fontsize=13, y=1.02)
    fig.tight_layout()
    return _save(fig, out / "01_initial_conditions.png")


def _fig_cfl(string: StringSolution, membrane: MembraneSolution, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    cases = ["1D string\n(CFL ≤ 1)", "2D membrane\n(CFL ≤ 1/√2)"]
    used = [string.r, membrane.r]
    limits = [1.0, 1.0 / np.sqrt(2.0)]
    x = np.arange(len(cases))
    ax.bar(x - 0.15, used, width=0.3, color=PALETTE["accent"], label="Working Courant number $r$")
    ax.bar(x + 0.15, limits, width=0.3, color=PALETTE["warm"], alpha=0.75, label="Stability limit")
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.set_ylabel(r"Courant number $r = c\,\Delta t/\Delta x$")
    ax.set_title("Stability: chosen time steps sit safely below the CFL bound")
    ax.set_ylim(0, 1.15)
    for i, (r, lim) in enumerate(zip(used, limits)):
        ax.text(i - 0.15, r + 0.03, f"{r:.2f}", ha="center", fontsize=9, color=PALETTE["ink"])
        ax.text(i + 0.15, lim + 0.03, f"{lim:.2f}", ha="center", fontsize=9, color=PALETTE["ink"])
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, out / "02_cfl_stability.png")


def _fig_string_evolution(sol: StringSolution, out: Path) -> Path:
    fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
    idxs = sol.snapshot_indices(fractions)
    colors = [PALETTE["cool"], PALETTE["accent"], PALETTE["accent_soft"], PALETTE["warm"], PALETTE["ink"]]
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    for frac, idx, color in zip(fractions, idxs, colors):
        label = rf"$t/T_1 = {frac:.2f}$" if frac else r"$t = 0$"
        ax.plot(sol.x, sol.u[idx], color=color, lw=2.0 if frac in (0.0, 1.0) else 1.6, label=label)
    ax.axhline(0.0, color=PALETTE["grid"], lw=1)
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$u(x,t)$ [m]")
    ax.set_title("Fixed ends: the pluck splits, reflects, and reforms after one period")
    ax.legend(ncol=3, fontsize=9, loc="upper right")
    fig.tight_layout()
    return _save(fig, out / "03_string_fixed_evolution.png")


def _fig_string_spacetime(sol: StringSolution, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    # Subsample time for a readable image
    stride = max(1, sol.u.shape[0] // 400)
    field = sol.u[::stride]
    t = sol.t[::stride]
    vmax = np.max(np.abs(field))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(
        field,
        aspect="auto",
        origin="lower",
        extent=[sol.x[0], sol.x[-1], t[0], t[-1]],
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
    )
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$t$ [s]")
    ax.set_xlim(sol.x[0], sol.x[-1])
    ax.set_ylim(t[0], t[-1])
    ax.set_title("Space–time diagram: characteristic reflections at the fixed ends")
    # Overlay characteristic slopes dx/dt = ±c
    c = sol.config.c
    t_char = 0.4 * sol.t[-1]
    x0 = sol.x[sol.x.size // 2]
    ax.plot([x0, x0 + c * t_char], [0, t_char], "--", color=PALETTE["ink"], lw=1.2, alpha=0.7)
    ax.plot([x0, x0 - c * t_char], [0, t_char], "--", color=PALETTE["ink"], lw=1.2, alpha=0.7,
            label=r"characteristics $dx/dt=\pm c$")
    ax.legend(loc="upper right", fontsize=9)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.08)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$u$ [m]")
    fig.tight_layout()
    return _save(fig, out / "04_string_spacetime.png")


def _fig_energy(sol: StringSolution | MembraneSolution, filename: str, title: str, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    e = sol.energy
    drift = relative_energy_drift(e)
    rel = e / e[0]
    ax.plot(sol.t * 1e3, rel, color=PALETTE["accent"], lw=2.0)
    ax.axhline(1.0, color=PALETTE["grid"], lw=1.2, ls="--")
    ax.set_xlabel(r"$t$ [ms]")
    ax.set_ylabel(r"$E(t)/E(0)$")
    ax.set_title(f"{title}: discrete energy conserved to {drift:.2e} relative drift")
    pad = max(0.02, 1.5 * drift)
    ax.set_ylim(1.0 - pad, 1.0 + pad)
    fig.tight_layout()
    return _save(fig, out / filename)


def _fig_period_check(sol: StringSolution, out: Path) -> Path:
    mid = sol.x.size // 2
    T_est = standing_wave_period_estimate(sol, mid)
    T_th = sol.config.fundamental_period(sol.x[-1] - sol.x[0])
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.plot(sol.t * 1e3, sol.u[:, mid] * 1e3, color=PALETTE["cool"], lw=1.8)
    ax.axhline(0.0, color=PALETTE["grid"], lw=1)
    ax.set_xlabel(r"$t$ [ms]")
    ax.set_ylabel(r"$u(L/2,t)$ [mm]")
    err = abs(T_est - T_th) / T_th * 100 if np.isfinite(T_est) else float("nan")
    ax.set_title(
        rf"Mid-point oscillation: $T_{{\mathrm{{num}}}}={T_est*1e3:.3f}$ ms vs "
        rf"$T_1=2L/c={T_th*1e3:.3f}$ ms ({err:.2f}% error)"
    )
    fig.tight_layout()
    return _save(fig, out / "05_string_period_check.png")


def _fig_driven(sol: StringSolution, out: Path) -> Path:
    fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
    idxs = sol.snapshot_indices(fractions)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))

    for frac, idx in zip(fractions, idxs):
        axes[0].plot(sol.x, sol.u[idx], lw=1.7, label=rf"$t/T_1={frac:.2f}$")
    axes[0].axhline(0.0, color=PALETTE["grid"], lw=1)
    axes[0].set_xlabel(r"$x$ [m]")
    axes[0].set_ylabel(r"$u(x,t)$ [m]")
    axes[0].set_title("Right end driven at the fundamental frequency")
    axes[0].legend(fontsize=8, ncol=1)

    stride = max(1, sol.u.shape[0] // 400)
    field = sol.u[::stride]
    t = sol.t[::stride]
    vmax = np.max(np.abs(field)) or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = axes[1].imshow(
        field,
        aspect="auto",
        origin="lower",
        extent=[sol.x[0], sol.x[-1], t[0], t[-1]],
        cmap="RdBu_r",
        norm=norm,
    )
    axes[1].set_xlabel(r"$x$ [m]")
    axes[1].set_ylabel(r"$t$ [s]")
    axes[1].set_title("Energy injects from the driven boundary")
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="4%", pad=0.08)
    fig.colorbar(im, cax=cax, label=r"$u$ [m]")

    fig.tight_layout()
    return _save(fig, out / "06_string_driven.png")


def _fig_membrane_evolution(sol: MembraneSolution, out: Path) -> Path:
    T11 = sol.config.fundamental_period()
    target_fracs = (0.0, 0.25, 0.5, 0.75)
    idxs = [int(np.argmin(np.abs(sol.t - f * T11))) for f in target_fracs]
    vmax = np.max(np.abs(sol.u[idxs]))
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.0))
    extent = [sol.x[0], sol.x[-1], sol.y[0], sol.y[-1]]
    for ax, frac, idx in zip(axes.ravel(), target_fracs, idxs):
        im = ax.imshow(
            sol.u[idx],
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="equal",
        )
        ax.set_title(rf"$t/T_{{11}} = {sol.t[idx]/T11:.2f}$")
        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$y$ [m]")
    fig.suptitle("Circular wavefronts expand, reflect, and interfere on the clamped drumhead", fontsize=12)
    fig.subplots_adjust(right=0.88, wspace=0.28, hspace=0.28)
    cax = fig.add_axes([0.90, 0.18, 0.025, 0.64])
    fig.colorbar(im, cax=cax, label=r"$u$ [m]")
    return _save(fig, out / "07_membrane_evolution.png")


def _fig_workflow_summary(
    string_fixed: StringSolution,
    string_driven: StringSolution,
    membrane: MembraneSolution,
    out: Path,
) -> Path:
    """One-panel summary tying diagnostics into the narrative close."""
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5))

    # 1 — IC peak vs final peak for fixed string (periodicity)
    mid = string_fixed.x.size // 2
    axes[0].plot(string_fixed.t * 1e3, string_fixed.u[:, mid] * 1e3, color=PALETTE["accent"], lw=1.6)
    axes[0].set_title("1D fixed: mid-point returns")
    axes[0].set_xlabel(r"$t$ [ms]")
    axes[0].set_ylabel(r"$u$ [mm]")

    # 2 — Driven amplitude growth at an interior probe
    probe = int(0.75 * (string_driven.x.size - 1))
    axes[1].plot(string_driven.t * 1e3, string_driven.u[:, probe] * 1e3, color=PALETTE["warm"], lw=1.6)
    axes[1].set_title("1D driven: resonant build-up")
    axes[1].set_xlabel(r"$t$ [ms]")
    axes[1].set_ylabel(r"$u$ [mm]")

    # 3 — Membrane center
    cy = membrane.u.shape[1] // 2
    cx = membrane.u.shape[2] // 2
    axes[2].plot(membrane.t * 1e3, membrane.u[:, cy, cx] * 1e3, color=PALETTE["cool"], lw=1.6)
    axes[2].set_title("2D membrane: center oscillates")
    axes[2].set_xlabel(r"$t$ [ms]")
    axes[2].set_ylabel(r"$u$ [mm]")

    for ax in axes:
        ax.axhline(0.0, color=PALETTE["grid"], lw=1)

    s_rep = cfl_report(string_fixed)
    m_rep = cfl_report(membrane)
    fig.suptitle(
        f"Workflow close-out — energy drift "
        f"1D: {s_rep['energy_drift']:.1e}, 2D: {m_rep['energy_drift']:.1e}",
        fontsize=12,
    )
    fig.tight_layout()
    return _save(fig, out / "08_workflow_summary.png")
