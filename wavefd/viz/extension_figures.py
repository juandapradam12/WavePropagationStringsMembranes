"""Figures for membrane verification, CFL demo, damping, and free-end dynamics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from wavefd.analysis.cfl_demo import CFLDemoResult
from wavefd.analysis.membrane_exact import compare_membrane_mode
from wavefd.analysis.membrane_modes import membrane_modal_spectrum
from wavefd.solvers.membrane_2d import MembraneSolution
from wavefd.solvers.string_1d import StringSolution
from wavefd.viz.style import PALETTE, apply_style


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_extension_figures(
    *,
    membrane_mode_sol: MembraneSolution,
    membrane_strike: MembraneSolution,
    cfl_demo: CFLDemoResult,
    string_damped: StringSolution,
    string_free: StringSolution,
    string_fixed: StringSolution,
    out_dir: str | Path,
) -> tuple[dict[str, Path], dict]:
    apply_style()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mem_cmp = compare_membrane_mode(membrane_mode_sol, m=1, n=1)
    mem_modes = membrane_modal_spectrum(membrane_strike, n_modes=6)

    paths = {
        "membrane_exact": _fig_membrane_exact(membrane_mode_sol, mem_cmp, out),
        "cfl_blowup": _fig_cfl_blowup(cfl_demo, out),
        "membrane_modes": _fig_membrane_modes(mem_modes, out),
        "damping": _fig_damping(string_fixed, string_damped, out),
        "free_end": _fig_free_end(string_free, out),
    }
    summary = {
        "membrane_mode": {
            "max_rel_l2": mem_cmp["max_rel_l2"],
            "final_rel_l2": mem_cmp["final_rel_l2"],
        },
        "cfl_demo": {
            "stable_r": float(cfl_demo.stable.r),
            "unstable_r": float(cfl_demo.unstable.r),
            "stable_final_peak": float(cfl_demo.stable_peak[-1]),
            "unstable_final_peak": float(cfl_demo.unstable_peak[-1]),
        },
        "membrane_modes": {
            "dominant": _dominant_mode(mem_modes),
        },
        "damping": {
            "gamma": float(string_damped.config.gamma),
            "energy_ratio": float(string_damped.energy[-1] / string_damped.energy[0]),
        },
    }
    return paths, summary


def _dominant_mode(modes: dict[str, np.ndarray]) -> dict[str, float | int]:
    frac = modes["energy_fraction"]
    idx = np.unravel_index(int(np.argmax(frac)), frac.shape)
    return {
        "m": int(modes["m"][idx]),
        "n": int(modes["n"][idx]),
        "energy_fraction": float(frac[idx]),
    }


def _fig_membrane_exact(sol: MembraneSolution, cmp: dict, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    cy, cx = sol.u.shape[1] // 2, sol.u.shape[2] // 2
    t_ms = sol.t * 1e3
    axes[0].plot(t_ms, sol.u[:, cy, cx] * 1e3, color=PALETTE["accent"], lw=2.0, label="numerical")
    axes[0].plot(
        t_ms,
        cmp["exact"][:, cy, cx] * 1e3,
        "--",
        color=PALETTE["warm"],
        lw=1.6,
        label="exact $(1,1)$ mode",
    )
    axes[0].set_xlabel(r"$t$ [ms]")
    axes[0].set_ylabel(r"$u(L/2,L/2,t)$ [mm]")
    axes[0].set_title("Membrane center: leapfrog vs exact mode")
    axes[0].legend(fontsize=9)

    axes[1].semilogy(t_ms, cmp["rel_l2"], color=PALETTE["cool"], lw=2.0)
    axes[1].set_xlabel(r"$t$ [ms]")
    axes[1].set_ylabel(r"$\|u_{\mathrm{num}}-u_{\mathrm{ex}}\|_2 / \|u_0\|_2$")
    axes[1].set_title(f"Relative $L^2$ error (max {cmp['max_rel_l2']:.2e})")
    fig.tight_layout()
    return _save(fig, out / "13_membrane_exact.png")


def _fig_cfl_blowup(demo: CFLDemoResult, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].semilogy(
        demo.stable.t * 1e3,
        demo.stable_peak,
        color=PALETTE["accent"],
        lw=2.0,
        label=rf"stable $r={demo.stable.r:.2f}$",
    )
    axes[0].semilogy(
        demo.unstable.t * 1e3,
        demo.unstable_peak,
        color=PALETTE["warm"],
        lw=2.0,
        label=rf"unstable $r={demo.unstable.r:.2f}$",
    )
    axes[0].set_xlabel(r"$t$ [ms]")
    axes[0].set_ylabel(r"$\max_x |u(x,t)|$ [m]")
    axes[0].set_title("CFL violation triggers exponential blow-up")
    axes[0].legend(fontsize=9)

    # Profile snapshot near the end (clip unstable for display)
    idx_s = -1
    idx_u = min(demo.unstable.t.size - 1, int(0.7 * (demo.unstable.t.size - 1)))
    axes[1].plot(
        demo.stable.x,
        demo.stable.u[idx_s],
        color=PALETTE["accent"],
        lw=2.0,
        label="stable profile",
    )
    u_uns = demo.unstable.u[idx_u]
    # Rescale for visibility if huge
    scale = np.max(np.abs(u_uns)) or 1.0
    axes[1].plot(
        demo.unstable.x,
        u_uns / scale * np.max(np.abs(demo.stable.u[idx_s])),
        color=PALETTE["warm"],
        lw=1.6,
        label="unstable (rescaled)",
    )
    axes[1].set_xlabel(r"$x$ [m]")
    axes[1].set_ylabel(r"$u$ [m] (rescaled)")
    axes[1].set_title("Late-time profiles")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, out / "14_cfl_blowup.png")


def _fig_membrane_modes(modes: dict[str, np.ndarray], out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    n_modes = modes["coefficients"].shape[0]
    im0 = axes[0].imshow(
        np.abs(modes["coefficients"]),
        origin="lower",
        cmap="YlGnBu_r",
        extent=[0.5, n_modes + 0.5, 0.5, n_modes + 0.5],
    )
    axes[0].set_xlabel(r"$m$")
    axes[0].set_ylabel(r"$n$")
    axes[0].set_title(r"Rectangular amplitudes $|a_{mn}|$")
    axes[0].set_xticks(range(1, n_modes + 1))
    axes[0].set_yticks(range(1, n_modes + 1))
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="4%", pad=0.08)
    fig.colorbar(im0, cax=cax)

    im1 = axes[1].imshow(
        modes["energy_fraction"],
        origin="lower",
        cmap="magma",
        extent=[0.5, n_modes + 0.5, 0.5, n_modes + 0.5],
    )
    axes[1].set_xlabel(r"$m$")
    axes[1].set_ylabel(r"$n$")
    axes[1].set_title("Modal energy fraction of the strike")
    axes[1].set_xticks(range(1, n_modes + 1))
    axes[1].set_yticks(range(1, n_modes + 1))
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="4%", pad=0.08)
    fig.colorbar(im1, cax=cax)
    fig.tight_layout()
    return _save(fig, out / "15_membrane_modes.png")


def _fig_damping(fixed: StringSolution, damped: StringSolution, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    mid = fixed.x.size // 2
    axes[0].plot(
        fixed.t * 1e3,
        fixed.u[:, mid] * 1e3,
        color=PALETTE["muted"],
        lw=1.4,
        label=r"$\gamma=0$",
    )
    axes[0].plot(
        damped.t * 1e3,
        damped.u[:, mid] * 1e3,
        color=PALETTE["accent"],
        lw=2.0,
        label=rf"$\gamma={damped.config.gamma:.0f}\,\mathrm{{s^{{-1}}}}$",
    )
    axes[0].set_xlabel(r"$t$ [ms]")
    axes[0].set_ylabel(r"$u(L/2,t)$ [mm]")
    axes[0].set_title("Viscous damping decays the mid-point oscillation")
    axes[0].legend(fontsize=9)

    axes[1].plot(
        fixed.t * 1e3,
        fixed.energy / fixed.energy[0],
        color=PALETTE["muted"],
        lw=1.4,
        label="undamped",
    )
    axes[1].plot(
        damped.t * 1e3,
        damped.energy / damped.energy[0],
        color=PALETTE["warm"],
        lw=2.0,
        label="damped",
    )
    axes[1].set_xlabel(r"$t$ [ms]")
    axes[1].set_ylabel(r"$E(t)/E(0)$")
    axes[1].set_title("Mechanical energy drains under linear friction")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, out / "16_damping.png")


def _fig_free_end(sol: StringSolution, out: Path) -> Path:
    fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
    idxs = sol.snapshot_indices(fractions)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for frac, idx in zip(fractions, idxs):
        axes[0].plot(sol.x, sol.u[idx], lw=1.7, label=rf"$t/T_1={frac:.2f}$")
    axes[0].axhline(0.0, color=PALETTE["grid"], lw=1)
    axes[0].set_xlabel(r"$x$ [m]")
    axes[0].set_ylabel(r"$u(x,t)$ [m]")
    axes[0].set_title(r"Free right end ($u_x=0$): no sign flip on reflection")
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
    axes[1].set_xlim(sol.x[0], sol.x[-1])
    axes[1].set_title("Space–time: reflection without inversion at $x=L$")
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="4%", pad=0.08)
    fig.colorbar(im, cax=cax, label=r"$u$ [m]")
    fig.tight_layout()
    return _save(fig, out / "17_free_end.png")
