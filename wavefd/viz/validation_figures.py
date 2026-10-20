"""Figures for exact comparison, convergence, modes, and dispersion."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wavefd.analysis.convergence import ConvergenceResult
from wavefd.analysis.dispersion import numerical_dispersion
from wavefd.analysis.exact import compare_to_exact_string
from wavefd.analysis.modes import plucked_string_modal_spectrum
from wavefd.solvers.string_1d import StringSolution
from wavefd.viz.style import PALETTE, apply_style


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_validation_figures(
    string_fixed: StringSolution,
    out_dir: str | Path,
) -> tuple[dict[str, Path], dict]:
    """Build the validation panel set and return paths plus numeric summaries."""
    apply_style()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    exact_cmp = compare_to_exact_string(string_fixed)
    from wavefd.analysis.convergence import run_string_convergence

    conv = run_string_convergence()
    modes = plucked_string_modal_spectrum(string_fixed, n_modes=30)
    disp = numerical_dispersion(c=string_fixed.config.c, dx=string_fixed.dx, r=string_fixed.r)

    paths = {
        "exact_error": _fig_exact_error(string_fixed, exact_cmp, out),
        "convergence": _fig_convergence(conv, out),
        "modal_spectrum": _fig_modal_spectrum(modes, out),
        "dispersion": _fig_dispersion(disp, out),
    }
    summary = {
        "exact": {
            "max_rel_l2": exact_cmp["max_rel_l2"],
            "final_rel_l2": exact_cmp["final_rel_l2"],
        },
        "convergence": {
            "dx": conv.dx.tolist(),
            "rel_l2": conv.rel_l2.tolist(),
            "observed_orders": [
                None if not np.isfinite(o) else float(o) for o in conv.observed_orders
            ],
            "mean_order": conv.mean_order,
        },
        "modes": {
            "odd_energy_fraction": float(
                np.sum(modes["energy_fraction_numerical"][0::2])
            ),
        },
    }
    return paths, summary


def _fig_exact_error(sol: StringSolution, cmp: dict, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    mid = sol.x.size // 2
    t_ms = sol.t * 1e3
    axes[0].plot(t_ms, sol.u[:, mid] * 1e3, color=PALETTE["accent"], lw=2.0, label="numerical")
    axes[0].plot(
        t_ms,
        cmp["exact"][:, mid] * 1e3,
        "--",
        color=PALETTE["warm"],
        lw=1.6,
        label="exact d'Alembert",
    )
    axes[0].set_xlabel(r"$t$ [ms]")
    axes[0].set_ylabel(r"$u(L/2,t)$ [mm]")
    axes[0].set_title("Mid-point: leapfrog vs d'Alembert")
    axes[0].legend(fontsize=9)

    axes[1].semilogy(t_ms, cmp["rel_l2"], color=PALETTE["cool"], lw=2.0)
    axes[1].set_xlabel(r"$t$ [ms]")
    axes[1].set_ylabel(r"$\|u_{\mathrm{num}}-u_{\mathrm{ex}}\|_2 / \|u_0\|_2$")
    axes[1].set_title(f"Relative $L^2$ error (max {cmp['max_rel_l2']:.2e})")
    fig.tight_layout()
    return _save(fig, out / "09_exact_comparison.png")


def _fig_convergence(conv: ConvergenceResult, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.loglog(conv.dx, conv.rel_l2, "o-", color=PALETTE["accent"], lw=2.0, ms=8, label="measured")
    # Reference slopes anchored at the finest point
    ref2 = conv.rel_l2[-1] * (conv.dx / conv.dx[-1]) ** 2
    ref4 = conv.rel_l2[-1] * (conv.dx / conv.dx[-1]) ** 4
    ax.loglog(conv.dx, ref2, "--", color=PALETTE["muted"], lw=1.4, label=r"$\mathcal{O}(\Delta x^2)$")
    ax.loglog(conv.dx, ref4, "--", color=PALETTE["warm"], lw=1.6, label=r"$\mathcal{O}(\Delta x^4)$")
    ax.set_xlabel(r"$\Delta x$ [m]")
    ax.set_ylabel(r"relative $L^2$ error at $t=T_1/2$")
    ax.set_title(
        f"Pure-mode convergence — observed order {conv.mean_order:.2f} "
        r"(scheme is formally $\mathcal{O}(\Delta x^2,\Delta t^2)$)"
    )
    ax.legend()
    ax.grid(True, which="both", color=PALETTE["grid"], lw=0.8)
    fig.tight_layout()
    return _save(fig, out / "10_convergence.png")


def _fig_modal_spectrum(modes: dict[str, np.ndarray], out: Path) -> Path:
    n = modes["n"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].stem(
        n,
        np.abs(modes["b_numerical"]),
        linefmt=PALETTE["accent"],
        markerfmt="o",
        basefmt=" ",
        label="numerical DST",
    )
    axes[0].plot(n, np.abs(modes["b_analytic"]), "s", color=PALETTE["warm"], ms=4, label="analytic $b_n$")
    axes[0].set_xlabel(r"mode $n$")
    axes[0].set_ylabel(r"$|b_n|$")
    axes[0].set_title("Sine coefficients of the triangular pluck")
    axes[0].legend(fontsize=9)

    axes[1].semilogy(
        n,
        modes["energy_fraction_numerical"],
        "o-",
        color=PALETTE["cool"],
        lw=1.8,
        label="numerical",
    )
    axes[1].semilogy(
        n,
        modes["energy_fraction_analytic"],
        "s--",
        color=PALETTE["warm"],
        lw=1.4,
        label="analytic",
    )
    axes[1].set_xlabel(r"mode $n$")
    axes[1].set_ylabel(r"energy fraction")
    axes[1].set_title("Odd harmonics carry the pluck energy")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, out / "11_modal_spectrum.png")


def _fig_dispersion(disp: dict[str, np.ndarray], out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    k = disp["kappa"]
    axes[0].plot(k, disp["omega_exact"], color=PALETTE["ink"], lw=2.0, label=r"exact $\omega=c\kappa$")
    axes[0].plot(
        k,
        disp["omega_numerical"],
        color=PALETTE["accent"],
        lw=2.0,
        label=r"leapfrog $\omega_{\mathrm{num}}$",
    )
    axes[0].set_xlabel(r"$\kappa$ [rad/m]")
    axes[0].set_ylabel(r"$\omega$ [rad/s]")
    axes[0].set_title(rf"Dispersion at $r={float(disp['r']):.2f}$")
    axes[0].legend(fontsize=9)

    axes[1].plot(k[1:], disp["phase_ratio"][1:], color=PALETTE["warm"], lw=2.0)
    axes[1].axhline(1.0, color=PALETTE["grid"], lw=1.2, ls="--")
    axes[1].set_xlabel(r"$\kappa$ [rad/m]")
    axes[1].set_ylabel(r"$\omega_{\mathrm{num}}/(c\kappa)$")
    axes[1].set_title("Phase-speed ratio (exact on long waves)")
    axes[1].set_ylim(0.7, 1.05)
    fig.tight_layout()
    return _save(fig, out / "12_dispersion.png")
