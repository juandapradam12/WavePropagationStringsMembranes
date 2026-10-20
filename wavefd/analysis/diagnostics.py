"""Diagnostic helpers used in the narrative figures and documentation."""

from __future__ import annotations

import numpy as np

from wavefd.solvers.string_1d import StringSolution
from wavefd.solvers.membrane_2d import MembraneSolution


def relative_energy_drift(energy: np.ndarray) -> float:
    """Peak relative deviation of discrete energy from its initial value."""
    e0 = energy[0]
    if abs(e0) < 1e-30:
        return float("nan")
    return float(np.max(np.abs(energy - e0)) / abs(e0))


def standing_wave_period_estimate(solution: StringSolution, node_index: int | None = None) -> float:
    """Estimate the fundamental period from spatial-field autocorrelation.

    The triangular pluck is a superposition of odd harmonics, so a single-node
    zero-crossing count is unreliable. Instead we measure the first recurrence
    of the full profile: the lag that maximizes the correlation of
    :math:`u(\\cdot,t)` with :math:`u(\\cdot,0)` after the trivial peak at
    lag zero. That lag approximates :math:`T_1=2L/c`.
    """
    del node_index  # retained for API compatibility
    u0 = solution.u[0]
    u0_center = u0 - u0.mean()
    denom = float(np.dot(u0_center, u0_center))
    if denom < 1e-30:
        return float("nan")

    # Skip a short early window so the peak at lag 0 is ignored.
    t_end = solution.t[-1]
    skip = max(1, int(0.2 * solution.u.shape[0]))
    best_corr = -np.inf
    best_idx = skip
    for i in range(skip, solution.u.shape[0]):
        ui = solution.u[i] - solution.u[i].mean()
        corr = float(np.dot(u0_center, ui)) / denom
        if corr > best_corr:
            best_corr = corr
            best_idx = i

    # Prefer a peak near the expected period when several candidates exist.
    T_hint = solution.config.fundamental_period(float(solution.x[-1] - solution.x[0]))
    if t_end >= 0.8 * T_hint:
        window = np.where(np.abs(solution.t - T_hint) < 0.15 * T_hint)[0]
        if window.size:
            corrs = []
            for i in window:
                ui = solution.u[i] - solution.u[i].mean()
                corrs.append(float(np.dot(u0_center, ui)) / denom)
            best_idx = int(window[int(np.argmax(corrs))])
    return float(solution.t[best_idx])


def cfl_report(solution: StringSolution | MembraneSolution) -> dict[str, float]:
    """Summarize stability numbers for logging and documentation."""
    if isinstance(solution, StringSolution):
        r_max = 1.0
        dim = 1
    else:
        r_max = 1.0 / np.sqrt(2.0)
        dim = 2
    return {
        "dimension": float(dim),
        "courant": float(solution.r),
        "courant_limit": float(r_max),
        "margin": float(r_max - solution.r),
        "energy_drift": relative_energy_drift(solution.energy),
    }
