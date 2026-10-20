"""Empirical demonstration of CFL stability vs blow-up."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavefd.solvers.string_1d import BoundaryMode, StringConfig, StringWaveSolver


@dataclass(frozen=True)
class CFLDemoResult:
    stable: object  # StringSolution
    unstable: object  # StringSolution
    stable_peak: np.ndarray
    unstable_peak: np.ndarray


def _pluck(n: int = 65, length: float = 0.64, amp: float = 0.016):
    x = np.linspace(0.0, length, n)
    mid = length / 2.0
    u0 = np.where(x <= mid, amp * x / mid, amp * (length - x) / mid)
    u0[0] = u0[-1] = 0.0
    return x, u0


def run_cfl_stability_demo(
    *,
    stable_cfl: float = 0.9,
    unstable_cfl: float = 1.05,
    n_periods: float = 0.35,
    n_x: int = 65,
) -> CFLDemoResult:
    """Integrate the same pluck just below and just above the CFL limit.

    The unstable run uses ``allow_unstable=True`` and a short horizon so the
    exponential growth is visible without overflowing float64 immediately.
    """
    x, u0 = _pluck(n_x)
    stable = StringWaveSolver(
        x, u0, StringConfig(cfl=stable_cfl, n_periods=n_periods)
    ).solve(BoundaryMode.FIXED)
    unstable = StringWaveSolver(
        x,
        u0,
        StringConfig(cfl=unstable_cfl, n_periods=n_periods, allow_unstable=True),
    ).solve(BoundaryMode.FIXED)
    return CFLDemoResult(
        stable=stable,
        unstable=unstable,
        stable_peak=np.max(np.abs(stable.u), axis=1),
        unstable_peak=np.max(np.abs(unstable.u), axis=1),
    )
