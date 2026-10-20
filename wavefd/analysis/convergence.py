"""Grid-refinement study for the 1D leapfrog scheme."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavefd.analysis.exact import exact_pure_mode, relative_l2_error
from wavefd.solvers.string_1d import BoundaryMode, StringConfig, StringWaveSolver


@dataclass(frozen=True)
class ConvergenceResult:
    n_x: np.ndarray
    dx: np.ndarray
    rel_l2: np.ndarray
    observed_orders: np.ndarray

    @property
    def mean_order(self) -> float:
        finite = self.observed_orders[np.isfinite(self.observed_orders)]
        return float(np.mean(finite)) if finite.size else float("nan")


def run_string_convergence(
    *,
    resolutions: tuple[int, ...] = (33, 65, 129, 257),
    length: float = 0.64,
    c: float = 250.0,
    amplitude: float = 0.01,
    mode: int = 1,
    cfl: float = 0.5,
    n_periods: float = 0.5,
) -> ConvergenceResult:
    """Refine ``Δx`` at fixed Courant number on a pure standing-wave problem.

    Using a single eigenmode removes series-truncation error so the measured
    rate reflects the leapfrog scheme alone. The error is the relative $L^2$
    discrepancy at the final time.
    """
    dx_list: list[float] = []
    err_list: list[float] = []
    n_list: list[int] = []

    for n_x in resolutions:
        x = np.linspace(0.0, length, n_x)
        u0 = amplitude * np.sin(mode * np.pi * x / length)
        cfg = StringConfig(c=c, cfl=cfl, n_periods=n_periods)
        sol = StringWaveSolver(x, u0, cfg).solve(mode=BoundaryMode.FIXED)
        exact = exact_pure_mode(
            sol.x,
            sol.t,
            length=length,
            c=c,
            n=mode,
            amplitude=amplitude,
        )
        rel = relative_l2_error(sol.u, exact, sol.dx)
        n_list.append(n_x)
        dx_list.append(sol.dx)
        err_list.append(float(rel[-1]))

    dx = np.asarray(dx_list, dtype=float)
    err = np.asarray(err_list, dtype=float)
    orders = np.full(err.size, np.nan)
    for i in range(1, err.size):
        ratio = dx[i - 1] / dx[i]
        if err[i] > 0 and ratio > 1:
            orders[i] = np.log(err[i - 1] / err[i]) / np.log(ratio)

    return ConvergenceResult(
        n_x=np.asarray(n_list, dtype=int),
        dx=dx,
        rel_l2=err,
        observed_orders=orders,
    )
