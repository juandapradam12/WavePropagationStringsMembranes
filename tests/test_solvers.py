"""Unit checks for the finite-difference wave solvers."""

from __future__ import annotations

import numpy as np
import pytest

from wavefd.analysis.diagnostics import relative_energy_drift, standing_wave_period_estimate
from wavefd.solvers.membrane_2d import MembraneConfig, MembraneWaveSolver
from wavefd.solvers.string_1d import BoundaryMode, StringConfig, StringWaveSolver


def _pluck(n: int = 65, length: float = 0.64, amp: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, length, n)
    mid = length / 2.0
    u0 = np.where(x <= mid, amp * x / mid, amp * (length - x) / mid)
    u0[0] = u0[-1] = 0.0
    return x, u0


def test_string_cfl_rejects_unstable_step():
    x, u0 = _pluck()
    cfg = StringConfig(dt=1e-3, n_periods=0.1)  # deliberately unstable
    with pytest.raises(ValueError, match="Unstable Courant"):
        StringWaveSolver(x, u0, cfg).solve()


def test_string_energy_and_period():
    x, u0 = _pluck()
    cfg = StringConfig(cfl=0.5, n_periods=1.0)
    sol = StringWaveSolver(x, u0, cfg).solve(mode=BoundaryMode.FIXED)
    assert relative_energy_drift(sol.energy) < 1e-2
    T_th = cfg.fundamental_period(x[-1] - x[0])
    T_est = standing_wave_period_estimate(sol)
    assert abs(T_est - T_th) / T_th < 0.05


def test_driven_injects_energy():
    x, u0 = _pluck()
    cfg = StringConfig(cfl=0.5, n_periods=1.0)
    sol = StringWaveSolver(x, u0, cfg).solve(mode=BoundaryMode.DRIVEN)
    assert sol.energy[-1] > sol.energy[0]


def test_membrane_energy_stable():
    n = 41
    L = 0.5
    xx = np.linspace(0.0, L, n)
    X, Y = np.meshgrid(xx, xx)
    u0 = 0.01 * np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
    cfg = MembraneConfig(length=L, cfl=0.4, n_periods=0.5)
    sol = MembraneWaveSolver(u0, cfg).solve()
    assert relative_energy_drift(sol.energy) < 2e-2
    assert sol.r <= 1.0 / np.sqrt(2.0) + 1e-12
