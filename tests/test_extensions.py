"""Tests for membrane exact mode, CFL demo, damping, free end, and membrane modes."""

from __future__ import annotations

import numpy as np

from wavefd.analysis.cfl_demo import run_cfl_stability_demo
from wavefd.analysis.membrane_exact import compare_membrane_mode, exact_membrane_mode
from wavefd.analysis.membrane_modes import membrane_modal_spectrum
from wavefd.solvers.membrane_2d import MembraneConfig, MembraneWaveSolver
from wavefd.solvers.string_1d import BoundaryMode, StringConfig, StringWaveSolver


def _pluck(n: int = 65, length: float = 0.64, amp: float = 0.016):
    x = np.linspace(0.0, length, n)
    mid = length / 2.0
    u0 = np.where(x <= mid, amp * x / mid, amp * (length - x) / mid)
    u0[0] = u0[-1] = 0.0
    return x, u0


def test_membrane_exact_mode_small_error():
    L, n = 0.5, 41
    xx = np.linspace(0.0, L, n)
    X, Y = np.meshgrid(xx, xx)
    u0 = 0.01 * np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
    sol = MembraneWaveSolver(u0, MembraneConfig(length=L, cfl=0.4, n_periods=1.0)).solve()
    cmp = compare_membrane_mode(sol, m=1, n=1)
    assert cmp["max_rel_l2"] < 5e-2


def test_exact_membrane_mode_shape():
    x = np.linspace(0.0, 0.5, 21)
    y = np.linspace(0.0, 0.5, 21)
    t = np.linspace(0.0, 0.001, 5)
    u = exact_membrane_mode(x, y, t, length=0.5, c=250.0, amplitude=0.01)
    assert u.shape == (5, 21, 21)
    assert np.allclose(u[:, 0, :], 0.0)
    assert np.allclose(u[:, :, 0], 0.0)


def test_cfl_demo_unstable_grows():
    demo = run_cfl_stability_demo(n_periods=0.3)
    assert demo.unstable_peak[-1] > 10.0 * demo.stable_peak[-1]
    assert demo.stable.r < 1.0 < demo.unstable.r


def test_damping_reduces_energy():
    x, u0 = _pluck()
    undamped = StringWaveSolver(x, u0, StringConfig(cfl=0.5, n_periods=2.0, gamma=0.0)).solve()
    damped = StringWaveSolver(x, u0, StringConfig(cfl=0.5, n_periods=2.0, gamma=500.0)).solve()
    assert damped.energy[-1] / damped.energy[0] < 0.5
    assert damped.energy[-1] < undamped.energy[-1]


def test_free_end_moves():
    x, u0 = _pluck()
    sol = StringWaveSolver(x, u0, StringConfig(cfl=0.5, n_periods=1.0)).solve(BoundaryMode.FREE)
    # Right endpoint is no longer pinned
    assert np.max(np.abs(sol.u[:, -1])) > 1e-4
    assert np.allclose(sol.u[:, 0], 0.0)


def test_membrane_modal_spectrum_peaked():
    L, n = 0.5, 51
    u0 = np.loadtxt("/workspace/data/membrane_ic.dat")
    # downsample-compatible: use file as-is (101x101)
    sol = MembraneWaveSolver(u0, MembraneConfig(length=L, cfl=0.4, n_periods=0.1)).solve()
    spec = membrane_modal_spectrum(sol, n_modes=5)
    assert np.isclose(np.sum(spec["energy_fraction"]), 1.0, atol=1e-6)
    # Central strike should put most energy in low (m,n)
    assert float(np.max(spec["energy_fraction"])) > 0.2
