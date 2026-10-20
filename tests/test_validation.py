"""Tests for exact comparison, convergence, modes, dispersion, and audio."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wavefd.analysis.audio import export_string_wav, midpoint_waveform
from wavefd.analysis.convergence import run_string_convergence
from wavefd.analysis.dispersion import numerical_dispersion
from wavefd.analysis.exact import compare_to_exact_string, triangular_pluck_coefficients
from wavefd.analysis.modes import plucked_string_modal_spectrum
from wavefd.solvers.string_1d import BoundaryMode, StringConfig, StringWaveSolver


def _pluck(n: int = 129, length: float = 0.64, amp: float = 0.016):
    x = np.linspace(0.0, length, n)
    mid = length / 2.0
    u0 = np.where(x <= mid, amp * x / mid, amp * (length - x) / mid)
    u0[0] = u0[-1] = 0.0
    return x, u0


def test_exact_comparison_small_error():
    x, u0 = _pluck()
    sol = StringWaveSolver(x, u0, StringConfig(cfl=0.5, n_periods=1.0)).solve(BoundaryMode.FIXED)
    cmp = compare_to_exact_string(sol)
    assert cmp["max_rel_l2"] < 5e-2


def test_convergence_order_at_least_two():
    conv = run_string_convergence(resolutions=(33, 65, 129), n_periods=0.5)
    assert conv.mean_order > 1.7
    # On a discrete eigenmode the observed rate is typically higher than two.
    assert np.all(np.diff(conv.rel_l2) < 0)


def test_modal_odd_harmonics_dominate():
    x, u0 = _pluck()
    sol = StringWaveSolver(x, u0, StringConfig(cfl=0.5, n_periods=0.25)).solve(BoundaryMode.FIXED)
    spec = plucked_string_modal_spectrum(sol, n_modes=20)
    odd_frac = float(np.sum(spec["energy_fraction_numerical"][0::2]))
    assert odd_frac > 0.95
    b_th = triangular_pluck_coefficients(0.016, 20)
    assert np.allclose(spec["b_analytic"], b_th)


def test_dispersion_long_wave_limit():
    disp = numerical_dispersion(c=250.0, dx=0.005, r=0.5)
    # Long-wave phase ratio should be near unity
    assert abs(disp["phase_ratio"][1] - 1.0) < 1e-3


def test_wav_export(tmp_path: Path):
    x, u0 = _pluck(n=65)
    sol = StringWaveSolver(x, u0, StringConfig(cfl=0.5, n_periods=1.0)).solve(BoundaryMode.FIXED)
    path = export_string_wav(
        midpoint_waveform(sol.u),
        float(sol.t[1] - sol.t[0]),
        tmp_path / "tone.wav",
        duration_s=0.2,
    )
    assert path.exists() and path.stat().st_size > 1000
