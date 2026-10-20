"""Modal decomposition of the vibrating string."""

from __future__ import annotations

import numpy as np

from wavefd.analysis.exact import triangular_pluck_coefficients
from wavefd.solvers.string_1d import StringSolution


def sine_mode_matrix(x: np.ndarray, length: float, n_modes: int) -> np.ndarray:
    """Rows are the Dirichlet eigenfunctions ``sin(n π x / L)``, shape ``(n_modes, n_x)``."""
    n = np.arange(1, n_modes + 1, dtype=float)
    return np.sin(np.outer(n, np.pi * x / length))


def project_sine_coefficients(u: np.ndarray, x: np.ndarray, length: float, n_modes: int) -> np.ndarray:
    """Discrete sine coefficients via trapezoidal projection.

    For a single profile ``u(x)`` returns shape ``(n_modes,)``.
    For a time series ``u(t, x)`` returns shape ``(n_t, n_modes)``.
    """
    modes = sine_mode_matrix(x, length, n_modes)
    # ∫ sin^2 = L/2  ⇒  b_n = (2/L) ∫ u sin
    if u.ndim == 1:
        return (2.0 / length) * np.trapezoid(u[None, :] * modes, x, axis=1)
    return (2.0 / length) * np.trapezoid(u[:, None, :] * modes[None, :, :], x, axis=2)


def modal_energy_weights(coefficients: np.ndarray, c: float, length: float) -> np.ndarray:
    """Modal energy proxy proportional to ``(n π c / L)^2 b_n^2`` (potential of a pure mode at max stretch)."""
    n_modes = coefficients.shape[-1]
    n = np.arange(1, n_modes + 1, dtype=float)
    omega = n * np.pi * c / length
    return 0.5 * (omega**2) * (coefficients**2) * (length / 2.0)


def plucked_string_modal_spectrum(
    sol: StringSolution,
    n_modes: int = 40,
) -> dict[str, np.ndarray]:
    """Compare discrete modal content of the IC to the analytic triangular-pluck series."""
    length = float(sol.x[-1] - sol.x[0])
    amp = float(np.max(np.abs(sol.u[0])))
    b_num = project_sine_coefficients(sol.u[0], sol.x, length, n_modes)
    b_th = triangular_pluck_coefficients(amp, n_modes)
    e_num = modal_energy_weights(b_num, sol.config.c, length)
    e_th = modal_energy_weights(b_th, sol.config.c, length)
    return {
        "n": np.arange(1, n_modes + 1),
        "b_numerical": b_num,
        "b_analytic": b_th,
        "energy_numerical": e_num,
        "energy_analytic": e_th,
        "energy_fraction_numerical": e_num / np.sum(e_num),
        "energy_fraction_analytic": e_th / np.sum(e_th),
    }
