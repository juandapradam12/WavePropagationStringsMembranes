"""Rectangular modal decomposition of a square membrane field."""

from __future__ import annotations

import numpy as np

from wavefd.solvers.membrane_2d import MembraneSolution


def project_rectangular_modes(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    length: float,
    n_modes: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a clamped field onto ``sin(mπx/L) sin(nπy/L)`` modes.

    Returns
    -------
    m_idx, n_idx:
        1-based mode index grids of shape ``(n_modes, n_modes)``.
    coefficients:
        Amplitude matrix :math:`a_{mn}` of the same shape.
    """
    # a_mn = (4/L^2) ∬ u sin(mπx/L) sin(nπy/L) dx dy
    coeffs = np.zeros((n_modes, n_modes), dtype=float)
    for mi, m in enumerate(range(1, n_modes + 1)):
        sx = np.sin(m * np.pi * x / length)
        for ni, n in enumerate(range(1, n_modes + 1)):
            sy = np.sin(n * np.pi * y / length)
            # trapz in x then y
            inner = np.trapezoid(field * sx[None, :], x, axis=1)
            coeffs[ni, mi] = (4.0 / length**2) * np.trapezoid(inner * sy, y)
    m_idx, n_idx = np.meshgrid(np.arange(1, n_modes + 1), np.arange(1, n_modes + 1))
    return m_idx, n_idx, coeffs


def membrane_modal_spectrum(sol: MembraneSolution, n_modes: int = 6) -> dict[str, np.ndarray]:
    """Modal amplitudes and energy fractions of the membrane initial strike."""
    m_idx, n_idx, coeffs = project_rectangular_modes(
        sol.u[0], sol.x, sol.y, sol.config.length, n_modes=n_modes
    )
    # Modal energy proxy ∝ ω_{mn}^2 a_{mn}^2 with ω = π c sqrt((m/L)^2+(n/L)^2)
    omega = (
        np.pi
        * sol.config.c
        * np.sqrt((m_idx / sol.config.length) ** 2 + (n_idx / sol.config.length) ** 2)
    )
    energy = 0.5 * (omega**2) * (coeffs**2) * (sol.config.length**2 / 4.0)
    total = float(np.sum(energy))
    frac = energy / total if total > 0 else energy
    return {
        "m": m_idx,
        "n": n_idx,
        "coefficients": coeffs,
        "energy": energy,
        "energy_fraction": frac,
    }
