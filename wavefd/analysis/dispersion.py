"""Numerical dispersion relation for the 1D leapfrog scheme."""

from __future__ import annotations

import numpy as np


def numerical_dispersion(
    *,
    c: float,
    dx: float,
    r: float,
    n_k: int = 200,
) -> dict[str, np.ndarray]:
    """Exact and discrete frequencies for resolvable wavenumbers.

    The leapfrog scheme with Courant number :math:`r=c\\Delta t/\\Delta x` has

    .. math::

        \\omega_{\\mathrm{num}}(\\kappa)
        =
        \\frac{2}{\\Delta t}\\arcsin\\!\\bigl(r\\sin(\\kappa\\Delta x/2)\\bigr),

    while the continuum branch is :math:`\\omega = c\\kappa`. The Nyquist
    mode is :math:`\\kappa_{\\max}=\\pi/\\Delta x`.
    """
    dt = r * dx / c
    kappa = np.linspace(0.0, np.pi / dx, n_k)
    arg = r * np.sin(0.5 * kappa * dx)
    # Clip tiny overshoots from floating point at the Nyquist end when r→1
    arg = np.clip(arg, -1.0, 1.0)
    omega_num = (2.0 / dt) * np.arcsin(arg)
    omega_exact = c * kappa
    with np.errstate(divide="ignore", invalid="ignore"):
        phase_ratio = np.where(omega_exact > 0, omega_num / omega_exact, 1.0)
    return {
        "kappa": kappa,
        "omega_numerical": omega_num,
        "omega_exact": omega_exact,
        "phase_ratio": phase_ratio,
        "dx": np.array(dx),
        "dt": np.array(dt),
        "r": np.array(r),
    }
