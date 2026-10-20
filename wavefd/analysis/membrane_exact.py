"""Exact rectangular-membrane references."""

from __future__ import annotations

import numpy as np

from wavefd.solvers.membrane_2d import MembraneSolution


def exact_membrane_mode(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    *,
    length: float,
    c: float,
    m: int = 1,
    n: int = 1,
    amplitude: float = 0.01,
) -> np.ndarray:
    """Exact clamped mode :math:`A\\cos(\\omega_{mn}t)\\sin(m\\pi x/L)\\sin(n\\pi y/L)`."""
    omega = np.pi * c * np.sqrt((m / length) ** 2 + (n / length) ** 2)
    sx = np.sin(m * np.pi * x / length)
    sy = np.sin(n * np.pi * y / length)
    spatial = amplitude * sy[:, None] * sx[None, :]
    return np.cos(omega * t)[:, None, None] * spatial[None, :, :]


def relative_l2_membrane(
    numerical: np.ndarray,
    exact: np.ndarray,
    dx: float,
) -> np.ndarray:
    """Relative L2 error normalized by the initial exact norm."""
    diff = numerical - exact
    err = np.sqrt(dx * dx * np.sum(diff * diff, axis=(1, 2)))
    denom = np.sqrt(dx * dx * np.sum(exact[0] * exact[0]))
    if denom < 1e-30:
        return np.full_like(err, np.nan)
    return err / denom


def compare_membrane_mode(sol: MembraneSolution, m: int = 1, n: int = 1) -> dict:
    """Compare a membrane run to the analytic rectangular mode matching its IC peak."""
    amp = float(np.max(np.abs(sol.u[0])))
    exact = exact_membrane_mode(
        sol.x,
        sol.y,
        sol.t,
        length=sol.config.length,
        c=sol.config.c,
        m=m,
        n=n,
        amplitude=amp,
    )
    rel = relative_l2_membrane(sol.u, exact, sol.dx)
    return {
        "exact": exact,
        "rel_l2": rel,
        "max_rel_l2": float(np.nanmax(rel)),
        "final_rel_l2": float(rel[-1]),
        "m": m,
        "n": n,
        "amplitude": amp,
    }
