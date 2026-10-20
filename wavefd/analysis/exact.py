"""Analytic reference solutions for the vibrating string."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from wavefd.solvers.string_1d import StringSolution


def triangular_pluck_coefficients(amplitude: float, n_modes: int) -> np.ndarray:
    """Fourier sine coefficients for a centered triangular pluck of height ``amplitude``.

    For a pluck peaking at mid-span,
    :math:`b_n = 8A/(n^2\\pi^2)` for odd :math:`n` and :math:`b_n=0` for even :math:`n`.
    """
    n = np.arange(1, n_modes + 1, dtype=float)
    b = np.zeros_like(n)
    odd = n % 2 == 1
    b[odd] = 8.0 * amplitude / (n[odd] ** 2 * np.pi**2)
    return b


def exact_string_modal(
    x: np.ndarray,
    t: np.ndarray,
    *,
    length: float,
    c: float,
    amplitude: float,
    n_modes: int = 200,
) -> np.ndarray:
    """Modal-series solution for a centered triangular pluck released from rest.

    Useful for spectrum checks; prefer :func:`exact_string_dalembert` when a
    pointwise-exact field is required, because the kink converges slowly.
    """
    b = triangular_pluck_coefficients(amplitude, n_modes)
    n = np.arange(1, n_modes + 1, dtype=float)
    omega = n * np.pi * c / length
    spatial = np.sin(np.outer(n, np.pi * x / length))
    temporal = np.cos(np.outer(t, omega))
    return (temporal * b) @ spatial


def _odd_periodic_extension(x: np.ndarray, u0: np.ndarray, length: float) -> interp1d:
    """Build the odd 2L-periodic extension of a Dirichlet profile on ``[0, L]``."""
    # Sample on [-L, L] by odd reflection, then rely on periodic wrap of period 2L.
    x_ext = np.concatenate([-x[-1:0:-1], x])
    u_ext = np.concatenate([-u0[-1:0:-1], u0])
    # Ensure endpoints: u(0)=u(L)=u(-L)=0
    return interp1d(x_ext, u_ext, kind="linear", fill_value="extrapolate", assume_sorted=True)


def exact_string_dalembert(
    x: np.ndarray,
    t: np.ndarray,
    u0: np.ndarray,
    *,
    length: float,
    c: float,
) -> np.ndarray:
    """Exact fixed-end solution via d'Alembert on the odd 2L-periodic extension.

    With vanishing initial velocity,
    :math:`u(x,t)=\\tfrac12\\bigl[F(x-ct)+F(x+ct)\\bigr]`, where :math:`F` is the
    odd extension of :math:`u_0` with period :math:`2L`.
    """
    F = _odd_periodic_extension(x, u0, length)
    period = 2.0 * length

    def eval_F(z: np.ndarray) -> np.ndarray:
        # Wrap into [-L, L]
        z_wrapped = (z + length) % period - length
        return F(z_wrapped)

    out = np.empty((t.size, x.size), dtype=float)
    for i, ti in enumerate(t):
        out[i] = 0.5 * (eval_F(x - c * ti) + eval_F(x + c * ti))
    return out


def exact_pure_mode(
    x: np.ndarray,
    t: np.ndarray,
    *,
    length: float,
    c: float,
    n: int = 1,
    amplitude: float = 0.01,
) -> np.ndarray:
    """Exact standing wave :math:`A\\cos(\\omega_n t)\\sin(n\\pi x/L)`."""
    omega = n * np.pi * c / length
    spatial = amplitude * np.sin(n * np.pi * x / length)
    return np.cos(omega * t)[:, None] * spatial[None, :]


def l2_error_series(numerical: np.ndarray, exact: np.ndarray, dx: float) -> np.ndarray:
    """Discrete L2 error ``||u_num - u_exact||_2`` at each time index."""
    diff = numerical - exact
    return np.sqrt(dx * np.sum(diff * diff, axis=1))


def relative_l2_error(
    numerical: np.ndarray,
    exact: np.ndarray,
    dx: float,
    *,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Relative L2 error normalized by a fixed reference norm (default: exact at t=0)."""
    err = l2_error_series(numerical, exact, dx)
    ref = exact[0] if reference is None else reference
    denom = np.sqrt(dx * np.sum(ref * ref))
    if denom < 1e-30:
        return np.full_like(err, np.nan)
    return err / denom


def compare_to_exact_string(sol: StringSolution) -> dict[str, np.ndarray | float]:
    """Compare a fixed-string run to the d'Alembert reference built from its IC."""
    exact = exact_string_dalembert(
        sol.x,
        sol.t,
        sol.u[0],
        length=float(sol.x[-1] - sol.x[0]),
        c=sol.config.c,
    )
    dx = sol.dx
    rel = relative_l2_error(sol.u, exact, dx)
    abs_err = l2_error_series(sol.u, exact, dx)
    return {
        "exact": exact,
        "abs_l2": abs_err,
        "rel_l2": rel,
        "max_rel_l2": float(np.nanmax(rel)),
        "final_rel_l2": float(rel[-1]) if rel[-1] == rel[-1] else float("nan"),
    }
