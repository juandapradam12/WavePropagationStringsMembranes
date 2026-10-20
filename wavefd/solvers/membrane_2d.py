"""Explicit finite-difference solver for the 2D wave equation on a square membrane.

.. math::

    \\partial_{tt} u = c^2 \\nabla^2 u,
    \\qquad (x,y)\\in(0,L)^2,\\ t>0,

with clamped edges :math:`u|_{\\partial\\Omega}=0` and vanishing initial velocity.
The five-point Laplacian scheme with Courant number
:math:`r = c\\,\\Delta t / \\Delta x` (for equal spacing) is stable when
:math:`r \\le 1/\\sqrt{2}`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MembraneConfig:
    """Physical and numerical parameters for a square drumhead."""

    c: float = 250.0
    length: float = 0.5  # side length [m]
    dt: float | None = None
    n_periods: float = 0.5  # duration in units of the (1,1) mode period
    cfl: float = 0.4  # target Courant number (must be <= 1/sqrt(2))
    snapshots_per_period: int = 8

    def mode_frequency(self, m: int = 1, n: int = 1) -> float:
        """Analytic frequency of the rectangular mode :math:`(m,n)`."""
        return 0.5 * self.c * np.sqrt((m / self.length) ** 2 + (n / self.length) ** 2)

    def fundamental_period(self) -> float:
        return 1.0 / self.mode_frequency(1, 1)


@dataclass
class MembraneSolution:
    """Container for a completed 2D simulation."""

    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    u: np.ndarray  # shape (n_t, n_y, n_x)
    r: float
    energy: np.ndarray
    config: MembraneConfig

    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0])

    def snapshot_indices(self, fractions: tuple[float, ...] = (0.0, 0.125, 0.25, 0.5)) -> list[int]:
        t_end = self.t[-1]
        return [int(np.argmin(np.abs(self.t - f * t_end))) for f in fractions]


class MembraneWaveSolver:
    """Advance the 2D wave equation with an explicit leapfrog scheme."""

    def __init__(self, u0: np.ndarray, config: MembraneConfig | None = None):
        self.u0 = np.asarray(u0, dtype=float)
        if self.u0.ndim != 2 or self.u0.shape[0] != self.u0.shape[1]:
            raise ValueError("Initial field must be a square 2D array.")
        self.config = config or MembraneConfig()
        n = self.u0.shape[0]
        self.dx = self.config.length / (n - 1)
        self.x = np.linspace(0.0, self.config.length, n)
        self.y = np.linspace(0.0, self.config.length, n)

    def _time_step(self) -> float:
        r_max = 1.0 / np.sqrt(2.0)
        if self.config.cfl > r_max + 1e-12:
            raise ValueError(f"2D CFL must satisfy r <= 1/sqrt(2) ≈ {r_max:.4f}.")
        if self.config.dt is not None:
            return float(self.config.dt)
        return self.config.cfl * self.dx / self.config.c

    def _n_steps(self, dt: float) -> int:
        t_end = self.config.n_periods * self.config.fundamental_period()
        return max(2, int(np.ceil(t_end / dt)) + 1)

    def solve(self) -> MembraneSolution:
        dt = self._time_step()
        r = self.config.c * dt / self.dx
        r_max = 1.0 / np.sqrt(2.0)
        if r > r_max + 1e-12:
            raise ValueError(f"Unstable 2D Courant number r={r:.4f} > 1/sqrt(2).")

        n_t = self._n_steps(dt)
        n = self.u0.shape[0]
        u = np.zeros((n_t, n, n), dtype=float)
        t = np.arange(n_t, dtype=float) * dt
        r2 = r * r

        u[0] = self.u0
        u[0, 0, :] = u[0, -1, :] = u[0, :, 0] = u[0, :, -1] = 0.0

        # First step from rest
        lap0 = _laplacian(u[0])
        u[1, 1:-1, 1:-1] = u[0, 1:-1, 1:-1] + 0.5 * r2 * lap0
        u[1, 0, :] = u[1, -1, :] = u[1, :, 0] = u[1, :, -1] = 0.0

        # Leapfrog: u^{n+1} = 2 u^n - u^{n-1} + r^2 Δ_h u^n
        for k in range(1, n_t - 1):
            lap = _laplacian(u[k])
            u[k + 1, 1:-1, 1:-1] = (
                2.0 * u[k, 1:-1, 1:-1] - u[k - 1, 1:-1, 1:-1] + r2 * lap
            )
            u[k + 1, 0, :] = u[k + 1, -1, :] = 0.0
            u[k + 1, :, 0] = u[k + 1, :, -1] = 0.0

        energy = discrete_membrane_energy(u, self.dx, dt, self.config.c)
        return MembraneSolution(
            x=self.x.copy(),
            y=self.y.copy(),
            t=t,
            u=u,
            r=r,
            energy=energy,
            config=self.config,
        )


def _laplacian(field: np.ndarray) -> np.ndarray:
    """Five-point Laplacian on the interior, without the 1/dx^2 factor.

    Returns an (ny-2, nx-2) array equal to
    ``u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{i,j}``.
    """
    return (
        field[2:, 1:-1]
        + field[:-2, 1:-1]
        + field[1:-1, 2:]
        + field[1:-1, :-2]
        - 4.0 * field[1:-1, 1:-1]
    )


def discrete_membrane_energy(u: np.ndarray, dx: float, dt: float, c: float) -> np.ndarray:
    """Discrete mechanical energy of a vibrating membrane."""
    ux = np.diff(u, axis=2) / dx
    uy = np.diff(u, axis=1) / dx
    # Align shapes by dropping the last row/col of the mismatched derivative
    potential = 0.5 * c * c * dx * dx * (
        np.sum(ux[:, :-1, :] ** 2, axis=(1, 2)) + np.sum(uy[:, :, :-1] ** 2, axis=(1, 2))
    )

    ut = np.diff(u, axis=0) / dt
    kinetic_half = 0.5 * dx * dx * np.sum(ut * ut, axis=(1, 2))

    kinetic = np.empty(u.shape[0], dtype=float)
    kinetic[0] = kinetic_half[0]
    kinetic[-1] = kinetic_half[-1]
    kinetic[1:-1] = 0.5 * (kinetic_half[:-1] + kinetic_half[1:])
    return kinetic + potential
