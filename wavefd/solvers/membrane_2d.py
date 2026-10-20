"""Explicit finite-difference solver for the 2D wave equation on a square membrane.

.. math::

    \\partial_{tt} u = c^2 \\nabla^2 u,
    \\qquad (x,y)\\in(0,L)^2,\\ t>0,

with either clamped edges :math:`u|_{\\partial\\Omega}=0` (Dirichlet) or free
edges :math:`\\partial_n u|_{\\partial\\Omega}=0` (Neumann), and vanishing
initial velocity. The five-point scheme with Courant number
:math:`r = c\\,\\Delta t / \\Delta x` is stable for :math:`r \\le 1/\\sqrt{2}`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class MembraneBoundary(str, Enum):
    CLAMPED = "clamped"
    FREE = "free"


@dataclass(frozen=True)
class MembraneConfig:
    """Physical and numerical parameters for a square drumhead."""

    c: float = 250.0
    length: float = 0.5  # side length [m]
    dt: float | None = None
    n_periods: float = 1.0  # duration in units of the clamped (1,1) mode period
    cfl: float = 0.4  # target Courant number (must be <= 1/sqrt(2))
    snapshots_per_period: int = 8

    def mode_frequency(self, m: int = 1, n: int = 1, *, free: bool = False) -> float:
        """Analytic frequency of rectangular mode :math:`(m,n)`.

        Clamped modes use sine factors with :math:`m,n\\ge 1`. Free (Neumann)
        modes use cosines with :math:`m,n\\ge 0`; the rigid mode `(0,0)` has
        frequency zero.
        """
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
    boundary: MembraneBoundary = MembraneBoundary.CLAMPED

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

    def solve(self, boundary: MembraneBoundary | str = MembraneBoundary.CLAMPED) -> MembraneSolution:
        boundary = MembraneBoundary(boundary)
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
        if boundary is MembraneBoundary.CLAMPED:
            u[0, 0, :] = u[0, -1, :] = u[0, :, 0] = u[0, :, -1] = 0.0

        # First step from rest
        if boundary is MembraneBoundary.CLAMPED:
            lap0 = _laplacian_dirichlet(u[0])
            u[1, 1:-1, 1:-1] = u[0, 1:-1, 1:-1] + 0.5 * r2 * lap0
            u[1, 0, :] = u[1, -1, :] = u[1, :, 0] = u[1, :, -1] = 0.0
        else:
            lap0 = _laplacian_neumann(u[0])
            u[1] = u[0] + 0.5 * r2 * lap0

        for k in range(1, n_t - 1):
            if boundary is MembraneBoundary.CLAMPED:
                lap = _laplacian_dirichlet(u[k])
                u[k + 1, 1:-1, 1:-1] = (
                    2.0 * u[k, 1:-1, 1:-1] - u[k - 1, 1:-1, 1:-1] + r2 * lap
                )
                u[k + 1, 0, :] = u[k + 1, -1, :] = 0.0
                u[k + 1, :, 0] = u[k + 1, :, -1] = 0.0
            else:
                lap = _laplacian_neumann(u[k])
                u[k + 1] = 2.0 * u[k] - u[k - 1] + r2 * lap

        energy = discrete_membrane_energy(u, self.dx, dt, self.config.c)
        return MembraneSolution(
            x=self.x.copy(),
            y=self.y.copy(),
            t=t,
            u=u,
            r=r,
            energy=energy,
            config=self.config,
            boundary=boundary,
        )


def _laplacian_dirichlet(field: np.ndarray) -> np.ndarray:
    """Five-point Laplacian on the interior (Dirichlet boundaries held fixed)."""
    return (
        field[2:, 1:-1]
        + field[:-2, 1:-1]
        + field[1:-1, 2:]
        + field[1:-1, :-2]
        - 4.0 * field[1:-1, 1:-1]
    )


def _laplacian_neumann(field: np.ndarray) -> np.ndarray:
    """Five-point Laplacian on the full grid with Neumann ghosts on all edges.

    Ghost reflections enforce :math:`\\partial_n u=0`:
    outside-left equals the first interior neighbour, and likewise on the
    other three sides. Corners see both reflected neighbours.
    """
    n = field.shape[0]
    # Pad with reflected boundary layers
    padded = np.empty((n + 2, n + 2), dtype=float)
    padded[1:-1, 1:-1] = field
    padded[0, 1:-1] = field[1, :]  # bottom ghost = first row above boundary
    padded[-1, 1:-1] = field[-2, :]
    padded[1:-1, 0] = field[:, 1]
    padded[1:-1, -1] = field[:, -2]
    # Corner ghosts (consistent with both Neumann conditions)
    padded[0, 0] = field[1, 1]
    padded[0, -1] = field[1, -2]
    padded[-1, 0] = field[-2, 1]
    padded[-1, -1] = field[-2, -2]

    return (
        padded[2:, 1:-1]
        + padded[:-2, 1:-1]
        + padded[1:-1, 2:]
        + padded[1:-1, :-2]
        - 4.0 * padded[1:-1, 1:-1]
    )


def discrete_membrane_energy(u: np.ndarray, dx: float, dt: float, c: float) -> np.ndarray:
    """Discrete mechanical energy of a vibrating membrane."""
    ux = np.diff(u, axis=2) / dx
    uy = np.diff(u, axis=1) / dx
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
