"""Explicit finite-difference solver for the 1D wave equation.

The continuous problem on a taut string is

.. math::

    \\partial_{tt} u = c^2 \\partial_{xx} u,
    \\qquad x \\in (0, L),\\ t > 0,

with initial data :math:`u(x,0)=u_0(x)`, :math:`\\partial_t u(x,0)=0`, and
Dirichlet (or driven) boundary conditions.

The second-order centered scheme with Courant number
:math:`r = c\\,\\Delta t / \\Delta x` reads

.. math::

    u^{n+1}_i = 2(1-r^2)\\,u^n_i - u^{n-1}_i
    + r^2\\,(u^n_{i+1}+u^n_{i-1}),

which is stable for :math:`r \\le 1` (CFL condition).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np


class BoundaryMode(str, Enum):
    FIXED = "fixed"
    DRIVEN = "driven"


@dataclass(frozen=True)
class StringConfig:
    """Physical and numerical parameters for a 1D string."""

    c: float = 250.0  # wave speed [m/s]
    dt: float | None = None  # time step [s]; chosen from CFL if None
    n_periods: float = 1.0  # simulated duration in fundamental periods
    cfl: float = 0.5  # target Courant number when dt is auto-selected
    snapshots_per_period: int = 8  # evenly spaced output frames per period

    def fundamental_frequency(self, length: float) -> float:
        return self.c / (2.0 * length)

    def fundamental_period(self, length: float) -> float:
        return 1.0 / self.fundamental_frequency(length)


@dataclass
class StringSolution:
    """Container for a completed 1D simulation."""

    x: np.ndarray
    t: np.ndarray
    u: np.ndarray  # shape (n_t, n_x)
    r: float
    energy: np.ndarray
    config: StringConfig
    mode: BoundaryMode

    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0])

    def snapshot_indices(self, fractions: tuple[float, ...] = (0.0, 0.125, 0.25, 0.5)) -> list[int]:
        """Map fractions of the total simulated time to nearest time indices."""
        t_end = self.t[-1]
        return [int(np.argmin(np.abs(self.t - f * t_end))) for f in fractions]


class StringWaveSolver:
    """Advance the 1D wave equation with an explicit leapfrog scheme."""

    def __init__(self, x: np.ndarray, u0: np.ndarray, config: StringConfig | None = None):
        if x.shape != u0.shape:
            raise ValueError("x and u0 must share the same shape.")
        if len(x) < 3:
            raise ValueError("Need at least three spatial nodes.")
        self.x = np.asarray(x, dtype=float)
        self.u0 = np.asarray(u0, dtype=float)
        self.config = config or StringConfig()
        self.dx = float(self.x[1] - self.x[0])
        if not np.allclose(np.diff(self.x), self.dx):
            raise ValueError("Spatial grid must be uniform.")
        self.length = float(self.x[-1] - self.x[0])

    def _time_step(self) -> float:
        if self.config.dt is not None:
            return float(self.config.dt)
        return self.config.cfl * self.dx / self.config.c

    def _n_steps(self, dt: float) -> int:
        t_end = self.config.n_periods * self.config.fundamental_period(self.length)
        return max(2, int(np.ceil(t_end / dt)) + 1)

    def _courant(self, dt: float) -> float:
        return self.config.c * dt / self.dx

    def solve(
        self,
        mode: BoundaryMode | str = BoundaryMode.FIXED,
        drive: Callable[[float], float] | None = None,
    ) -> StringSolution:
        """Integrate the string dynamics.

        Parameters
        ----------
        mode:
            ``fixed`` — homogeneous Dirichlet at both ends.
            ``driven`` — left end fixed, right end prescribed by ``drive(t)``.
        drive:
            Boundary forcing for the right endpoint. Defaults to a sinusoidal
            drive at the fundamental frequency with amplitude equal to the
            initial pluck height.
        """
        mode = BoundaryMode(mode)
        dt = self._time_step()
        r = self._courant(dt)
        if r > 1.0 + 1e-12:
            raise ValueError(f"Unstable Courant number r={r:.4f} > 1.")

        n_t = self._n_steps(dt)
        n_x = self.x.size
        u = np.zeros((n_t, n_x), dtype=float)
        t = np.arange(n_t, dtype=float) * dt

        amp = float(np.max(np.abs(self.u0)))
        omega = 2.0 * np.pi * self.config.fundamental_frequency(self.length)
        if drive is None:
            drive = lambda tau: amp * np.sin(omega * tau)

        # t = 0
        u[0] = self.u0
        u[0, 0] = 0.0
        if mode is BoundaryMode.FIXED:
            u[0, -1] = 0.0
        else:
            u[0, -1] = float(drive(0.0))

        # First step from rest (centered, second-order)
        r2 = r * r
        interior = slice(1, -1)
        lap0 = u[0, 2:] - 2.0 * u[0, 1:-1] + u[0, :-2]
        u[1, interior] = u[0, interior] + 0.5 * r2 * lap0
        u[1, 0] = 0.0
        u[1, -1] = 0.0 if mode is BoundaryMode.FIXED else float(drive(t[1]))

        # Leapfrog: u^{n+1} = 2 u^n - u^{n-1} + r^2 Δ_h u^n
        for n in range(1, n_t - 1):
            lap = u[n, 2:] - 2.0 * u[n, 1:-1] + u[n, :-2]
            u[n + 1, interior] = 2.0 * u[n, interior] - u[n - 1, interior] + r2 * lap
            u[n + 1, 0] = 0.0
            if mode is BoundaryMode.FIXED:
                u[n + 1, -1] = 0.0
            else:
                u[n + 1, -1] = float(drive(t[n + 1]))

        energy = discrete_string_energy(u, self.dx, dt, self.config.c)
        return StringSolution(
            x=self.x.copy(),
            t=t,
            u=u,
            r=r,
            energy=energy,
            config=self.config,
            mode=mode,
        )


def discrete_string_energy(u: np.ndarray, dx: float, dt: float, c: float) -> np.ndarray:
    """Second-order discrete mechanical energy of a vibrating string.

    Uses centered differences for :math:`\\partial_x u` and a staggered
    approximation for :math:`\\partial_t u` between consecutive frames, then
    averages so the returned array aligns with each time index.
    """
    # Spatial potential: (c^2/2) ∫ (u_x)^2 dx
    ux = np.diff(u, axis=1) / dx
    potential = 0.5 * c * c * dx * np.sum(ux * ux, axis=1)

    # Kinetic: (1/2) ∫ (u_t)^2 dx between n and n+1
    ut = np.diff(u, axis=0) / dt
    kinetic_half = 0.5 * dx * np.sum(ut * ut, axis=1)

    kinetic = np.empty(u.shape[0], dtype=float)
    kinetic[0] = kinetic_half[0]
    kinetic[-1] = kinetic_half[-1]
    kinetic[1:-1] = 0.5 * (kinetic_half[:-1] + kinetic_half[1:])
    return kinetic + potential
