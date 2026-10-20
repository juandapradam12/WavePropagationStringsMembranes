"""Explicit finite-difference solver for the 1D wave equation.

The continuous problem on a taut string is

.. math::

    \\partial_{tt} u = c^2 \\partial_{xx} u - \\gamma \\partial_t u,
    \\qquad x \\in (0, L),\\ t > 0,

with initial data :math:`u(x,0)=u_0(x)`, :math:`\\partial_t u(x,0)=0`, and
Dirichlet, driven, or Neumann boundary conditions.

The second-order centered scheme with Courant number
:math:`r = c\\,\\Delta t / \\Delta x` is stable for :math:`r \\le 1`
when :math:`\\gamma=0`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np


class BoundaryMode(str, Enum):
    FIXED = "fixed"
    DRIVEN = "driven"
    FREE = "free"  # left Dirichlet, right Neumann (u_x = 0)


@dataclass(frozen=True)
class StringConfig:
    """Physical and numerical parameters for a 1D string."""

    c: float = 250.0  # wave speed [m/s]
    dt: float | None = None  # time step [s]; chosen from CFL if None
    n_periods: float = 1.0  # simulated duration in fundamental periods
    cfl: float = 0.5  # target Courant number when dt is auto-selected
    gamma: float = 0.0  # linear viscous damping [1/s]
    snapshots_per_period: int = 8  # evenly spaced output frames per period
    allow_unstable: bool = False  # permit r > 1 for CFL blow-up demos

    def fundamental_frequency(self, length: float) -> float:
        """Fixed-fixed fundamental; for free end use the quarter-wave formula separately."""
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
            ``free`` — left end fixed, right end Neumann (:math:`u_x=0`).
        drive:
            Boundary forcing for the right endpoint in driven mode. Defaults to
            a sinusoid at the fixed-fixed fundamental frequency.
        """
        mode = BoundaryMode(mode)
        dt = self._time_step()
        r = self._courant(dt)
        if r > 1.0 + 1e-12 and not self.config.allow_unstable:
            raise ValueError(f"Unstable Courant number r={r:.4f} > 1.")

        n_t = self._n_steps(dt)
        n_x = self.x.size
        u = np.zeros((n_t, n_x), dtype=float)
        t = np.arange(n_t, dtype=float) * dt
        gamma = float(self.config.gamma)
        damp_p = 0.5 * gamma * dt
        denom = 1.0 + damp_p

        amp = float(np.max(np.abs(self.u0)))
        omega = 2.0 * np.pi * self.config.fundamental_frequency(self.length)
        if drive is None:
            drive = lambda tau: amp * np.sin(omega * tau)

        # t = 0
        u[0] = self.u0
        u[0, 0] = 0.0
        if mode is BoundaryMode.FIXED:
            u[0, -1] = 0.0
        elif mode is BoundaryMode.DRIVEN:
            u[0, -1] = float(drive(0.0))
        # FREE: keep IC at the right end (Neumann will be enforced in the update)

        r2 = r * r

        # First step from rest (u_t = 0 ⇒ odd extension in time)
        # With damping and u_t(0)=0 the starter remains the undamped centered step
        # to second order: u^1 = u^0 + (1/2) r^2 Δ u^0.
        if mode is BoundaryMode.FREE:
            lap0 = _laplacian_free_right(u[0])
            u[1, 1:] = u[0, 1:] + 0.5 * r2 * lap0
        else:
            lap0 = u[0, 2:] - 2.0 * u[0, 1:-1] + u[0, :-2]
            u[1, 1:-1] = u[0, 1:-1] + 0.5 * r2 * lap0
        u[1, 0] = 0.0
        if mode is BoundaryMode.FIXED:
            u[1, -1] = 0.0
        elif mode is BoundaryMode.DRIVEN:
            u[1, -1] = float(drive(t[1]))

        # Leapfrog with optional linear damping:
        # u^{n+1} = [2 u^n - (1 - γΔt/2) u^{n-1} + r^2 Δ u^n] / (1 + γΔt/2)
        for n in range(1, n_t - 1):
            if mode is BoundaryMode.FREE:
                lap = _laplacian_free_right(u[n])
                u[n + 1, 1:] = (
                    2.0 * u[n, 1:] - (1.0 - damp_p) * u[n - 1, 1:] + r2 * lap
                ) / denom
            else:
                lap = u[n, 2:] - 2.0 * u[n, 1:-1] + u[n, :-2]
                u[n + 1, 1:-1] = (
                    2.0 * u[n, 1:-1] - (1.0 - damp_p) * u[n - 1, 1:-1] + r2 * lap
                ) / denom
            u[n + 1, 0] = 0.0
            if mode is BoundaryMode.FIXED:
                u[n + 1, -1] = 0.0
            elif mode is BoundaryMode.DRIVEN:
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


def _laplacian_free_right(field: np.ndarray) -> np.ndarray:
    """Second difference on indices 1..n-1 with a Neumann ghost at the right end.

    Ghost condition :math:`u_x=0` ⇒ :math:`u_{N+1}=u_{N-1}`, so the stencil at
    the boundary node is :math:`2u_{N-1}-2u_N` in zero-based indexing
    (:math:`2u[-2]-2u[-1]`).
    """
    n = field.size
    lap = np.empty(n - 1, dtype=float)
    lap[:-1] = field[2:] - 2.0 * field[1:-1] + field[:-2]
    lap[-1] = 2.0 * field[-2] - 2.0 * field[-1]
    return lap


def discrete_string_energy(u: np.ndarray, dx: float, dt: float, c: float) -> np.ndarray:
    """Second-order discrete mechanical energy of a vibrating string.

    Uses centered differences for :math:`\\partial_x u` and a staggered
    approximation for :math:`\\partial_t u` between consecutive frames, then
    averages so the returned array aligns with each time index.
    """
    ux = np.diff(u, axis=1) / dx
    potential = 0.5 * c * c * dx * np.sum(ux * ux, axis=1)

    ut = np.diff(u, axis=0) / dt
    kinetic_half = 0.5 * dx * np.sum(ut * ut, axis=1)

    kinetic = np.empty(u.shape[0], dtype=float)
    kinetic[0] = kinetic_half[0]
    kinetic[-1] = kinetic_half[-1]
    kinetic[1:-1] = 0.5 * (kinetic_half[:-1] + kinetic_half[1:])
    return kinetic + potential
