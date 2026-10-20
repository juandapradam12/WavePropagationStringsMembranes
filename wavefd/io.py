"""I/O helpers for initial-condition grids."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_string_ic(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a plucked-string initial condition.

    Expects two columns: position ``x`` [m] and transverse displacement ``u`` [m].
    """
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected Nx2 array in {path}, got shape {data.shape}")
    x = data[:, 0].astype(float)
    u0 = data[:, 1].astype(float)
    if not np.isclose(u0[0], 0.0) or not np.isclose(u0[-1], 0.0):
        raise ValueError("String endpoints must vanish for Dirichlet boundaries.")
    return x, u0


def load_membrane_ic(path: str | Path) -> np.ndarray:
    """Load a square membrane initial displacement field ``u(x, y, t=0)``."""
    u0 = np.loadtxt(path).astype(float)
    if u0.ndim != 2 or u0.shape[0] != u0.shape[1]:
        raise ValueError(f"Expected square grid in {path}, got shape {u0.shape}")
    if not (
        np.allclose(u0[0, :], 0.0)
        and np.allclose(u0[-1, :], 0.0)
        and np.allclose(u0[:, 0], 0.0)
        and np.allclose(u0[:, -1], 0.0)
    ):
        raise ValueError("Membrane boundary must vanish for Dirichlet conditions.")
    return u0


def save_array(path: str | Path, array: np.ndarray) -> None:
    """Persist a NumPy array as whitespace-delimited text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, array)
