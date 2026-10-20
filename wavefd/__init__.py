"""Finite-difference solvers for the one- and two-dimensional wave equation."""

from .solvers.string_1d import StringWaveSolver, StringConfig
from .solvers.membrane_2d import MembraneWaveSolver, MembraneConfig

__all__ = [
    "StringWaveSolver",
    "StringConfig",
    "MembraneWaveSolver",
    "MembraneConfig",
]

__version__ = "1.0.0"
