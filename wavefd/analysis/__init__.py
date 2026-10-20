"""Post-processing diagnostics for wave simulations."""

from .diagnostics import (
    cfl_report,
    relative_energy_drift,
    standing_wave_period_estimate,
)

__all__ = [
    "cfl_report",
    "relative_energy_drift",
    "standing_wave_period_estimate",
]
