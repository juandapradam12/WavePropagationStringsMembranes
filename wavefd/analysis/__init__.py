"""Post-processing diagnostics for wave simulations."""

from .audio import export_string_wav, midpoint_waveform
from .cfl_demo import run_cfl_stability_demo
from .convergence import ConvergenceResult, run_string_convergence
from .diagnostics import (
    cfl_report,
    relative_energy_drift,
    standing_wave_period_estimate,
)
from .dispersion import numerical_dispersion
from .exact import compare_to_exact_string, exact_string_dalembert, exact_string_modal
from .membrane_exact import compare_membrane_mode, exact_membrane_mode
from .membrane_modes import membrane_modal_spectrum
from .modes import plucked_string_modal_spectrum

__all__ = [
    "cfl_report",
    "relative_energy_drift",
    "standing_wave_period_estimate",
    "compare_to_exact_string",
    "exact_string_dalembert",
    "exact_string_modal",
    "run_string_convergence",
    "ConvergenceResult",
    "plucked_string_modal_spectrum",
    "numerical_dispersion",
    "export_string_wav",
    "midpoint_waveform",
    "run_cfl_stability_demo",
    "compare_membrane_mode",
    "exact_membrane_mode",
    "membrane_modal_spectrum",
]
