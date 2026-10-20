"""Publication-style figures and animations for the wave-equation workflow."""

from .figures import generate_all_figures
from .animations import animate_membrane_gif, animate_string_gif

__all__ = ["generate_all_figures", "animate_string_gif", "animate_membrane_gif"]
