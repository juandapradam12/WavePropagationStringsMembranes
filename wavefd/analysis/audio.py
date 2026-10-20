"""Audible readout of the vibrating-string mid-point."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def midpoint_waveform(u: np.ndarray, node_index: int | None = None) -> np.ndarray:
    """Extract the transverse displacement at mid-span."""
    if node_index is None:
        node_index = u.shape[1] // 2
    return np.asarray(u[:, node_index], dtype=float)


def export_string_wav(
    signal: np.ndarray,
    dt: float,
    path: str | Path,
    *,
    duration_s: float = 1.5,
    sample_rate: int = 44100,
) -> Path:
    """Tile one numerical period into an audible WAV at ``sample_rate``.

    The physical fundamental is already in the hearing range
    (:math:`f_1=c/(2L)\\approx 195\\,\\mathrm{Hz}` for the default string).
    The short simulation window is repeated to reach ``duration_s``, then
    resampled from ``1/dt`` to the target audio rate.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sig = np.asarray(signal, dtype=float)
    sig = sig - np.mean(sig)
    peak = np.max(np.abs(sig))
    if peak < 1e-30:
        raise ValueError("Mid-point signal is identically zero.")
    sig = sig / peak

    source_rate = 1.0 / dt
    n_needed = int(np.ceil(duration_s * source_rate))
    reps = int(np.ceil(n_needed / sig.size))
    tiled = np.tile(sig, reps)[:n_needed]

    # Rational resampling source_rate → sample_rate
    # Use integer factors via approximate gcd-friendly upsample/downsample.
    up = sample_rate
    down = int(round(source_rate))
    # Reduce fraction
    g = np.gcd(up, down)
    up //= g
    down //= g
    audio = resample_poly(tiled, up, down)
    audio = audio / (np.max(np.abs(audio)) + 1e-30)
    pcm = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    wavfile.write(path, sample_rate, pcm)
    return path
