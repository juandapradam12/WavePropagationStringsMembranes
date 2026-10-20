#!/usr/bin/env python3
"""End-to-end simulation and figure pipeline for the wave-equation study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wavefd.analysis.audio import export_string_wav, midpoint_waveform
from wavefd.analysis.cfl_demo import run_cfl_stability_demo
from wavefd.analysis.diagnostics import (
    cfl_report,
    relative_energy_drift,
    standing_wave_period_estimate,
)
from wavefd.io import load_membrane_ic, load_string_ic, save_array
from wavefd.solvers.membrane_2d import MembraneConfig, MembraneWaveSolver
from wavefd.solvers.string_1d import BoundaryMode, StringConfig, StringWaveSolver
from wavefd.viz.extension_figures import generate_extension_figures
from wavefd.viz.figures import generate_all_figures


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--figure-dir", type=Path, default=ROOT / "figures")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--string-periods", type=float, default=1.0)
    p.add_argument("--membrane-periods", type=float, default=1.0)
    p.add_argument("--cfl-1d", type=float, default=0.5)
    p.add_argument("--cfl-2d", type=float, default=0.4)
    p.add_argument("--damping-gamma", type=float, default=500.0)
    p.add_argument("--audio-duration", type=float, default=1.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    x, u0 = load_string_ic(args.data_dir / "string_ic.dat")
    membrane_u0 = load_membrane_ic(args.data_dir / "membrane_ic.dat")

    string_cfg = StringConfig(n_periods=args.string_periods, cfl=args.cfl_1d)
    membrane_cfg = MembraneConfig(n_periods=args.membrane_periods, cfl=args.cfl_2d)
    damped_cfg = StringConfig(
        n_periods=max(args.string_periods, 2.0),
        cfl=args.cfl_1d,
        gamma=args.damping_gamma,
    )

    string_fixed = StringWaveSolver(x, u0, string_cfg).solve(mode=BoundaryMode.FIXED)
    string_driven = StringWaveSolver(x, u0, string_cfg).solve(mode=BoundaryMode.DRIVEN)
    string_free = StringWaveSolver(x, u0, string_cfg).solve(mode=BoundaryMode.FREE)
    string_damped = StringWaveSolver(x, u0, damped_cfg).solve(mode=BoundaryMode.FIXED)
    membrane = MembraneWaveSolver(membrane_u0, membrane_cfg).solve()

    # Pure (1,1) membrane mode for exact verification
    L_m = membrane_cfg.length
    xx = np.linspace(0.0, L_m, membrane_u0.shape[0])
    X, Y = np.meshgrid(xx, xx)
    mode_u0 = 0.01 * np.sin(np.pi * X / L_m) * np.sin(np.pi * Y / L_m)
    membrane_mode = MembraneWaveSolver(mode_u0, membrane_cfg).solve()

    cfl_demo = run_cfl_stability_demo()

    L = float(x[-1] - x[0])
    T1 = string_cfg.fundamental_period(L)
    T_est = standing_wave_period_estimate(string_fixed)
    dt = float(string_fixed.t[1] - string_fixed.t[0])
    dt_d = float(string_damped.t[1] - string_damped.t[0])

    wav_path = export_string_wav(
        midpoint_waveform(string_fixed.u),
        dt,
        args.results_dir / "string_midpoint.wav",
        duration_s=args.audio_duration,
    )
    wav_damped = export_string_wav(
        midpoint_waveform(string_damped.u),
        dt_d,
        args.results_dir / "string_midpoint_damped.wav",
        duration_s=args.audio_duration,
    )

    paths, validation = generate_all_figures(
        string_fixed=string_fixed,
        string_driven=string_driven,
        membrane=membrane,
        string_x0=x,
        string_u0=u0,
        membrane_u0=membrane_u0,
        out_dir=args.figure_dir,
    )
    ext_paths, ext_summary = generate_extension_figures(
        membrane_mode_sol=membrane_mode,
        membrane_strike=membrane,
        cfl_demo=cfl_demo,
        string_damped=string_damped,
        string_free=string_free,
        string_fixed=string_fixed,
        out_dir=args.figure_dir,
    )
    paths.update(ext_paths)

    summary = {
        "string": {
            "length_m": L,
            "wave_speed_m_s": string_cfg.c,
            "dx_m": float(x[1] - x[0]),
            "dt_s": dt,
            "courant": string_fixed.r,
            "n_time": int(string_fixed.t.size),
            "fundamental_period_s": T1,
            "fundamental_frequency_hz": 1.0 / T1,
            "estimated_period_s": T_est,
            "period_relative_error": abs(T_est - T1) / T1 if T_est == T_est else None,
            "energy_drift": relative_energy_drift(string_fixed.energy),
            "cfl": cfl_report(string_fixed),
            "audio_wav": str(wav_path.relative_to(ROOT)),
            "audio_wav_damped": str(wav_damped.relative_to(ROOT)),
        },
        "membrane": {
            "length_m": membrane_cfg.length,
            "wave_speed_m_s": membrane_cfg.c,
            "dx_m": membrane.dx,
            "dt_s": float(membrane.t[1] - membrane.t[0]),
            "courant": membrane.r,
            "n_time": int(membrane.t.size),
            "mode_11_period_s": membrane_cfg.fundamental_period(),
            "energy_drift": relative_energy_drift(membrane.energy),
            "cfl": cfl_report(membrane),
        },
        "driven_string": {
            "energy_drift": relative_energy_drift(string_driven.energy),
            "final_over_initial_energy": float(
                string_driven.energy[-1] / string_driven.energy[0]
            ),
        },
        "validation": validation,
        "extensions": ext_summary,
    }

    summary_path = args.results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    for name, sol, fracs in [
        ("string_fixed", string_fixed, (0.0, 0.25, 0.5, 0.75, 1.0)),
        ("string_driven", string_driven, (0.0, 0.25, 0.5, 0.75, 1.0)),
        ("string_free", string_free, (0.0, 0.25, 0.5, 0.75, 1.0)),
    ]:
        for frac, idx in zip(fracs, sol.snapshot_indices(fracs)):
            save_array(args.results_dir / f"{name}_tfrac_{frac:.2f}.dat", sol.u[idx])

    for frac, idx in zip((0.0, 0.25, 0.5, 0.75), membrane.snapshot_indices((0.0, 0.25, 0.5, 0.75))):
        save_array(args.results_dir / f"membrane_tfrac_{frac:.2f}.dat", membrane.u[idx])

    print("Simulation summary")
    print(json.dumps(summary, indent=2))
    print("\nFigures written:")
    for key, path in paths.items():
        print(f"  {key:20s} -> {path}")
    print(f"\nAudio written: {wav_path}")
    print(f"Damped audio written: {wav_damped}")


if __name__ == "__main__":
    main()
