#!/usr/bin/env python3
"""End-to-end simulation and figure pipeline for the wave-equation study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wavefd.analysis.diagnostics import (
    cfl_report,
    relative_energy_drift,
    standing_wave_period_estimate,
)
from wavefd.io import load_membrane_ic, load_string_ic, save_array
from wavefd.solvers.membrane_2d import MembraneConfig, MembraneWaveSolver
from wavefd.solvers.string_1d import BoundaryMode, StringConfig, StringWaveSolver
from wavefd.viz.figures import generate_all_figures


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--figure-dir", type=Path, default=ROOT / "figures")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--string-periods", type=float, default=1.0)
    p.add_argument("--membrane-periods", type=float, default=0.5)
    p.add_argument("--cfl-1d", type=float, default=0.5)
    p.add_argument("--cfl-2d", type=float, default=0.4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    x, u0 = load_string_ic(args.data_dir / "string_ic.dat")
    membrane_u0 = load_membrane_ic(args.data_dir / "membrane_ic.dat")

    string_cfg = StringConfig(n_periods=args.string_periods, cfl=args.cfl_1d)
    membrane_cfg = MembraneConfig(n_periods=args.membrane_periods, cfl=args.cfl_2d)

    string_fixed = StringWaveSolver(x, u0, string_cfg).solve(mode=BoundaryMode.FIXED)
    string_driven = StringWaveSolver(x, u0, string_cfg).solve(mode=BoundaryMode.DRIVEN)
    membrane = MembraneWaveSolver(membrane_u0, membrane_cfg).solve()

    # Persist compact diagnostics and a few snapshots for reproducibility
    L = float(x[-1] - x[0])
    T1 = string_cfg.fundamental_period(L)
    T_est = standing_wave_period_estimate(string_fixed)
    summary = {
        "string": {
            "length_m": L,
            "wave_speed_m_s": string_cfg.c,
            "dx_m": float(x[1] - x[0]),
            "dt_s": float(string_fixed.t[1] - string_fixed.t[0]),
            "courant": string_fixed.r,
            "n_time": int(string_fixed.t.size),
            "fundamental_period_s": T1,
            "estimated_period_s": T_est,
            "period_relative_error": abs(T_est - T1) / T1 if T_est == T_est else None,
            "energy_drift": relative_energy_drift(string_fixed.energy),
            "cfl": cfl_report(string_fixed),
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
    }

    summary_path = args.results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Sparse snapshots for optional downstream use
    for name, sol, fracs in [
        ("string_fixed", string_fixed, (0.0, 0.25, 0.5, 0.75, 1.0)),
        ("string_driven", string_driven, (0.0, 0.25, 0.5, 0.75, 1.0)),
    ]:
        for frac, idx in zip(fracs, sol.snapshot_indices(fracs)):
            save_array(args.results_dir / f"{name}_tfrac_{frac:.2f}.dat", sol.u[idx])

    for frac, idx in zip((0.0, 0.25, 0.5, 0.75), membrane.snapshot_indices((0.0, 0.25, 0.5, 0.75))):
        save_array(args.results_dir / f"membrane_tfrac_{frac:.2f}.dat", membrane.u[idx])

    paths = generate_all_figures(
        string_fixed=string_fixed,
        string_driven=string_driven,
        membrane=membrane,
        string_x0=x,
        string_u0=u0,
        membrane_u0=membrane_u0,
        out_dir=args.figure_dir,
    )

    print("Simulation summary")
    print(json.dumps(summary, indent=2))
    print("\nFigures written:")
    for key, path in paths.items():
        print(f"  {key:20s} -> {path}")


if __name__ == "__main__":
    main()
