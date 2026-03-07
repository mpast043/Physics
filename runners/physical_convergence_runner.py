#!/usr/bin/env python3
"""
Physical Convergence Runner v3 (data collection only)

Purpose:
- Compute ED reference data for Ising / Heisenberg models
- Run MERA optimization across chi values and restarts
- Collect raw per-restart data with strong reproducibility safeguards
- Save data artifacts only (no falsifiers, verdicts, or acceptance criteria)

Usage:
  python3 physical_convergence_runner_updated.py --L 8 --A_size 4 \
    --model ising_open --j 1.0 --h 1.0 --chi_sweep 2,4,8,16 \
    --restarts_per_chi 3 --fit_steps 100 --seed 42 --output <RUN_DIR>
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import socket
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.mera_backend import exact_diagonalization, optimize_mera_for_fidelity  # noqa: E402


# ============================================================
# Types and Configuration
# ============================================================


@dataclass(frozen=True)
class Config:
    L: int
    A_size: int
    model: str
    chi_sweep: List[int]
    restarts_per_chi: int
    fit_steps: int
    seed: int
    output_dir: Path
    j: float = 1.0
    h: float = 1.0

    def validate(self) -> None:
        if self.L <= 1:
            raise ValueError("--L must be > 1")
        if self.A_size <= 0 or self.A_size >= self.L:
            raise ValueError("--A_size must satisfy 1 <= A_size < L")
        if not self.chi_sweep:
            raise ValueError("--chi_sweep must contain at least one chi value")
        if any(chi <= 0 for chi in self.chi_sweep):
            raise ValueError("All chi values must be positive")
        if self.restarts_per_chi <= 0:
            raise ValueError("--restarts_per_chi must be > 0")
        if self.fit_steps <= 0:
            raise ValueError("--fit_steps must be > 0")
        if self.model not in {
            "ising_open",
            "ising_cyclic",
            "heisenberg_open",
            "heisenberg_cyclic",
        }:
            raise ValueError(f"Unsupported model: {self.model}")


@dataclass
class OptimizationResult:
    chi: int
    restart_idx: int
    fidelity: float
    entropy: float
    final_energy: float
    seed: int
    converged: bool
    num_steps: int
    elapsed_sec: float
    error_message: Optional[str] = None


# ============================================================
# Utilities
# ============================================================


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_chi_sweep(text: str) -> List[int]:
    values = []
    for raw in text.split(","):
        raw = raw.strip()
        if raw:
            values.append(int(raw))
    return sorted(set(values))


def make_run_id() -> str:
    return f"RUN_{utc_now().strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}_{os.urandom(4).hex()}"


def create_unique_run_dir(base_output: Path) -> Tuple[str, Path]:
    base_output.mkdir(parents=True, exist_ok=True)
    for _ in range(16):
        run_id = make_run_id()
        run_dir = base_output / run_id
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
            return run_id, run_dir
        except FileExistsError:
            continue
    raise RuntimeError("Could not create a unique run directory after multiple attempts")


def json_default(obj: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None

    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dt.datetime):
        return obj.isoformat()
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if np is not None and isinstance(obj, (np.integer,)):
        return int(obj)
    if np is not None and isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=json_default)
    os.replace(tmp_path, path)


def write_csv_atomic(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(tmp_path, path)


def snapshot_runtime_environment() -> Dict[str, Any]:
    return {
        "timestamp_utc": utc_now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "argv": sys.argv,
        "pid": os.getpid(),
    }


def best_result_for_chi(results: Sequence[OptimizationResult]) -> OptimizationResult:
    if not results:
        raise ValueError("No results available for chi")

    def key(r: OptimizationResult) -> Tuple[int, float, float]:
        return (
            1 if r.converged else 0,
            float(r.fidelity),
            float(-r.final_energy),
        )

    return max(results, key=key)


# ============================================================
# MERA sweep using shared backend
# ============================================================


def run_mera_with_restarts(
    L: int,
    A_size: int,
    chi: int,
    ed_psi,
    model: str,
    steps: int,
    restarts: int,
    seed_base: int,
    j: float = 1.0,
    h: float = 1.0,
) -> List[OptimizationResult]:
    results: List[OptimizationResult] = []

    for restart in range(restarts):
        seed = seed_base + restart * 1000 + chi * 10000
        print(f"    [MERA] chi={chi}, restart={restart + 1}/{restarts}, seed={seed}")

        t0 = time.perf_counter()
        opt_result = optimize_mera_for_fidelity(
            L=L,
            chi=chi,
            ed_psi=ed_psi,
            model=model,
            steps=steps,
            seed=seed,
            j=j,
            h=h,
            A_size=A_size,
        )
        elapsed = time.perf_counter() - t0

        if opt_result.converged:
            print(
                f"      converged=True, fidelity={opt_result.fidelity:.8f}, "
                f"S={opt_result.entropy:.8f}, E={opt_result.final_energy:.8f}, t={elapsed:.2f}s"
            )
        else:
            print(
                f"      converged=False, error={opt_result.error_message}, t={elapsed:.2f}s"
            )

        results.append(
            OptimizationResult(
                chi=chi,
                restart_idx=restart,
                fidelity=float(opt_result.fidelity),
                entropy=float(opt_result.entropy),
                final_energy=float(opt_result.final_energy),
                seed=seed,
                converged=bool(opt_result.converged),
                num_steps=steps,
                elapsed_sec=float(elapsed),
                error_message=opt_result.error_message,
            )
        )

    return results


# ============================================================
# Main Runner
# ============================================================


def build_config_from_args(args: argparse.Namespace) -> Config:
    config = Config(
        L=args.L,
        A_size=args.A_size,
        model=args.model,
        chi_sweep=parse_chi_sweep(args.chi_sweep),
        restarts_per_chi=args.restarts_per_chi,
        fit_steps=args.fit_steps,
        seed=args.seed,
        output_dir=Path(args.output),
        j=args.j,
        h=args.h,
    )
    config.validate()
    return config


def main() -> int:
    ap = argparse.ArgumentParser(description="Physical convergence data runner")
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--A_size", "--A-size", type=int, required=True)
    ap.add_argument(
        "--model",
        choices=["ising_open", "heisenberg_open", "ising_cyclic", "heisenberg_cyclic"],
        required=True,
    )
    ap.add_argument("--j", type=float, default=1.0)
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--chi_sweep", "--chi-sweep", type=str, default="8,16,32,64")
    ap.add_argument("--restarts_per_chi", "--restarts-per-chi", type=int, default=2)
    ap.add_argument("--fit_steps", "--fit-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, required=True)

    args = ap.parse_args()
    config = build_config_from_args(args)

    run_id, run_dir = create_unique_run_dir(config.output_dir)
    env = snapshot_runtime_environment()

    print("=" * 72)
    print("PHYSICAL CONVERGENCE DATA RUNNER")
    print("=" * 72)
    print(f"Model:       {config.model}")
    print(f"System:      L={config.L}, A_size={config.A_size}")
    print(f"Chi sweep:   {config.chi_sweep}")
    print(f"Restarts:    {config.restarts_per_chi}")
    print(f"Fit steps:   {config.fit_steps}")
    print(f"Seed:        {config.seed}")
    print(f"Output dir:  {run_dir}")
    print("=" * 72)

    total_t0 = time.perf_counter()

    print("\n[Phase 1] Exact Diagonalization")
    ed_t0 = time.perf_counter()
    ed_result = exact_diagonalization(
        L=config.L,
        model=config.model,
        A_size=config.A_size,
        j=config.j,
        h=config.h,
    )
    ed_elapsed = time.perf_counter() - ed_t0

    print("\n[Phase 2] MERA Variational Optimization")

    all_results: List[OptimizationResult] = []
    best_results: List[OptimizationResult] = []

    for chi in config.chi_sweep:
        print(f"\n  Chi = {chi}")
        chi_results = run_mera_with_restarts(
            L=config.L,
            A_size=config.A_size,
            chi=chi,
            ed_psi=ed_result.ground_state_psi,
            model=config.model,
            steps=config.fit_steps,
            restarts=config.restarts_per_chi,
            seed_base=config.seed,
            j=config.j,
            h=config.h,
        )

        best = best_result_for_chi(chi_results)
        best_results.append(best)
        all_results.extend(chi_results)

        print(
            f"  Best: converged={best.converged}, "
            f"fidelity={best.fidelity:.8f}, S={best.entropy:.8f}, E={best.final_energy:.8f}"
        )

    total_elapsed = time.perf_counter() - total_t0

    print(f"\n{'=' * 72}")
    print("RUN COMPLETE")
    print(f"TOTAL ELAPSED: {total_elapsed:.2f}s")
    print(f"{'=' * 72}")

    print(f"\n[Phase 3] Saving artifacts to {run_dir}")

    config_payload = {
        "run_id": run_id,
        "config": {
            "L": config.L,
            "A_size": config.A_size,
            "model": config.model,
            "chi_sweep": config.chi_sweep,
            "restarts_per_chi": config.restarts_per_chi,
            "fit_steps": config.fit_steps,
            "seed": config.seed,
            "output_dir": str(config.output_dir),
            "j": config.j,
            "h": config.h,
        },
    }

    ed_payload = {
        "run_id": run_id,
        "energy": float(ed_result.ground_state_energy),
        "entropy": float(ed_result.entanglement_entropy),
        "entanglement_spectrum": None
        if ed_result.entanglement_spectrum is None
        else ed_result.entanglement_spectrum.tolist(),
        "entanglement_gap": ed_result.entanglement_gap,
        "n_sites": ed_result.n_sites,
        "elapsed_sec": float(ed_elapsed),
    }

    summary_payload = {
        "run_id": run_id,
        "model": config.model,
        "L": config.L,
        "A_size": config.A_size,
        "chi_sweep": config.chi_sweep,
        "ed_energy": float(ed_result.ground_state_energy),
        "ed_entropy": float(ed_result.entanglement_entropy),
        "num_total_restarts": len(all_results),
        "num_failed_restarts": sum(0 if r.converged else 1 for r in all_results),
        "num_successful_restarts": sum(1 if r.converged else 0 for r in all_results),
        "best_per_chi": [
            {
                "chi": r.chi,
                "restart_idx": r.restart_idx,
                "seed": r.seed,
                "converged": r.converged,
                "fidelity": r.fidelity,
                "entropy": r.entropy,
                "final_energy": r.final_energy,
                "elapsed_sec": r.elapsed_sec,
                "error_message": r.error_message,
            }
            for r in best_results
        ],
        "timing": {
            "ed_elapsed_sec": float(ed_elapsed),
            "total_elapsed_sec": float(total_elapsed),
        },
    }

    failures_payload = {
        "run_id": run_id,
        "failures": [
            {
                "chi": r.chi,
                "restart_idx": r.restart_idx,
                "seed": r.seed,
                "num_steps": r.num_steps,
                "elapsed_sec": r.elapsed_sec,
                "error_message": r.error_message,
            }
            for r in all_results
            if not r.converged
        ],
    }

    manifest_payload = {
        "run_id": run_id,
        "environment": env,
        "artifacts": [
            "config.json",
            "ed_reference.json",
            "summary.json",
            "manifest.json",
            "raw_results.csv",
            "best_per_chi.csv",
            "failures.json",
        ],
    }

    write_json_atomic(run_dir / "config.json", config_payload)
    write_json_atomic(run_dir / "ed_reference.json", ed_payload)
    write_json_atomic(run_dir / "summary.json", summary_payload)
    write_json_atomic(run_dir / "failures.json", failures_payload)
    write_json_atomic(run_dir / "manifest.json", manifest_payload)

    raw_rows = [
        [
            r.chi,
            r.restart_idx,
            r.seed,
            int(r.converged),
            r.num_steps,
            r.elapsed_sec,
            r.fidelity,
            r.entropy,
            r.final_energy,
            r.error_message or "",
        ]
        for r in all_results
    ]
    write_csv_atomic(
        run_dir / "raw_results.csv",
        header=[
            "chi",
            "restart_idx",
            "seed",
            "converged",
            "num_steps",
            "elapsed_sec",
            "fidelity",
            "entropy",
            "final_energy",
            "error_message",
        ],
        rows=raw_rows,
    )

    best_rows = [
        [
            r.chi,
            r.restart_idx,
            r.seed,
            int(r.converged),
            r.elapsed_sec,
            r.fidelity,
            r.entropy,
            r.final_energy,
            r.error_message or "",
        ]
        for r in best_results
    ]
    write_csv_atomic(
        run_dir / "best_per_chi.csv",
        header=[
            "chi",
            "best_restart_idx",
            "seed",
            "converged",
            "elapsed_sec",
            "fidelity",
            "entropy",
            "final_energy",
            "error_message",
        ],
        rows=best_rows,
    )

    print(
        "  Saved: config.json, ed_reference.json, summary.json, "
        "manifest.json, raw_results.csv, best_per_chi.csv, failures.json"
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)
