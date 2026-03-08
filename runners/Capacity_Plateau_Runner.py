#!/usr/bin/env python3
"""
Capacity Plateau Scan Runner

Purpose:
- Collect MERA entropy, fidelity, and energy data across bond dimension chi
- Compare against ED for small systems when available
- Run multiple restarts per chi and select the best result per chi
- Fit log-linear vs saturating models to the best-per-chi entropy curve
- Save data artifacts only
- Add validation metadata so non-credible branches are clearly marked

Default behavior:
- --output is treated as a base directory
- each run creates a unique subdirectory under --output
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Import path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.mera_backend import exact_diagonalization, optimize_mera_for_fidelity  # noqa: E402


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunnerConfig:
    chi_sweep: tuple[int, ...]
    L: int
    A_size: int
    model: str
    fit_steps: int
    restarts_per_chi: int
    seed: int
    output: Path
    overwrite: bool


@dataclass(frozen=True)
class ModelParams:
    is_heisenberg: bool
    j: float
    h: float


@dataclass(frozen=True)
class MeasurementRow:
    chi: int
    restart_idx: int
    seed: int
    entropy: float
    fidelity: float
    energy: float
    converged: bool
    elapsed_sec: float
    ed_entropy: float | None = None
    entropy_error: float | None = None
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Time / run-dir utilities
# ---------------------------------------------------------------------------
def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat(timespec="seconds")


def make_run_id() -> str:
    timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    random_suffix = os.urandom(4).hex()
    return f"RUN_{timestamp}_{random_suffix}"


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"--output is not a directory: {output_dir}")
        if any(output_dir.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                f"Use --overwrite to allow replacing files."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)


def prepare_run_output_dir(base_output: Path, overwrite: bool) -> tuple[Path, str]:
    run_name = make_run_id()

    if overwrite:
        ensure_output_dir(base_output, overwrite=True)
        return base_output, run_name

    base_output.mkdir(parents=True, exist_ok=True)
    run_dir = base_output / run_name
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir, run_name


# ---------------------------------------------------------------------------
# Parsing / config
# ---------------------------------------------------------------------------
def parse_chi_sweep(raw: str) -> tuple[int, ...]:
    try:
        chis = tuple(sorted({int(part.strip()) for part in raw.split(",") if part.strip()}))
    except ValueError as exc:
        raise ValueError(f"Invalid --chi_sweep value: {raw}") from exc

    if not chis:
        raise ValueError("--chi_sweep must contain at least one integer")
    if any(chi <= 0 for chi in chis):
        raise ValueError("--chi_sweep values must all be positive")
    return chis


def parse_args() -> RunnerConfig:
    parser = argparse.ArgumentParser(description="Capacity Plateau Scan (Real MERA)")
    parser.add_argument("--chi_sweep", "--chi-sweep", dest="chi_sweep", default="2,4,8,16,32")
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--A_size", "--A-size", dest="A_size", type=int, default=4)
    parser.add_argument(
        "--model",
        default="ising_cyclic",
        choices=[
            "ising_open",
            "ising_cyclic",
            "heisenberg_open",
            "heisenberg_cyclic",
        ],
    )
    parser.add_argument("--fit_steps", "--fit-steps", dest="fit_steps", type=int, default=80)
    parser.add_argument(
        "--restarts_per_chi",
        "--restarts-per-chi",
        dest="restarts_per_chi",
        type=int,
        default=1,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.L <= 0:
        raise ValueError("--L must be positive")
    if args.A_size <= 0:
        raise ValueError("--A_size must be positive")
    if args.A_size > args.L:
        raise ValueError("--A_size cannot exceed --L")
    if args.fit_steps <= 0:
        raise ValueError("--fit_steps must be positive")
    if args.restarts_per_chi <= 0:
        raise ValueError("--restarts_per_chi must be positive")

    return RunnerConfig(
        chi_sweep=parse_chi_sweep(args.chi_sweep),
        L=args.L,
        A_size=args.A_size,
        model=args.model,
        fit_steps=args.fit_steps,
        restarts_per_chi=args.restarts_per_chi,
        seed=args.seed,
        output=Path(args.output),
        overwrite=bool(args.overwrite),
    )


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, default=str))


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------
def build_model_params(model: str) -> ModelParams:
    is_heisenberg = "heisenberg" in model
    return ModelParams(
        is_heisenberg=is_heisenberg,
        j=1.0,
        h=0.0 if is_heisenberg else 1.0,
    )


def branch_policy_for_model(model: str) -> str:
    return "baseline" if model == "heisenberg_open" else "investigation"


def seed_for_restart(seed_base: int, chi: int, restart_idx: int) -> int:
    return seed_base + restart_idx * 1000 + chi * 10000


def safe_fidelity(x: float) -> float:
    return -1.0 if (x is None or not np.isfinite(x)) else float(x)


def best_result_for_chi(rows: Sequence[MeasurementRow]) -> MeasurementRow:
    if not rows:
        raise ValueError("No rows available for chi")

    def key(r: MeasurementRow) -> Tuple[int, float, float]:
        return (
            1 if r.converged else 0,
            safe_fidelity(r.fidelity),
            float(-r.energy),
        )

    return max(rows, key=key)


def energy_sanity_status(
    rows: Sequence[MeasurementRow],
    ed_energy: float | None,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    if ed_energy is None:
        return {
            "passed": False,
            "reason": "no_ed_reference",
            "tolerance": float(tolerance),
            "invalid_points": [],
        }

    invalid_points: List[Dict[str, Any]] = []

    for row in rows:
        if row.converged and (row.energy < ed_energy - tolerance):
            invalid_points.append(
                {
                    "chi": row.chi,
                    "restart_idx": row.restart_idx,
                    "energy": float(row.energy),
                    "ed_energy": float(ed_energy),
                    "delta": float(row.energy - ed_energy),
                }
            )

    return {
        "passed": len(invalid_points) == 0,
        "reason": None,
        "tolerance": float(tolerance),
        "invalid_points": invalid_points,
    }


# ---------------------------------------------------------------------------
# Model fitting helpers
# ---------------------------------------------------------------------------
def aic_bic_from_rss(rss: float, n: int, k: int, eps: float = 1e-12) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be > 0")
    if k < 0:
        raise ValueError("k must be >= 0")

    rss = max(float(rss), eps)
    aic = n * math.log(rss / n) + 2 * k
    bic = n * math.log(rss / n) + k * math.log(n)
    return float(aic), float(bic)


def fit_loglin(log_chis: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if len(log_chis) != len(y) or len(y) < 2:
        return {"ok": False, "a": None, "b": None, "rss": None, "aic": None, "bic": None}

    x = np.column_stack([log_chis, np.ones_like(log_chis)])
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    b, a = float(coef[0]), float(coef[1])
    yhat = x @ coef
    rss = float(np.sum((y - yhat) ** 2))
    aic, bic = aic_bic_from_rss(rss, n=len(y), k=2)

    return {"ok": True, "a": a, "b": b, "rss": rss, "aic": aic, "bic": bic}


def fit_sat(chis: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if len(chis) != len(y) or len(y) < 3:
        return {
            "ok": False,
            "S_inf": None,
            "c": None,
            "alpha": None,
            "rss": None,
            "aic": None,
            "bic": None,
        }

    best: dict[str, Any] = {
        "ok": False,
        "S_inf": None,
        "c": None,
        "alpha": None,
        "rss": float("inf"),
        "aic": None,
        "bic": None,
    }

    max_y = float(np.max(y))
    s_inf_multipliers = (1.0, 1.01, 1.02, 1.03, 1.05, 1.07, 1.1, 1.15, 1.2, 1.5, 2.0)

    for s_inf_mult in s_inf_multipliers:
        s_inf = max_y * s_inf_mult + 0.05
        delta = s_inf - y
        valid = delta > 1e-12

        if int(np.sum(valid)) < 3:
            continue

        log_d = np.log(delta[valid])
        log_chis = np.log(chis[valid])
        x = np.column_stack([-log_chis, np.ones_like(log_chis)])

        try:
            coef, _, _, _ = np.linalg.lstsq(x, log_d, rcond=None)
            alpha = float(coef[0])
            log_c0 = float(coef[1])
            c = float(np.exp(log_c0))
            yhat = s_inf - c * np.power(chis, -alpha)
            rss = float(np.sum((y - yhat) ** 2))
            aic, bic = aic_bic_from_rss(rss, n=len(y), k=3)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            continue

        if rss < best["rss"]:
            best = {
                "ok": True,
                "S_inf": float(s_inf),
                "c": float(c),
                "alpha": float(alpha),
                "rss": float(rss),
                "aic": float(aic),
                "bic": float(bic),
            }

    if not best["ok"]:
        best["rss"] = None

    return best


def build_fit_results(best_rows: list[MeasurementRow], ed_result: Any | None) -> tuple[dict[str, Any], dict[str, Any]]:
    chis_arr = np.array([row.chi for row in best_rows], dtype=float)
    ents = np.array([row.entropy for row in best_rows], dtype=float)
    log_chis = np.log(chis_arr)

    loglin = fit_loglin(log_chis, ents)
    sat = fit_sat(chis_arr, ents)

    delta_aic: float | None = None
    delta_bic: float | None = None
    if loglin["ok"] and sat["ok"]:
        delta_aic = float(sat["aic"] - loglin["aic"])
        delta_bic = float(sat["bic"] - loglin["bic"])

    monotonic_entropy = all(
        best_rows[i + 1].entropy >= best_rows[i].entropy - 1e-9
        for i in range(len(best_rows) - 1)
    )
    high_fidelity = (
        all(np.isfinite(row.fidelity) and row.fidelity > 0.9 for row in best_rows)
        if ed_result is not None
        else None
    )

    fits = {
        "loglinear": loglin,
        "saturating": sat,
        "delta_aic": delta_aic,
        "delta_bic": delta_bic,
    }

    derived_checks = {
        "sat_preferred": bool(delta_aic is not None and delta_aic < 0),
        "monotonic_entropy": bool(monotonic_entropy),
        "high_fidelity": high_fidelity,
    }

    return fits, derived_checks


# ---------------------------------------------------------------------------
# Physics collection
# ---------------------------------------------------------------------------
def compute_ed_reference(cfg: RunnerConfig, params: ModelParams) -> Any | None:
    if cfg.L > 16:
        return None

    print(f"Computing ED reference for L={cfg.L}...")
    ed_result = exact_diagonalization(
        L=cfg.L,
        model=cfg.model,
        A_size=cfg.A_size,
        j=params.j,
        h=params.h,
    )
    print(
        f"ED: E0={ed_result.ground_state_energy:.6f}, "
        f"S={ed_result.entanglement_entropy:.6f}"
    )
    return ed_result


def collect_measurements(
    cfg: RunnerConfig,
    params: ModelParams,
    ed_result: Any | None,
) -> tuple[list[MeasurementRow], list[MeasurementRow]]:
    print(
        f"Running capacity plateau scan for L={cfg.L}, "
        f"A_size={cfg.A_size}, restarts_per_chi={cfg.restarts_per_chi}"
    )
    print(f"Model: {cfg.model}, chi values: {list(cfg.chi_sweep)}")

    all_rows: list[MeasurementRow] = []
    best_rows: list[MeasurementRow] = []

    ed_entropy = float(ed_result.entanglement_entropy) if ed_result is not None else None
    ed_psi = ed_result.ground_state_psi if ed_result is not None else None

    for chi in cfg.chi_sweep:
        print(f"Optimizing MERA with chi={chi}...")
        chi_rows: list[MeasurementRow] = []

        for restart_idx in range(cfg.restarts_per_chi):
            seed = seed_for_restart(cfg.seed, chi, restart_idx)
            print(f"  restart {restart_idx + 1}/{cfg.restarts_per_chi}, seed={seed}")

            t0 = time.perf_counter()
            opt_result = optimize_mera_for_fidelity(
                L=cfg.L,
                chi=chi,
                ed_psi=ed_psi,
                model=cfg.model,
                steps=cfg.fit_steps,
                seed=seed,
                j=params.j,
                h=params.h,
                A_size=cfg.A_size,
            )
            elapsed = time.perf_counter() - t0

            row = MeasurementRow(
                chi=int(chi),
                restart_idx=int(restart_idx),
                seed=int(seed),
                entropy=float(opt_result.entropy),
                fidelity=float(opt_result.fidelity),
                energy=float(opt_result.final_energy),
                converged=bool(opt_result.converged),
                elapsed_sec=float(elapsed),
                ed_entropy=ed_entropy,
                entropy_error=(
                    abs(float(opt_result.entropy) - ed_entropy)
                    if ed_entropy is not None
                    else None
                ),
                error_message=opt_result.error_message,
            )
            chi_rows.append(row)
            all_rows.append(row)

            fid_str = "nan" if not np.isfinite(row.fidelity) else f"{row.fidelity:.6f}"
            print(
                f"    S={row.entropy:.6f}, fid={fid_str}, "
                f"E={row.energy:.6f}, conv={row.converged}, t={row.elapsed_sec:.2f}s"
            )
            if row.error_message:
                print(f"    warning: {row.error_message}")

        best = best_result_for_chi(chi_rows)
        best_rows.append(best)
        best_fid_str = "nan" if not np.isfinite(best.fidelity) else f"{best.fidelity:.6f}"
        print(
            f"  best chi={chi}: restart={best.restart_idx}, seed={best.seed}, "
            f"S={best.entropy:.6f}, fid={best_fid_str}, E={best.energy:.6f}, "
            f"conv={best.converged}"
        )

    return all_rows, best_rows


# ---------------------------------------------------------------------------
# Payload building / output
# ---------------------------------------------------------------------------
def build_result_payload(cfg: RunnerConfig, resolved_output_dir: Path, run_name: str) -> dict[str, Any]:
    params = build_model_params(cfg.model)
    ed_result = compute_ed_reference(cfg, params)
    all_rows, best_rows = collect_measurements(cfg, params, ed_result)
    fits, derived_checks = build_fit_results(best_rows, ed_result)

    ed_energy = float(ed_result.ground_state_energy) if ed_result is not None else None
    policy = branch_policy_for_model(cfg.model)
    energy_check = energy_sanity_status(best_rows, ed_energy, tolerance=1e-6)
    framework_eligible = bool(policy == "baseline" and energy_check["passed"])

    return {
        "metadata": {
            "run_id": run_name,
            "timestamp_utc": utc_now_iso(),
            "config": {
                "chi_sweep": list(cfg.chi_sweep),
                "L": cfg.L,
                "A_size": cfg.A_size,
                "model": cfg.model,
                "fit_steps": cfg.fit_steps,
                "restarts_per_chi": cfg.restarts_per_chi,
                "seed": cfg.seed,
                "output_base": str(cfg.output),
                "resolved_output_dir": str(resolved_output_dir),
                "overwrite": cfg.overwrite,
            },
            "test": "Capacity Plateau",
            "version": "3.4.1",
        },
        "measurements": [asdict(row) for row in all_rows],
        "best_per_chi": [asdict(row) for row in best_rows],
        "fits": fits,
        "ed_reference": (
            {
                "energy": float(ed_result.ground_state_energy),
                "entropy": float(ed_result.entanglement_entropy),
            }
            if ed_result is not None
            else None
        ),
        "derived_checks": derived_checks,
        "validation": {
            "branch_policy": policy,
            "energy_sanity": energy_check,
            "framework_eligible": framework_eligible,
        },
    }


def write_out(res: dict[str, Any], out_dir: Path) -> None:
    out_dir = Path(out_dir)

    metadata_path = out_dir / "metadata.json"
    raw_path = out_dir / "raw.csv"
    best_path = out_dir / "best_per_chi.csv"
    fits_path = out_dir / "fits.json"
    summary_path = out_dir / "summary.json"

    atomic_write_json(metadata_path, res["metadata"])

    raw_fieldnames = [
        "chi",
        "restart_idx",
        "seed",
        "entropy",
        "fidelity",
        "energy",
        "converged",
        "elapsed_sec",
        "ed_entropy",
        "entropy_error",
        "error_message",
    ]
    atomic_write_csv(raw_path, res["measurements"], raw_fieldnames)
    atomic_write_csv(best_path, res["best_per_chi"], raw_fieldnames)
    atomic_write_json(fits_path, res["fits"])

    summary = {
        "metadata": res["metadata"],
        "ed_reference": res.get("ed_reference"),
        "best_per_chi": res.get("best_per_chi"),
        "fits": res["fits"],
        "derived_checks": res.get("derived_checks"),
        "validation": res.get("validation"),
    }
    atomic_write_json(summary_path, summary)

    print(f"[P2] Results written to {out_dir}")

    delta_aic = res["fits"].get("delta_aic")
    delta_bic = res["fits"].get("delta_bic")

    if delta_aic is not None:
        print(f"[P2] ΔAIC = {delta_aic:.4f}")
    else:
        print("[P2] ΔAIC = unavailable")

    if delta_bic is not None:
        print(f"[P2] ΔBIC = {delta_bic:.4f}")
    else:
        print("[P2] ΔBIC = unavailable")

    validation = res.get("validation", {})
    energy_sanity = validation.get("energy_sanity", {})
    print(f"[P2] Branch policy = {validation.get('branch_policy')}")
    print(f"[P2] Energy sanity passed = {energy_sanity.get('passed')}")
    print(f"[P2] Framework eligible = {validation.get('framework_eligible')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = parse_args()

    print("[P2] Capacity Plateau Scan v3.4.1")
    print(
        f"[P2] L={cfg.L}, A_size={cfg.A_size}, model={cfg.model}, "
        f"restarts_per_chi={cfg.restarts_per_chi}"
    )

    resolved_output_dir, run_name = prepare_run_output_dir(cfg.output, cfg.overwrite)
    print(f"[P2] Run ID: {run_name}")
    print(f"[P2] Output directory: {resolved_output_dir}")

    res = build_result_payload(cfg, resolved_output_dir, run_name)
    write_out(res, resolved_output_dir)


if __name__ == "__main__":
    main()