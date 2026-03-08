#!/usr/bin/env python3
"""
Capacity Plateau Scan Runner (v3.2 - Real MERA, Unique Run Outputs)

Collects MERA entropy, fidelity, and energy data across bond dimension chi,
with exact diagonalization comparison for small systems.

Default behavior:
- --output is treated as a base directory
- each run creates a unique subdirectory under --output

Example:
  python3 Capacity_Plateau_Runner.py --L 8 --A_size 4 \
    --model ising_cyclic --chi_sweep 2,4,8,16 --seed 42 \
    --output results/capacity_plateau

This will create something like:
  results/capacity_plateau/RUN_20260307T181530Z_a1b2c3d4/

Outputs inside that run directory:
  metadata.json
  raw.csv
  fits.json
  summary.json

Optional:
- Use --overwrite to write directly into the exact --output directory instead.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.mera_backend import exact_diagonalization, optimize_mera_for_fidelity  # noqa: E402


@dataclass(frozen=True)
class RunnerConfig:
    chi_sweep: tuple[int, ...]
    L: int
    A_size: int
    model: str
    fit_steps: int
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
    entropy: float
    fidelity: float
    energy: float
    converged: bool
    ed_entropy: float | None = None
    entropy_error: float | None = None
    error_message: str | None = None


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat(timespec="seconds")


def run_id() -> str:
    timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    random_suffix = os.urandom(4).hex()
    return f"RUN_{timestamp}_{random_suffix}"


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


def build_model_params(model: str) -> ModelParams:
    is_heisenberg = "heisenberg" in model
    return ModelParams(
        is_heisenberg=is_heisenberg,
        j=1.0,
        h=0.0 if is_heisenberg else 1.0,
    )


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
        return {
            "ok": False,
            "a": None,
            "b": None,
            "rss": None,
            "aic": None,
            "bic": None,
        }

    x = np.column_stack([log_chis, np.ones_like(log_chis)])
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    b, a = float(coef[0]), float(coef[1])
    yhat = x @ coef
    rss = float(np.sum((y - yhat) ** 2))
    aic, bic = aic_bic_from_rss(rss, n=len(y), k=2)

    return {
        "ok": True,
        "a": a,
        "b": b,
        "rss": rss,
        "aic": aic,
        "bic": bic,
    }


def fit_sat(chis: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """
    Fit saturating model: S(chi) = S_inf - c * chi^(-alpha)
    """
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


def parse_args() -> RunnerConfig:
    parser = argparse.ArgumentParser(description="P2 Capacity Plateau Scan (Real MERA)")
    parser.add_argument("--chi_sweep", "--chi-sweep", default="2,4,8,16,32")
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
    parser.add_argument("--fit_steps", "--fit-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        required=True,
        help="Base output directory. A unique run subdirectory will be created here unless --overwrite is used.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Write directly into the exact --output directory, allowing replacement behavior.",
    )

    args = parser.parse_args()

    if args.L <= 0:
        raise ValueError("--L must be positive")
    if args.A_size <= 0:
        raise ValueError("--A_size must be positive")
    if args.A_size > args.L:
        raise ValueError("--A_size cannot exceed --L")
    if args.fit_steps <= 0:
        raise ValueError("--fit_steps must be positive")

    return RunnerConfig(
        chi_sweep=parse_chi_sweep(args.chi_sweep),
        L=args.L,
        A_size=args.A_size,
        model=args.model,
        fit_steps=args.fit_steps,
        seed=args.seed,
        output=Path(args.output),
        overwrite=bool(args.overwrite),
    )


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
    """
    Default behavior:
      - create unique run directory inside base_output

    Overwrite behavior:
      - write directly into base_output
    """
    run_name = run_id()

    if overwrite:
        ensure_output_dir(base_output, overwrite=True)
        return base_output, run_name

    base_output.mkdir(parents=True, exist_ok=True)
    run_dir = base_output / run_name
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir, run_name


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


def compute_ed_reference(cfg: RunnerConfig, params: ModelParams) -> Any | None:
    if cfg.L > 12:
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


def collect_measurements(cfg: RunnerConfig, params: ModelParams, ed_result: Any | None) -> list[MeasurementRow]:
    print(f"Running capacity plateau scan for L={cfg.L}, A_size={cfg.A_size}")
    print(f"Model: {cfg.model}, chi values: {list(cfg.chi_sweep)}")

    rows: list[MeasurementRow] = []

    ed_entropy = float(ed_result.entanglement_entropy) if ed_result is not None else None
    ed_psi = ed_result.ground_state_psi if ed_result is not None else None

    for chi in cfg.chi_sweep:
        print(f"Optimizing MERA with chi={chi}...")

        opt_result = optimize_mera_for_fidelity(
            L=cfg.L,
            chi=chi,
            ed_psi=ed_psi,
            model=cfg.model,
            steps=cfg.fit_steps,
            seed=cfg.seed,
            j=params.j,
            h=params.h,
            A_size=cfg.A_size,
        )

        entropy = float(opt_result.entropy)
        fidelity = float(opt_result.fidelity)
        energy = float(opt_result.final_energy)
        converged = bool(opt_result.converged)

        row = MeasurementRow(
            chi=int(chi),
            entropy=entropy,
            fidelity=fidelity,
            energy=energy,
            converged=converged,
            ed_entropy=ed_entropy,
            entropy_error=(abs(entropy - ed_entropy) if ed_entropy is not None else None),
            error_message=opt_result.error_message,
        )
        rows.append(row)

        print(
            f"chi={chi}: S={entropy:.4f}, "
            f"fid={fidelity:.6f}, conv={converged}"
        )
        if opt_result.error_message:
            print(f"  warning: {opt_result.error_message}")

    return rows


def build_fit_results(rows: list[MeasurementRow], ed_result: Any | None) -> tuple[dict[str, Any], dict[str, Any]]:
    chis_arr = np.array([row.chi for row in rows], dtype=float)
    ents = np.array([row.entropy for row in rows], dtype=float)
    log_chis = np.log(chis_arr)

    loglin = fit_loglin(log_chis, ents)
    sat = fit_sat(chis_arr, ents)

    delta_aic: float | None = None
    delta_bic: float | None = None
    if loglin["ok"] and sat["ok"]:
        delta_aic = float(sat["aic"] - loglin["aic"])
        delta_bic = float(sat["bic"] - loglin["bic"])

    monotonic_entropy = all(
        rows[i + 1].entropy >= rows[i].entropy - 1e-9
        for i in range(len(rows) - 1)
    )
    high_fidelity = (
        all(row.fidelity > 0.9 for row in rows) if ed_result is not None else None
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


def build_result_payload(cfg: RunnerConfig, resolved_output_dir: Path, run_name: str) -> dict[str, Any]:
    params = build_model_params(cfg.model)
    ed_result = compute_ed_reference(cfg, params)
    rows = collect_measurements(cfg, params, ed_result)
    fits, derived_checks = build_fit_results(rows, ed_result)

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
                "seed": cfg.seed,
                "output_base": str(cfg.output),
                "resolved_output_dir": str(resolved_output_dir),
                "overwrite": cfg.overwrite,
            },
            "test": "Capacity Plateau",
            "version": "3.2.0",
        },
        "measurements": [asdict(row) for row in rows],
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
    }


def write_out(res: dict[str, Any], out_dir: Path) -> None:
    out_dir = Path(out_dir)

    metadata_path = out_dir / "metadata.json"
    raw_path = out_dir / "raw.csv"
    fits_path = out_dir / "fits.json"
    summary_path = out_dir / "summary.json"

    atomic_write_json(metadata_path, res["metadata"])

    fieldnames = [
        "chi",
        "entropy",
        "fidelity",
        "energy",
        "converged",
        "ed_entropy",
        "entropy_error",
        "error_message",
    ]
    atomic_write_csv(raw_path, res["measurements"], fieldnames)

    atomic_write_json(fits_path, res["fits"])

    summary = {
        "metadata": res["metadata"],
        "ed_reference": res.get("ed_reference"),
        "fits": res["fits"],
        "derived_checks": res.get("derived_checks"),
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


def main() -> None:
    cfg = parse_args()

    print("[P2] Capacity Plateau Scan v3.2")
    print(f"[P2] L={cfg.L}, A_size={cfg.A_size}, model={cfg.model}")

    resolved_output_dir, run_name = prepare_run_output_dir(cfg.output, cfg.overwrite)
    print(f"[P2] Run ID: {run_name}")
    print(f"[P2] Output directory: {resolved_output_dir}")

    res = build_result_payload(cfg, resolved_output_dir, run_name)
    write_out(res, resolved_output_dir)


if __name__ == "__main__":
    main()