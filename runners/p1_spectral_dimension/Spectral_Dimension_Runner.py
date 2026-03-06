#!/usr/bin/env python3
"""P1 spectral dimension data runner."""

from __future__ import annotations

import argparse
import csv
import json
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

VERSION = "2.1.0"
DEFAULT_STEPS = [10, 20, 40, 80, 160, 320, 640, 1280]
FIT_START_INDEX = 1
MIN_RETURN_PROB = 1e-12


@dataclass(frozen=True)
class RunConfig:
    seed: int
    true_ds: float
    noise_level: float
    steps: list[int]
    label: str | None = None


@dataclass(frozen=True)
class FitSummary:
    d_s_estimated: float
    slope: float
    intercept: float
    r_squared: float
    n_points: int
    fit_start_index: int


def utc_now_iso() -> str:
    """Return an ISO 8601 UTC timestamp with a trailing Z."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def make_run_id() -> str:
    """Build a collision-resistant run identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{secrets.token_hex(4)}"


def parse_steps(raw: str) -> list[int]:
    """Parse a comma-separated list of positive, strictly increasing integers."""
    try:
        steps = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--steps must contain only integers.") from exc

    if len(steps) < 2:
        raise argparse.ArgumentTypeError("--steps must contain at least two values.")
    if any(step <= 0 for step in steps):
        raise argparse.ArgumentTypeError("--steps values must be positive.")
    if len(set(steps)) != len(steps):
        raise argparse.ArgumentTypeError("--steps values must be unique.")
    if any(curr <= prev for prev, curr in zip(steps, steps[1:])):
        raise argparse.ArgumentTypeError("--steps values must be strictly increasing.")

    return steps


def estimate_spectral_dim(steps: np.ndarray, return_probs: np.ndarray) -> FitSummary:
    """Estimate spectral dimension from return-probability scaling."""
    if steps.shape != return_probs.shape:
        raise ValueError("steps and return_probs must have the same shape.")
    if steps.size <= FIT_START_INDEX:
        raise ValueError("Not enough measurements for the requested fit window.")
    if np.any(steps <= 0):
        raise ValueError("steps must be strictly positive.")
    if np.any(return_probs <= 0):
        raise ValueError("return_probs must be strictly positive.")

    fit_steps = steps[FIT_START_INDEX:].astype(np.float64)
    fit_probs = return_probs[FIT_START_INDEX:].astype(np.float64)

    log_steps = np.log(fit_steps)
    log_probs = np.log(fit_probs)

    design = np.column_stack((log_steps, np.ones_like(log_steps)))
    coef, _, _, _ = np.linalg.lstsq(design, log_probs, rcond=None)
    slope, intercept = map(float, coef)

    predicted = slope * log_steps + intercept
    residual_sum_squares = float(np.sum((log_probs - predicted) ** 2))
    total_sum_squares = float(np.sum((log_probs - np.mean(log_probs)) ** 2))
    r_squared = 1.0 if total_sum_squares == 0.0 else 1.0 - residual_sum_squares / total_sum_squares

    return FitSummary(
        d_s_estimated=-2.0 * slope,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        n_points=int(fit_steps.size),
        fit_start_index=FIT_START_INDEX,
    )


def simulate_return_probabilities(cfg: RunConfig) -> np.ndarray:
    """Generate noisy return probabilities for the configured step counts."""
    rng = np.random.default_rng(cfg.seed)
    steps = np.asarray(cfg.steps, dtype=np.float64)
    baseline = np.power(steps, -cfg.true_ds / 2.0)
    multiplicative_noise = 1.0 + cfg.noise_level * rng.standard_normal(steps.size)
    noisy = baseline * multiplicative_noise
    return np.maximum(MIN_RETURN_PROB, noisy)


def build_measurements(steps: np.ndarray, return_probs: np.ndarray) -> list[dict[str, float | int]]:
    """Build row-wise measurement records."""
    log_steps = np.log(steps.astype(np.float64))
    log_return_probs = np.log(return_probs.astype(np.float64))

    records: list[dict[str, float | int]] = []
    for step, prob, log_step, log_prob in zip(
        steps.tolist(),
        return_probs.tolist(),
        log_steps.tolist(),
        log_return_probs.tolist(),
    ):
        records.append(
            {
                "step": int(step),
                "return_prob": float(prob),
                "log_step": float(log_step),
                "log_return_prob": float(log_prob),
            }
        )
    return records


def run_p1(cfg: RunConfig) -> dict[str, object]:
    """Run the P1 spectral-dimension simulation and collect outputs."""
    steps = np.asarray(cfg.steps, dtype=np.int64)
    return_probs = simulate_return_probabilities(cfg)
    measurements = build_measurements(steps, return_probs)
    summary = estimate_spectral_dim(steps, return_probs)
    run_id = make_run_id()

    return {
        "metadata": {
            "run_id": run_id,
            "timestamp_utc": utc_now_iso(),
            "test": "P1",
            "runner": "spectral_dimension",
            "version": VERSION,
        },
        "config": asdict(cfg),
        "summary": asdict(summary),
        "measurements": measurements,
    }


def prepare_run_directory(output_root: Path, run_id: str) -> Path:
    """Create a unique per-run directory so prior results are never overwritten."""
    output_root.mkdir(parents=True, exist_ok=True)

    candidate = output_root / f"run_{run_id}"
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"run_{run_id}_{suffix:02d}"
        suffix += 1

    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


def write_json(path: Path, payload: object) -> None:
    """Write JSON with consistent formatting."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_measurements_csv(path: Path, measurements: list[dict[str, object]]) -> None:
    """Write tabular measurements to CSV for quick inspection."""
    if not measurements:
        raise ValueError("measurements cannot be empty.")

    fieldnames = ["step", "return_prob", "log_step", "log_return_prob"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(measurements)


def write_outputs(result: dict[str, object], output_root: str | Path) -> Path:
    """Persist all outputs to a unique run directory."""
    metadata = result["metadata"]
    if not isinstance(metadata, dict) or "run_id" not in metadata:
        raise ValueError("result must contain metadata.run_id.")

    run_dir = prepare_run_directory(Path(output_root), str(metadata["run_id"]))

    write_json(run_dir / "metadata.json", result["metadata"])
    write_json(run_dir / "config.json", result["config"])
    write_json(run_dir / "summary.json", result["summary"])
    write_json(run_dir / "measurements.json", result["measurements"])
    write_measurements_csv(run_dir / "measurements.csv", result["measurements"])
    write_json(run_dir / "run.json", result)

    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description="P1 spectral dimension data runner")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        required=True,
        help="Root output directory. A unique run subdirectory is created automatically.",
    )
    parser.add_argument(
        "--true-ds",
        type=float,
        default=1.35,
        help="Reference spectral dimension used to generate synthetic data.",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.05,
        help="Multiplicative Gaussian noise scale applied to return probabilities.",
    )
    parser.add_argument(
        "--steps",
        type=parse_steps,
        default=DEFAULT_STEPS,
        help="Comma-separated positive, strictly increasing step counts.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional free-text label stored with the run configuration.",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    cfg = RunConfig(
        seed=args.seed,
        true_ds=args.true_ds,
        noise_level=args.noise_level,
        steps=list(args.steps),
        label=args.label,
    )

    result = run_p1(cfg)
    run_dir = write_outputs(result, args.output)

    summary = result["summary"]
    print(f"[P1] Run directory: {run_dir}")
    print(
        "[P1] "
        f"d_s_estimated={summary['d_s_estimated']:.6f} "
        f"slope={summary['slope']:.6f} "
        f"r_squared={summary['r_squared']:.6f}"
    )


if __name__ == "__main__":
    main()
