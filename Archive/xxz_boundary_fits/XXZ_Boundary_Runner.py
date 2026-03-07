#!/usr/bin/env python3
"""XXZ boundary data-first runner.

This runner does not encode literature verdicts. It ingests measured
entanglement-entropy data, fits the appropriate CFT-inspired log form for
open or periodic boundary conditions, and writes standardized run outputs.

Expected input fields per row:
    delta,boundary,L,ell,entropy

Boundary aliases accepted:
    open, obc, periodic, pbc
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import secrets
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

VERSION = "1.0.0"
TEST_NAME = "XXZ_BOUNDARY_DATA"
MIN_FIT_POINTS = 3


@dataclass(frozen=True)
class RunConfig:
    input_path: str
    output_root: str
    exclude_endpoints: int
    parity_filter: str
    estimate_ceff: bool
    label: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def make_run_id(prefix: str = "xxz_data") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}_{secrets.token_hex(4)}"


def normalize_boundary(raw: str) -> str:
    value = str(raw).strip().lower()
    mapping = {
        "open": "open",
        "obc": "open",
        "periodic": "periodic",
        "pbc": "periodic",
    }
    if value not in mapping:
        raise ValueError(f"Unsupported boundary '{raw}'. Use open/obc or periodic/pbc.")
    return mapping[value]


def parse_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"delta", "boundary", "L", "ell", "entropy"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"CSV must include columns: {sorted(required)}. "
                f"Found: {reader.fieldnames}"
            )

        for idx, row in enumerate(reader, start=2):
            try:
                rows.append(
                    {
                        "delta": float(row["delta"]),
                        "boundary": normalize_boundary(row["boundary"]),
                        "L": int(row["L"]),
                        "ell": int(row["ell"]),
                        "entropy": float(row["entropy"]),
                        "source_row": idx,
                    }
                )
            except Exception as exc:
                raise ValueError(f"Invalid data in CSV row {idx}: {row}") from exc
    return rows


def parse_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        if "measurements" not in payload or not isinstance(payload["measurements"], list):
            raise ValueError("JSON object input must contain a 'measurements' list.")
        raw_rows = payload["measurements"]
    elif isinstance(payload, list):
        raw_rows = payload
    else:
        raise ValueError("JSON input must be either a list or an object with 'measurements'.")

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_rows, start=1):
        try:
            rows.append(
                {
                    "delta": float(row["delta"]),
                    "boundary": normalize_boundary(row["boundary"]),
                    "L": int(row["L"]),
                    "ell": int(row["ell"]),
                    "entropy": float(row["entropy"]),
                    "source_row": idx,
                }
            )
        except Exception as exc:
            raise ValueError(f"Invalid data in JSON measurement #{idx}: {row}") from exc
    return rows


def load_measurements(path_str: str) -> list[dict[str, Any]]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = parse_csv(path)
    elif suffix == ".json":
        rows = parse_json(path)
    else:
        raise ValueError("Input file must be .csv or .json")

    if not rows:
        raise ValueError("No measurements found in input file.")

    validate_measurements(rows)
    return rows


def validate_measurements(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        L = row["L"]
        ell = row["ell"]
        entropy = row["entropy"]

        if L <= 1:
            raise ValueError(f"Invalid system size L={L}. Must be > 1.")
        if not (1 <= ell <= L - 1):
            raise ValueError(f"Invalid bipartition ell={ell} for L={L}. Must satisfy 1 <= ell <= L-1.")
        if not np.isfinite(entropy):
            raise ValueError(f"Invalid entropy value: {entropy}")


def compute_log_form_x(boundary: str, L: int, ell: int) -> tuple[float, float]:
    """Return (chord_length, x=log(chord_length)) for the appropriate boundary form."""
    sine_term = math.sin(math.pi * ell / L)
    if sine_term <= 0.0:
        raise ValueError(f"Non-positive sine term for L={L}, ell={ell}.")

    if boundary == "periodic":
        chord_length = (L / math.pi) * sine_term
    elif boundary == "open":
        chord_length = (2.0 * L / math.pi) * sine_term
    else:
        raise ValueError(f"Unsupported boundary: {boundary}")

    if chord_length <= 0.0:
        raise ValueError(f"Non-positive chord length for L={L}, ell={ell}, boundary={boundary}")

    return chord_length, math.log(chord_length)


def build_group_key(row: dict[str, Any]) -> tuple[float, str, int]:
    return (float(row["delta"]), str(row["boundary"]), int(row["L"]))


def parity_matches(ell: int, parity_filter: str) -> bool:
    if parity_filter == "all":
        return True
    if parity_filter == "even":
        return ell % 2 == 0
    if parity_filter == "odd":
        return ell % 2 == 1
    raise ValueError(f"Unsupported parity_filter: {parity_filter}")


def annotate_rows(
    rows: list[dict[str, Any]],
    exclude_endpoints: int,
    parity_filter: str,
) -> list[dict[str, Any]]:
    """Add fit-related fields to each measurement row without discarding raw data."""
    annotated: list[dict[str, Any]] = []

    for row in rows:
        delta = float(row["delta"])
        boundary = str(row["boundary"])
        L = int(row["L"])
        ell = int(row["ell"])
        entropy = float(row["entropy"])

        chord_length, x_value = compute_log_form_x(boundary, L, ell)

        include = True
        exclusion_reason = ""

        if ell <= exclude_endpoints or ell >= (L - exclude_endpoints):
            include = False
            exclusion_reason = f"endpoint_window_{exclude_endpoints}"

        if include and not parity_matches(ell, parity_filter):
            include = False
            exclusion_reason = f"parity_filter_{parity_filter}"

        annotated.append(
            {
                "delta": delta,
                "boundary": boundary,
                "L": L,
                "ell": ell,
                "entropy": entropy,
                "chord_length": float(chord_length),
                "log_form_x": float(x_value),
                "included_in_fit": include,
                "exclusion_reason": exclusion_reason or None,
                "source_row": row["source_row"],
            }
        )

    return annotated


def fit_linear_model(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if x.size < MIN_FIT_POINTS:
        raise ValueError(f"Need at least {MIN_FIT_POINTS} points to fit; got {x.size}.")

    X = np.column_stack((x, np.ones_like(x)))
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    slope = float(coef[0])
    intercept = float(coef[1])

    y_hat = slope * x + intercept
    residuals = y - y_hat

    rss = float(np.sum(residuals ** 2))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 if tss == 0.0 else float(1.0 - rss / tss)

    return {
        "slope": slope,
        "intercept": intercept,
        "rss": rss,
        "rmse": rmse,
        "r_squared": r_squared,
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals, ddof=0)),
        "y_hat": y_hat.tolist(),
        "residuals": residuals.tolist(),
    }


def estimate_ceff_from_slope(boundary: str, slope: float) -> float:
    """
    Effective central charge estimate from fitted slope.

    Periodic: S ~ (c/3) log(chord) + const  => c_eff = 3*slope
    Open:     S ~ (c/6) log(chord) + const  => c_eff = 6*slope
    """
    if boundary == "periodic":
        return 3.0 * slope
    if boundary == "open":
        return 6.0 * slope
    raise ValueError(f"Unsupported boundary: {boundary}")


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[float, str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[float, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[build_group_key(row)].append(row)

    for key in grouped:
        grouped[key].sort(key=lambda r: (r["ell"], r["source_row"]))
    return dict(grouped)


def analyze_groups(
    grouped_rows: dict[tuple[float, str, int], list[dict[str, Any]]],
    estimate_ceff: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fit_summaries: list[dict[str, Any]] = []
    fit_points: list[dict[str, Any]] = []

    for (delta, boundary, L), rows in grouped_rows.items():
        used_rows = [row for row in rows if row["included_in_fit"]]
        skipped_rows = [row for row in rows if not row["included_in_fit"]]

        summary: dict[str, Any] = {
            "delta": delta,
            "boundary": boundary,
            "L": L,
            "n_total": len(rows),
            "n_used": len(used_rows),
            "n_skipped": len(skipped_rows),
            "fit_success": False,
            "fit_error": None,
        }

        if len(used_rows) < MIN_FIT_POINTS:
            summary["fit_error"] = f"Need at least {MIN_FIT_POINTS} included points; got {len(used_rows)}."
            fit_summaries.append(summary)

            for row in rows:
                fit_points.append(
                    {
                        **row,
                        "fitted_entropy": None,
                        "residual": None,
                    }
                )
            continue

        x = np.asarray([row["log_form_x"] for row in used_rows], dtype=np.float64)
        y = np.asarray([row["entropy"] for row in used_rows], dtype=np.float64)

        fit = fit_linear_model(x, y)

        summary.update(
            {
                "fit_success": True,
                "slope": fit["slope"],
                "intercept": fit["intercept"],
                "rss": fit["rss"],
                "rmse": fit["rmse"],
                "r_squared": fit["r_squared"],
                "residual_mean": fit["residual_mean"],
                "residual_std": fit["residual_std"],
            }
        )

        if estimate_ceff:
            summary["c_eff"] = estimate_ceff_from_slope(boundary, fit["slope"])

        y_hat_lookup = {
            (row["ell"], row["source_row"]): (fitted, resid)
            for row, fitted, resid in zip(used_rows, fit["y_hat"], fit["residuals"])
        }

        for row in rows:
            key = (row["ell"], row["source_row"])
            fitted_entropy, residual = y_hat_lookup.get(key, (None, None))
            fit_points.append(
                {
                    **row,
                    "fitted_entropy": fitted_entropy,
                    "residual": residual,
                }
            )

        fit_summaries.append(summary)

    fit_summaries.sort(key=lambda r: (r["delta"], r["boundary"], r["L"]))
    fit_points.sort(key=lambda r: (r["delta"], r["boundary"], r["L"], r["ell"], r["source_row"]))
    return fit_summaries, fit_points


def summarize_run(
    raw_rows: list[dict[str, Any]],
    fit_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    boundaries = sorted({row["boundary"] for row in raw_rows})
    deltas = sorted({float(row["delta"]) for row in raw_rows})
    system_sizes = sorted({int(row["L"]) for row in raw_rows})

    successful = [row for row in fit_summaries if row["fit_success"]]
    failed = [row for row in fit_summaries if not row["fit_success"]]

    summary: dict[str, Any] = {
        "measurement_count": len(raw_rows),
        "group_count": len(fit_summaries),
        "fit_success_count": len(successful),
        "fit_failure_count": len(failed),
        "boundaries": boundaries,
        "delta_values": deltas,
        "system_sizes": system_sizes,
    }

    if successful:
        summary["r_squared_min"] = min(row["r_squared"] for row in successful)
        summary["r_squared_max"] = max(row["r_squared"] for row in successful)
        summary["rmse_min"] = min(row["rmse"] for row in successful)
        summary["rmse_max"] = max(row["rmse"] for row in successful)

    return summary


def prepare_run_directory(output_root: Path, run_id: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)

    candidate = output_root / f"run_{run_id}"
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"run_{run_id}_{suffix:02d}"
        suffix += 1

    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XXZ boundary data-first runner")
    parser.add_argument("--input", required=True, help="Path to .csv or .json entropy measurements.")
    parser.add_argument("--output", required=True, help="Root output directory.")
    parser.add_argument(
        "--exclude-endpoints",
        type=int,
        default=0,
        help="Exclude ell <= n and ell >= L-n from the fit. Default: 0",
    )
    parser.add_argument(
        "--parity-filter",
        choices=["all", "even", "odd"],
        default="all",
        help="Optional parity filter for ell to inspect OBC oscillation sensitivity.",
    )
    parser.add_argument(
        "--estimate-ceff",
        action="store_true",
        help="Include c_eff estimate derived from fitted slope.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional free-text label stored in config.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.exclude_endpoints < 0:
        raise ValueError("--exclude-endpoints must be >= 0")

    config = RunConfig(
        input_path=args.input,
        output_root=args.output,
        exclude_endpoints=args.exclude_endpoints,
        parity_filter=args.parity_filter,
        estimate_ceff=args.estimate_ceff,
        label=args.label,
    )

    raw_rows = load_measurements(config.input_path)
    annotated_rows = annotate_rows(
        rows=raw_rows,
        exclude_endpoints=config.exclude_endpoints,
        parity_filter=config.parity_filter,
    )
    grouped = group_rows(annotated_rows)
    fit_summaries, fit_points = analyze_groups(grouped, estimate_ceff=config.estimate_ceff)
    run_summary = summarize_run(raw_rows, fit_summaries)

    run_id = make_run_id()
    timestamp_utc = utc_now_iso()

    result = {
        "metadata": {
            "run_id": run_id,
            "timestamp_utc": timestamp_utc,
            "test": TEST_NAME,
            "version": VERSION,
            "runner": "xxz_boundary_data",
        },
        "config": asdict(config),
        "summary": run_summary,
        "fits": fit_summaries,
        "fit_points": fit_points,
    }

    run_dir = prepare_run_directory(Path(config.output_root), run_id)

    write_json(run_dir / "metadata.json", result["metadata"])
    write_json(run_dir / "config.json", result["config"])
    write_json(run_dir / "summary.json", result["summary"])
    write_json(run_dir / "fits.json", result["fits"])
    write_json(run_dir / "fit_points.json", result["fit_points"])
    write_json(run_dir / "run.json", result)

    write_csv(
        run_dir / "fits.csv",
        fit_summaries,
        [
            "delta",
            "boundary",
            "L",
            "n_total",
            "n_used",
            "n_skipped",
            "fit_success",
            "fit_error",
            "slope",
            "intercept",
            "rss",
            "rmse",
            "r_squared",
            "residual_mean",
            "residual_std",
            "c_eff",
        ],
    )
    write_csv(
        run_dir / "fit_points.csv",
        fit_points,
        [
            "delta",
            "boundary",
            "L",
            "ell",
            "entropy",
            "chord_length",
            "log_form_x",
            "included_in_fit",
            "exclusion_reason",
            "fitted_entropy",
            "residual",
            "source_row",
        ],
    )

    print(f"[XXZ] Run directory: {run_dir}")
    print(
        f"[XXZ] measurements={run_summary['measurement_count']} "
        f"groups={run_summary['group_count']} "
        f"fit_success={run_summary['fit_success_count']} "
        f"fit_failure={run_summary['fit_failure_count']}"
    )


if __name__ == "__main__":
    main()
