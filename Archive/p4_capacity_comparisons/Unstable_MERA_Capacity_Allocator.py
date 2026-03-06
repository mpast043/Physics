#!/usr/bin/env python3
"""
MERA Capacity Allocator

Compares a simple structured 1D MERA-like capacity model against a random
tensor-network baseline.

Design goals
------------
1. Preserve raw trial data.
2. Never overwrite prior outputs.
3. Remain reproducible.
4. Stream summaries without retaining all trials in memory.
5. Stay simple and auditable.

Notes
-----
This is a proxy capacity-accounting experiment, not a full tensor contraction
or fidelity engine.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NetworkMetrics:
    network_type: str
    n_sites: int
    chi: int
    depth: int
    c_geo: int
    c_int: int
    c_total: int
    estimated_error: float


@dataclass(frozen=True)
class TrialRecord:
    run_id: str
    timestamp_utc: str
    trial_index: int
    seed: int
    n_sites: int
    chi: int
    mera_depth: int
    random_depth: int
    mera_c_geo: int
    mera_c_int: int
    mera_c_total: int
    mera_error: float
    random_c_geo: int
    random_c_int: int
    random_c_total: int
    random_error: float
    capacity_advantage: float
    capacity_ratio_random_over_mera: float
    error_ratio_random_over_mera: float
    mera_beats_random_on_capacity: bool


class RunningStats:
    """Simple streaming stats accumulator."""

    __slots__ = ("n", "sum", "min", "max")

    def __init__(self) -> None:
        self.n = 0
        self.sum = 0.0
        self.min = math.inf
        self.max = -math.inf

    def update(self, value: float) -> None:
        self.n += 1
        self.sum += value
        if value < self.min:
            self.min = value
        if value > self.max:
            self.max = value

    def mean(self) -> float | None:
        if self.n == 0:
            return None
        return self.sum / self.n

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "n": self.n,
            "sum": self.sum if self.n else None,
            "mean": self.mean(),
            "min": self.min if self.n else None,
            "max": self.max if self.n else None,
        }


class ConditionAggregator:
    """Streaming summary for one (n_sites, chi) condition."""

    def __init__(self, n_sites: int, chi: int) -> None:
        self.n_sites = n_sites
        self.chi = chi
        self.trials = 0

        self.mera_c_total = RunningStats()
        self.random_c_total = RunningStats()
        self.capacity_advantage = RunningStats()
        self.capacity_ratio = RunningStats()
        self.mera_error = RunningStats()
        self.random_error = RunningStats()
        self.error_ratio = RunningStats()

        self.mera_wins = 0

    def update(self, rec: TrialRecord) -> None:
        self.trials += 1
        self.mera_c_total.update(float(rec.mera_c_total))
        self.random_c_total.update(float(rec.random_c_total))
        self.capacity_advantage.update(float(rec.capacity_advantage))
        self.capacity_ratio.update(float(rec.capacity_ratio_random_over_mera))
        self.mera_error.update(float(rec.mera_error))
        self.random_error.update(float(rec.random_error))
        self.error_ratio.update(float(rec.error_ratio_random_over_mera))
        if rec.mera_beats_random_on_capacity:
            self.mera_wins += 1

    def to_row(self) -> dict[str, Any]:
        return {
            "n_sites": self.n_sites,
            "chi": self.chi,
            "trials": self.trials,
            "mera_c_total_mean": self.mera_c_total.mean(),
            "random_c_total_mean": self.random_c_total.mean(),
            "capacity_advantage_mean": self.capacity_advantage.mean(),
            "capacity_ratio_random_over_mera_mean": self.capacity_ratio.mean(),
            "mera_error_mean": self.mera_error.mean(),
            "random_error_mean": self.random_error.mean(),
            "error_ratio_random_over_mera_mean": self.error_ratio.mean(),
            "mera_beats_random_fraction": (self.mera_wins / self.trials) if self.trials else None,
            "mera_c_total_min": self.mera_c_total.min if self.trials else None,
            "mera_c_total_max": self.mera_c_total.max if self.trials else None,
            "random_c_total_min": self.random_c_total.min if self.trials else None,
            "random_c_total_max": self.random_c_total.max if self.trials else None,
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_mkdir_unique(base_dir: Path, prefix: str = "RUN") -> Path:
    """Create a unique output directory so nothing is overwritten."""
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    host = socket.gethostname().split(".")[0]
    pid = os.getpid()

    for _ in range(100):
        suffix = uuid.uuid4().hex[:8]
        run_dir = base_dir / f"{prefix}_{timestamp}_{host}_{pid}_{suffix}"
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue

    raise RuntimeError("Unable to create a unique run directory after 100 attempts.")


def atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write CSV with no rows.")

    fieldnames = list(rows[0].keys())
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def compute_mera_depth(n_sites: int) -> int:
    if n_sites < 2:
        return 0

    depth = 0
    current_sites = n_sites
    while current_sites >= 2:
        current_sites //= 2
        depth += 1
    return depth


def estimate_error(network_type: str, chi: int) -> float:
    """
    Proxy error model only.
    MERA is assumed to use chi more efficiently than a random TN baseline.
    """
    if chi < 1:
        raise ValueError("chi must be >= 1")

    if network_type == "mera":
        return float(chi ** -0.5)

    if network_type == "random":
        effective_chi = max(1.0, chi / 2.0)
        return float(effective_chi ** -0.5)

    raise ValueError(f"Unknown network_type: {network_type}")


def create_mera_1d(n_sites: int, chi: int) -> NetworkMetrics:
    """
    Structured 1D MERA-like capacity proxy.

    Per effective pair at each level:
    - disentangler contribution: chi^2
    - isometry contribution: 3 * chi
    """
    if n_sites < 1:
        raise ValueError("n_sites must be >= 1")
    if chi < 1:
        raise ValueError("chi must be >= 1")

    c_geo = 2 * n_sites
    c_int = 0
    depth = 0
    current_sites = n_sites

    while current_sites >= 2:
        n_pairs = current_sites // 2
        c_int += n_pairs * (chi * chi + 3 * chi)
        current_sites = n_pairs
        depth += 1

    return NetworkMetrics(
        network_type="mera",
        n_sites=n_sites,
        chi=chi,
        depth=depth,
        c_geo=c_geo,
        c_int=c_int,
        c_total=c_geo + c_int,
        estimated_error=estimate_error("mera", chi),
    )


def create_random_tn_1d(
    n_sites: int,
    chi: int,
    depth: int,
    rng: np.random.Generator,
) -> NetworkMetrics:
    """
    Random unstructured TN capacity proxy.

    Vectorized per level for speed.
    Interaction proxy per pair:
    bd_in + bd_out + bd_in * bd_out
    """
    if n_sites < 1:
        raise ValueError("n_sites must be >= 1")
    if chi < 1:
        raise ValueError("chi must be >= 1")
    if depth < 0:
        raise ValueError("depth must be >= 0")

    c_geo = 2 * n_sites
    c_int = 0
    current_sites = n_sites
    realized_depth = 0

    for _ in range(depth):
        if current_sites < 2:
            break

        n_pairs = current_sites // 2
        bd_in = rng.integers(1, chi + 1, size=n_pairs)
        bd_out = rng.integers(1, chi + 1, size=n_pairs)
        c_int += int(np.sum(bd_in + bd_out + (bd_in * bd_out)))
        current_sites = n_pairs
        realized_depth += 1

    return NetworkMetrics(
        network_type="random",
        n_sites=n_sites,
        chi=chi,
        depth=realized_depth,
        c_geo=c_geo,
        c_int=c_int,
        c_total=c_geo + c_int,
        estimated_error=estimate_error("random", chi),
    )


def build_trial_record(
    run_id: str,
    trial_index: int,
    seed: int,
    mera: NetworkMetrics,
    random_net: NetworkMetrics,
) -> TrialRecord:
    capacity_advantage = float(random_net.c_total - mera.c_total)
    capacity_ratio = float(random_net.c_total / mera.c_total) if mera.c_total > 0 else math.inf
    error_ratio = (
        float(random_net.estimated_error / mera.estimated_error)
        if mera.estimated_error > 0
        else math.inf
    )

    return TrialRecord(
        run_id=run_id,
        timestamp_utc=utc_now_iso(),
        trial_index=trial_index,
        seed=seed,
        n_sites=mera.n_sites,
        chi=mera.chi,
        mera_depth=mera.depth,
        random_depth=random_net.depth,
        mera_c_geo=mera.c_geo,
        mera_c_int=mera.c_int,
        mera_c_total=mera.c_total,
        mera_error=mera.estimated_error,
        random_c_geo=random_net.c_geo,
        random_c_int=random_net.c_int,
        random_c_total=random_net.c_total,
        random_error=random_net.estimated_error,
        capacity_advantage=capacity_advantage,
        capacity_ratio_random_over_mera=capacity_ratio,
        error_ratio_random_over_mera=error_ratio,
        mera_beats_random_on_capacity=bool(mera.c_total < random_net.c_total),
    )


def open_raw_csv_writer(path: Path) -> tuple[Any, csv.DictWriter]:
    fieldnames = list(TrialRecord.__dataclass_fields__.keys())
    f = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return f, writer


def maybe_make_plots(run_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        atomic_write_text(
            run_dir / "plots_status.txt",
            "Plot generation skipped: matplotlib is not available.\n",
        )
        return

    # group by chi for line plots vs n_sites
    by_chi: dict[int, list[dict[str, Any]]] = {}
    for row in summary_rows:
        by_chi.setdefault(int(row["chi"]), []).append(row)

    for chi, rows in by_chi.items():
        rows = sorted(rows, key=lambda r: int(r["n_sites"]))
        xs = [int(r["n_sites"]) for r in rows]
        ys = [float(r["capacity_ratio_random_over_mera_mean"]) for r in rows]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("n_sites")
        ax.set_ylabel("mean capacity ratio (random / mera)")
        ax.set_title(f"Capacity Ratio vs n_sites (chi={chi})")
        fig.tight_layout()
        fig.savefig(run_dir / f"capacity_ratio_chi_{chi}.png", dpi=150)
        plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = list(range(len(summary_rows)))
    ys = [float(r["mera_beats_random_fraction"]) for r in summary_rows]
    labels = [f"L={r['n_sites']}, χ={r['chi']}" for r in summary_rows]
    ax.plot(xs, ys, marker="o")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("MERA win fraction")
    ax.set_title("MERA Win Fraction by Condition")
    fig.tight_layout()
    fig.savefig(run_dir / "mera_win_fraction.png", dpi=150)
    plt.close(fig)


def overall_summary(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not summary_rows:
        return {
            "conditions": 0,
            "mean_capacity_ratio_random_over_mera": None,
            "median_capacity_ratio_random_over_mera": None,
            "mean_mera_beats_random_fraction": None,
            "all_conditions_mostly_favor_mera": None,
        }

    cap_ratios = np.array(
        [float(row["capacity_ratio_random_over_mera_mean"]) for row in summary_rows],
        dtype=float,
    )
    beat_fracs = np.array(
        [float(row["mera_beats_random_fraction"]) for row in summary_rows],
        dtype=float,
    )

    return {
        "conditions": len(summary_rows),
        "mean_capacity_ratio_random_over_mera": float(np.mean(cap_ratios)),
        "median_capacity_ratio_random_over_mera": float(np.median(cap_ratios)),
        "mean_mera_beats_random_fraction": float(np.mean(beat_fracs)),
        "all_conditions_mostly_favor_mera": bool(np.all(beat_fracs >= 0.5)),
    }


def build_readme(run_id: str, manifest: dict[str, Any], overall: dict[str, Any]) -> str:
    return f"""# MERA Capacity Allocator Run

Run ID: `{run_id}`
Created (UTC): `{manifest["created_utc"]}`
Seed: `{manifest["seed"]}`

## Inputs
- n_sites_values: {manifest["n_sites_values"]}
- chi_values: {manifest["chi_values"]}
- trials_per_condition: {manifest["trials_per_condition"]}

## Files
- `manifest.json` — reproducibility metadata
- `raw_trials.jsonl` — raw per-trial JSON records
- `raw_trials.csv` — raw per-trial CSV records
- `summary.csv` — grouped summary by `(n_sites, chi)`
- `summary.json` — grouped summary in JSON
- `overall_summary.json` — top-level summary
- `capacity_ratio_chi_*.png` — optional plots when `--plots` is enabled
- `mera_win_fraction.png` — optional plot when `--plots` is enabled

## Overall
- conditions: {overall.get("conditions")}
- mean capacity ratio (random / mera): {overall.get("mean_capacity_ratio_random_over_mera")}
- median capacity ratio (random / mera): {overall.get("median_capacity_ratio_random_over_mera")}
- mean MERA win fraction: {overall.get("mean_mera_beats_random_fraction")}
- all conditions mostly favor MERA: {overall.get("all_conditions_mostly_favor_mera")}
"""


def parse_int_list(value: str) -> list[int]:
    items: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parsed = int(raw)
        if parsed < 1:
            raise argparse.ArgumentTypeError("All values must be >= 1.")
        items.append(parsed)

    if not items:
        raise argparse.ArgumentTypeError("At least one integer value is required.")

    return items


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run structured MERA vs random TN capacity experiments."
    )
    parser.add_argument(
        "--n-sites",
        type=parse_int_list,
        default=[8, 16, 32, 64],
        help="Comma-separated site counts. Example: 8,16,32,64",
    )
    parser.add_argument(
        "--chi",
        type=parse_int_list,
        default=[2, 4, 8, 16],
        help="Comma-separated bond dimensions. Example: 2,4,8,16",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=25,
        help="Trials per (n_sites, chi) condition.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master RNG seed.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs") / "mera_capacity_allocator",
        help="Root directory under which unique run folders are created.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate summary plots if matplotlib is available.",
    )
    return parser


def run_experiment(
    n_sites_values: list[int],
    chi_values: list[int],
    trials_per_condition: int,
    output_root: Path,
    seed: int,
    make_plots: bool,
) -> Path:
    run_dir = safe_mkdir_unique(output_root, prefix="RUN")
    run_id = run_dir.name

    raw_jsonl_path = run_dir / "raw_trials.jsonl"
    raw_csv_path = run_dir / "raw_trials.csv"

    master_rng = np.random.default_rng(seed)

    manifest = {
        "run_id": run_id,
        "created_utc": utc_now_iso(),
        "script": Path(sys.argv[0]).name if sys.argv else "MERA_Capacity_Allocator.py",
        "output_dir": str(run_dir),
        "seed": seed,
        "n_sites_values": n_sites_values,
        "chi_values": chi_values,
        "trials_per_condition": trials_per_condition,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "hostname": socket.gethostname(),
        "plots_requested": make_plots,
    }
    atomic_write_json(run_dir / "manifest.json", manifest)

    aggregators: dict[tuple[int, int], ConditionAggregator] = {}

    raw_csv_file, raw_csv_writer = open_raw_csv_writer(raw_csv_path)
    try:
        with raw_jsonl_path.open("w", encoding="utf-8") as jsonl_file:
            for n_sites in n_sites_values:
                base_depth = compute_mera_depth(n_sites)

                for chi in chi_values:
                    key = (n_sites, chi)
                    aggregators[key] = ConditionAggregator(n_sites=n_sites, chi=chi)
                    mera = create_mera_1d(n_sites=n_sites, chi=chi)

                    for trial_index in range(trials_per_condition):
                        child_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
                        trial_rng = np.random.default_rng(child_seed)

                        random_net = create_random_tn_1d(
                            n_sites=n_sites,
                            chi=chi,
                            depth=base_depth,
                            rng=trial_rng,
                        )

                        rec = build_trial_record(
                            run_id=run_id,
                            trial_index=trial_index,
                            seed=child_seed,
                            mera=mera,
                            random_net=random_net,
                        )

                        rec_dict = asdict(rec)
                        jsonl_file.write(json.dumps(rec_dict, sort_keys=True) + "\n")
                        raw_csv_writer.writerow(rec_dict)
                        aggregators[key].update(rec)

    finally:
        raw_csv_file.close()

    summary_rows = [
        aggregators[key].to_row()
        for key in sorted(aggregators.keys(), key=lambda x: (x[0], x[1]))
    ]
    csv_write(run_dir / "summary.csv", summary_rows)
    atomic_write_json(run_dir / "summary.json", summary_rows)

    overall = overall_summary(summary_rows)
    atomic_write_json(run_dir / "overall_summary.json", overall)

    atomic_write_text(run_dir / "README.md", build_readme(run_id, manifest, overall))

    if make_plots:
        maybe_make_plots(run_dir, summary_rows)

    return run_dir


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials must be >= 1")

    run_dir = run_experiment(
        n_sites_values=args.n_sites,
        chi_values=args.chi,
        trials_per_condition=args.trials,
        output_root=args.output_root,
        seed=args.seed,
        make_plots=args.plots,
    )

    print(f"Run complete: {run_dir}")
    print(f"Raw JSONL:    {run_dir / 'raw_trials.jsonl'}")
    print(f"Raw CSV:      {run_dir / 'raw_trials.csv'}")
    print(f"Summary CSV:  {run_dir / 'summary.csv'}")
    print(f"Summary JSON: {run_dir / 'summary.json'}")
    print(f"Overall:      {run_dir / 'overall_summary.json'}")


if __name__ == "__main__":
    main()
