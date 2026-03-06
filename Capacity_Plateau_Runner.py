#!/usr/bin/env python3
"""
Capacity Plateau Scan Runner (v2 - Real MERA)

Collects MERA entropy, fidelity, and energy data across bond dimension chi,
with exact diagonalization comparison for small systems.

Outputs:
  Raw MERA measurements across chi
  ED reference values for small systems
  Log-linear and saturating model fit statistics
  ΔAIC / ΔBIC for later interpretation

Usage:
  python3 Capacity_Plateau_Runner.py --L 8 --A_size 4 \
    --model ising_cyclic --chi_sweep 2,4,8,16 --seed 42 --output <DIR>
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Allow local module imports when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from claim3.PHYS_PHYSICAL_CONVERGENCE_runner_v2 import (
    exact_diagonalization,
    optimize_mera_for_fidelity,
)


def run_id() -> str:
    t = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    r = os.urandom(4).hex()
    return f"{t}_{r}"


def aic_bic_from_rss(rss: float, n: int, k: int, eps: float = 1e-12) -> tuple[float, float]:
    """Compute AIC/BIC from residual sum of squares."""
    rss = max(float(rss), eps)
    aic = n * math.log(rss / n) + 2 * k
    bic = n * math.log(rss / n) + k * math.log(n)
    return float(aic), float(bic)


def fit_sat(chis: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Fit saturating model: S(chi) = S_inf - c * chi^(-alpha)."""
    best = {
        "S_inf": float("nan"),
        "c": float("nan"),
        "alpha": float("nan"),
        "rss": float("inf"),
        "aic": float("inf"),
        "bic": float("inf"),
    }

    max_y = float(np.max(y))

    for s_inf_mult in [1.0, 1.01, 1.02, 1.03, 1.05, 1.07, 1.1, 1.15, 1.2, 1.5, 2.0]:
        s_inf = max_y * s_inf_mult + 0.05
        delta = s_inf - y
        valid = delta > 1e-12

        if np.sum(valid) < 3:
            continue

        log_d = np.log(delta[valid])
        log_chis = np.log(chis[valid])
        x = np.column_stack([-log_chis, np.ones_like(log_chis)])

        try:
            coef, _, _, _ = np.linalg.lstsq(x, log_d, rcond=None)
            alpha, log_c0 = float(coef[0]), float(coef[1])
            c = float(np.exp(log_c0))
            yhat = s_inf - c * np.power(chis, -alpha)
            rss = float(np.sum((y - yhat) ** 2))
            aic, bic = aic_bic_from_rss(rss, n=len(y), k=3)

            if rss < best["rss"]:
                best = {
                    "S_inf": float(s_inf),
                    "c": float(c),
                    "alpha": float(alpha),
                    "rss": float(rss),
                    "aic": float(aic),
                    "bic": float(bic),
                }
        except Exception:
            continue

    return best


def fit_loglin(log_chis: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Fit log-linear model: S = a + b*log(chi)."""
    x = np.column_stack([log_chis, np.ones_like(log_chis)])
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    b, a = float(coef[0]), float(coef[1])
    yhat = x @ coef
    rss = float(np.sum((y - yhat) ** 2))
    aic, bic = aic_bic_from_rss(rss, n=len(y), k=2)
    return {"a": a, "b": b, "rss": rss, "aic": aic, "bic": bic}


def real_mera(cfg: Dict) -> Dict:
    """
    Run MERA data collection.

    For L <= 12:
      Use exact diagonalization as a reference when available.

    For larger systems:
      Collect MERA-only measurements and fit statistics.
    """
    L = cfg["L"]
    A_size = cfg["A_size"]
    model = cfg["model"]
    chi_sweep = cfg["chi_sweep"]
    seed = cfg["seed"]

    print(f"Running capacity plateau scan for L={L}, A_size={A_size}")
    print(f"Model: {model}, chi values: {chi_sweep}")

    ed_result: Optional[object] = None
    if L <= 12:
        print(f"Computing ED reference for L={L}...")

        is_heis = "heisenberg" in model
        j = 1.0
        h = 0.0 if is_heis else 1.0

        ed_result = exact_diagonalization(
            L=L,
            model=model,
            A_size=A_size,
            j=j,
            h=h,
        )

        print(
            f"ED: E0={ed_result.ground_state_energy:.6f}, "
            f"S={ed_result.entanglement_entropy:.6f}"
        )

    records = []
    for chi in chi_sweep:
        print(f"Optimizing MERA with chi={chi}...")

        is_heis = "heisenberg" in model
        opt_result = optimize_mera_for_fidelity(
            L=L,
            chi=chi,
            ed_psi=ed_result.ground_state_psi if ed_result else None,
            model=model,
            steps=cfg.get("fit_steps", 80),
            seed=seed,
            j=1.0,
            h=0.0 if is_heis else 1.0,
        )

        record = {
            "chi": int(chi),
            "entropy": float(opt_result.entropy),
            "fidelity": float(opt_result.fidelity),
            "energy": float(opt_result.final_energy),
            "converged": bool(opt_result.converged),
        }

        if ed_result is not None:
            record["ed_entropy"] = float(ed_result.entanglement_entropy)
            record["entropy_error"] = float(
                abs(opt_result.entropy - ed_result.entanglement_entropy)
            )

        records.append(record)
        print(
            f"chi={chi}: S={opt_result.entropy:.4f}, "
            f"fid={opt_result.fidelity:.6f}, conv={opt_result.converged}"
        )

    chis_arr = np.array([r["chi"] for r in records], dtype=float)
    log_chis = np.log(chis_arr)
    ents = np.array([r["entropy"] for r in records], dtype=float)

    loglin = fit_loglin(log_chis, ents)
    sat = fit_sat(chis_arr, ents)
    delta_aic = float(sat["aic"] - loglin["aic"])
    delta_bic = float(sat["bic"] - loglin["bic"])

    monotonic_entropy = all(
        records[i + 1]["entropy"] >= records[i]["entropy"] - 1e-9
        for i in range(len(records) - 1)
    )
    high_fidelity = (
        all(r["fidelity"] > 0.9 for r in records) if ed_result is not None else None
    )

    return {
        "metadata": {
            "run_id": run_id(),
            "timestamp": dt.datetime.utcnow().isoformat(),
            "config": cfg,
            "test": "Capacity Plateau",
            "version": "2.0.0",
        },
        "measurements": records,
        "fits": {
            "loglinear": loglin,
            "saturating": sat,
            "delta_aic": delta_aic,
            "delta_bic": delta_bic,
        },
        "ed_reference": (
            {
                "energy": float(ed_result.ground_state_energy),
                "entropy": float(ed_result.entanglement_entropy),
            }
            if ed_result is not None
            else None
        ),
        "derived_checks": {
            "sat_preferred": bool(delta_aic < 0),
            "monotonic_entropy": bool(monotonic_entropy),
            "high_fidelity": high_fidelity,
        },
    }


def write_out(res: Dict, out_dir: Path) -> None:
    """Write results to output directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(res["metadata"], f, indent=2, default=str)

    with open(out_dir / "raw.csv", "w", newline="") as f:
        fieldnames = ["chi", "entropy", "fidelity", "energy", "converged"]
        if res["measurements"] and res["measurements"][0].get("ed_entropy") is not None:
            fieldnames.extend(["ed_entropy", "entropy_error"])

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(res["measurements"])

    with open(out_dir / "fits.json", "w") as f:
        json.dump(res["fits"], f, indent=2)

    summary = {
        "metadata": res["metadata"],
        "ed_reference": res.get("ed_reference"),
        "fits": res["fits"],
        "derived_checks": res.get("derived_checks"),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[P2] Results written to {out_dir}")
    print(f"[P2] ΔAIC = {res['fits']['delta_aic']:.4f}")
    print(f"[P2] ΔBIC = {res['fits']['delta_bic']:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description="P2 Capacity Plateau Scan (Real MERA)")
    p.add_argument("--chi_sweep", default="2,4,8,16,32")
    p.add_argument("--L", type=int, default=8)
    p.add_argument("--A_size", type=int, default=4)
    p.add_argument(
        "--model",
        default="ising_cyclic",
        choices=[
            "ising_open",
            "ising_cyclic",
            "heisenberg_open",
            "heisenberg_cyclic",
        ],
    )
    p.add_argument("--fit_steps", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    cfg = {
        "chi_sweep": [int(x) for x in a.chi_sweep.split(",") if x.strip()],
        "L": a.L,
        "A_size": a.A_size,
        "model": a.model,
        "fit_steps": a.fit_steps,
        "seed": a.seed,
    }

    print("[P2] Capacity Plateau Scan v2.0")
    print(f"[P2] L={cfg['L']}, A_size={cfg['A_size']}, model={cfg['model']}")

    res = real_mera(cfg)
    write_out(res, Path(a.output))


if __name__ == "__main__":
    main()
