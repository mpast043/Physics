#!/usr/bin/env python3
"""
Capacity Plateau Scan Runner (v2 - Real MERA)

Tests capacity-limited saturation using real MERA simulations with exact
diagonalization comparison for L=8 systems.

Falsifiers:
  Saturating model preferred (ΔAIC < 0)
  Monotonic entropy increase with chi

Usage:
  python3 Capacity_Plateau_Runner.py --L 8 --A_size 4 \\
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
from typing import Dict, List, Tuple, Optional

import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent))
from claim3.PHYS_PHYSICAL_CONVERGENCE_runner_v2 import (
    exact_diagonalization, optimize_mera_for_fidelity, Config, EDResult
)


def run_id():
    t = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    r = os.urandom(4).hex()
    return f"{t}_{r}"


def fit_linear(x, y):
    """Linear fit: y = a*x + b"""
    X = np.column_stack([x, np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    yhat = X @ coef
    rss = float(np.sum((y - yhat) ** 2))
    return {"a": a, "b": b, "rss": rss}


def aic_bic_from_rss(rss, n, k, eps=1e-12):
    """AIC/BIC from residual sum of squares"""
    rss = max(float(rss), eps)
    aic = n * math.log(rss / n) + 2 * k
    bic = n * math.log(rss / n) + k * math.log(n)
    return float(aic), float(bic)


def fit_sat(chis, y):
    """Fit saturating model: S(chi) = S_inf - c * chi^(-alpha)"""
    best = {"S_inf": float("nan"), "c": float("nan"), "alpha": float("nan"),
            "rss": float("inf"), "aic": float("inf"), "bic": float("inf")}
    max_y = float(np.max(y))
    
    for S_inf_mult in [1.0, 1.01, 1.02, 1.03, 1.05, 1.07, 1.1, 1.15, 1.2, 1.5, 2.0]:
        S_inf = max_y * S_inf_mult + 0.05
        delta = S_inf - y
        valid = delta > 1e-12
        if np.sum(valid) < 3:
            continue
        log_d = np.log(delta[valid])
        log_c = np.log(chis[valid])
        X = np.column_stack([-log_c, np.ones_like(log_c)])
        try:
            coef, _, _, _ = np.linalg.lstsq(X, log_d, rcond=None)
            alpha, log_c0 = float(coef[0]), float(coef[1])
            c = np.exp(log_c0)
            yhat = S_inf - c * np.power(chis, -alpha)
            rss = float(np.sum((y - yhat) ** 2))
            aic, bic = aic_bic_from_rss(rss, n=len(y), k=3)
            if rss < best["rss"]:
                best = {
                    "S_inf": float(S_inf), "c": float(c), "alpha": float(alpha),
                    "rss": float(rss), "aic": float(aic), "bic": float(bic)
                }
        except Exception:
            continue
    return best


def fit_loglin(log_chis, y):
    """Fit log-linear model: S = a + b*log(chi)"""
    X = np.column_stack([log_chis, np.ones_like(log_chis)])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    b, a = float(coef[0]), float(coef[1])
    yhat = X @ coef
    rss = float(np.sum((y - yhat) ** 2))
    aic, bic = aic_bic_from_rss(rss, n=len(y), k=2)
    return {"a": a, "b": b, "rss": rss, "aic": aic, "bic": bic}


def real_mera(cfg: Dict) -> Dict:
    """
    Run P2 test using real MERA optimization.

    For L=8: Use exact diagonalization as ground truth
    For L>8: Use largest chi as reference

    """
    L = cfg["L"]
    A_size = cfg["A_size"]
    model = cfg["model"]
    chi_sweep = cfg["chi_sweep"]
    seed = cfg["seed"]
    
    print(f"Running capacity plateau scan for L={L}, A_size={A_size}")
    print(f"Model: {model}, chi values: {chi_sweep}")
    
    # Get ED reference for small systems
    ed_result = None
    if L <= 12:
        print(f"Computing ED reference for L={L}...")
        is_heis = "heisenberg" in model
        j = 1.0 if not is_heis else 1.0
        h = 1.0 if not is_heis else 0.0
        ed_result = exact_diagonalization(
            L=L, 
            model="heisenberg_open" if is_heis else "ising_open",
            A_size=A_size,
            j=j, 
            h=h
        )
        print(f"ED: E0={ed_result.ground_state_energy:.6f}, S={ed_result.entanglement_entropy:.6f}")
    
    # Run MERA for each chi
    records = []
    for chi in chi_sweep:
        print(f"Optimizing MERA with chi={chi}...")
        
        opt_result = optimize_mera_for_fidelity(
            L=L,
            chi=chi,
            ed_psi=ed_result.ground_state_psi if ed_result else None,
            model="heisenberg_open" if "heisenberg" in model else "ising_open",
            steps=cfg.get("fit_steps", 80),
            seed=seed,
            j=1.0,
            h=1.0
        )
        
        record = {
            "chi": chi,
            "entropy": opt_result.entropy,
            "fidelity": opt_result.fidelity,
            "energy": opt_result.final_energy,
            "converged": opt_result.converged,
        }
        if ed_result:
            record["ed_entropy"] = ed_result.entanglement_entropy
            record["entropy_error"] = abs(opt_result.entropy - ed_result.entanglement_entropy)
        
        records.append(record)
        print(f"chi={chi}: S={opt_result.entropy:.4f}, fid={opt_result.fidelity:.6f}")
    
    # Model fitting
    chis_arr = np.array([r["chi"] for r in records], dtype=float)
    log_chis = np.log(chis_arr)
    ents = np.array([r["entropy"] for r in records], dtype=float)
    
    loglin = fit_loglin(log_chis, ents)
    sat = fit_sat(chis_arr, ents)
    delta_aic = sat["aic"] - loglin["aic"]
    
    # Falsifier checks
    p21 = delta_aic < 0  # Saturating model preferred
    p22 = all(records[i+1]["entropy"] >= records[i]["entropy"] - 1e-9
              for i in range(len(records)-1))  # Monotonic

    # Enhanced checks
    p23 = all(r["fidelity"] > 0.9 for r in records) if ed_result else None

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
            "delta_aic": float(delta_aic),
            "delta_bic": float(sat["bic"] - loglin["bic"])
        },
        "ed_reference": {
            "energy": ed_result.ground_state_energy if ed_result else None,
            "entropy": ed_result.entanglement_entropy if ed_result else None,
        } if ed_result else None,
        "verdict": "ACCEPT" if (p21 and p22) else "REJECT",
        "passed": {
            "P2.1_sat_preferred": p21,
            "P2.2_monotonic": p22,
            "P2.3_fidelity": p23,
        }
    }


def write_out(res: Dict, out_dir: Path):
    """Write results to output directory"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Metadata
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(res["metadata"], f, indent=2, default=str)
    
    # Raw data
    with open(out_dir / "raw.csv", "w", newline="") as f:
        fieldnames = ["chi", "entropy", "fidelity", "energy", "converged"]
        if res["measurements"][0].get("ed_entropy") is not None:
            fieldnames.extend(["ed_entropy", "entropy_error"])
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(res["measurements"])
    
    # Fits
    with open(out_dir / "fits.json", "w") as f:
        json.dump(res["fits"], f, indent=2)
    
    # Verdict
    verdict = {
        "test": "P2",
        "test_name": "capacity_threshold_plateau_scan",
        "verdict": res["verdict"],
        "status": "COMPLETE",
        "passed": res["passed"],
        "delta_aic": res["fits"]["delta_aic"],
    }
    with open(out_dir / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    
    print(f"[P2] Results written to {out_dir}")
    print(f"[P2] Verdict: {res['verdict']}")
    print(f"[P2] ΔAIC = {res['fits']['delta_aic']:.4f}")


def main():
    p = argparse.ArgumentParser(description="P2 Capacity Plateau Scan (Real MERA)")
    p.add_argument("--chi_sweep", default="2,4,8,16,32")
    p.add_argument("--L", type=int, default=8)
    p.add_argument("--A_size", type=int, default=4)
    p.add_argument("--model", default="ising_cyclic", 
                   choices=["ising_open", "ising_cyclic", "heisenberg_open", "heisenberg_cyclic"])
    p.add_argument("--fit_steps", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    
    cfg = {
        "chi_sweep": [int(x) for x in a.chi_sweep.split(",")],
        "L": a.L,
        "A_size": a.A_size,
        "model": a.model,
        "fit_steps": a.fit_steps,
        "seed": a.seed,
    }
    
    print(f"[P2] Capacity Plateau Scan v2.0")
    print(f"[P2] L={cfg['L']}, A_size={cfg['A_size']}, model={cfg['model']}")
    
    res = real_mera(cfg)
    write_out(res, Path(a.output))


if __name__ == "__main__":
    main()
