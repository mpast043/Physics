#!/usr/bin/env python3
"""
Capacity of Entanglement (C_E) Pipeline
========================================
Connects entanglement_utils.py to ED ground states for the XXZ chain.

C_E = Var(H_A) = Σ_i λ_i (ln λ_i)^2 - S^2
    where λ_i are Schmidt eigenvalues of rho_A,
    S = -Σ λ_i ln λ_i  (von Neumann entropy)

Holographic bound (de Boer et al. PRD 2019):
  - Critical (CFT) phases: S / C_E → 1 as c → ∞
  - Gapped phases: S / C_E takes a different characteristic value

Usage:
    python3 ce_pipeline.py
"""

from __future__ import annotations
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_PHYSICS = Path("/tmp/host-adapters-live/experiments/physics")
DATA_DIR     = Path("/tmp/host-adapters-experimental-data")
ED_RESULTS   = DATA_DIR / "xxz_boundary_ed_gate_v2/run_ed_da14eeb2/xxz_boundary_results.json"
VAULT        = Path("/sessions/beautiful-modest-einstein/mnt/Obsidian Vault")

sys.path.insert(0, str(REPO_PHYSICS))

# ── Import entanglement_utils (rho-based API) ──────────────────────────────────
from entanglement_utils import (
    capacity_of_entanglement,
    von_neumann_entropy,
    reduced_density_matrix,
)
print("[ok] entanglement_utils imported (rho-based API)")


# ── ED: build XXZ Hamiltonian and ground state (sparse) ───────────────────────
def xxz_hamiltonian_sparse(L: int, delta: float) -> sp.csr_matrix:
    """Sparse XXZ Hamiltonian, OBC, full 2^L Hilbert space."""
    n = 2**L
    rows, cols, vals = [], [], []
    for i in range(L - 1):
        for s in range(n):
            si = (s >> i) & 1
            sj = (s >> (i + 1)) & 1
            # Sz_i Sz_{i+1} diagonal
            diag_val = delta * (si - 0.5) * (sj - 0.5)
            rows.append(s); cols.append(s); vals.append(diag_val)
            # S+S- / S-S+ off-diagonal
            if si != sj:
                sf = s ^ (1 << i) ^ (1 << (i + 1))
                rows.append(s); cols.append(sf); vals.append(0.5)
    H = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return H


def ground_state(L: int, delta: float) -> Tuple[float, np.ndarray]:
    """Return (E0, normalised ground-state vector) via sparse Lanczos."""
    H = xxz_hamiltonian_sparse(L, delta)
    if L <= 10:
        # Small enough for dense eigh
        Hd = H.toarray()
        evals, evecs = np.linalg.eigh(Hd)
        E0 = float(evals[0])
        psi0 = evecs[:, 0].real
    else:
        # Sparse Lanczos — find smallest algebraic eigenvalue
        evals, evecs = spla.eigsh(H, k=1, which="SA", tol=1e-12)
        E0 = float(evals[0])
        psi0 = evecs[:, 0].real
    psi0 /= np.linalg.norm(psi0)
    return E0, psi0


# ── Load reference S values ───────────────────────────────────────────────────
def load_s_ref(path: Path) -> Dict[float, Dict[int, float]]:
    with open(path) as f:
        d = json.load(f)
    out: Dict[float, Dict[int, float]] = {}
    for r in d["results"]:
        out[float(r["delta"])] = {
            int(m["L"]): float(m["S_entropy"])
            for m in r.get("measurements", [])
        }
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    s_ref  = load_s_ref(ED_RESULTS)
    deltas = [0.5, 2.0]
    Ls     = [8, 12, 16]
    results: Dict = {}

    for delta in deltas:
        results[delta] = {}
        phase = "XY crit." if delta < 1 else "Néel gap."
        print(f"\n=== Δ={delta} ({phase}) ===")
        for L in Ls:
            # Build ground state
            print(f"  L={L}: ED...", end=" ", flush=True)
            E0, psi0 = ground_state(L, delta)

            # Reduced density matrix for left half-chain (sites 0..L/2-1)
            subsystem_A = list(range(L // 2))
            rho_A = reduced_density_matrix(psi0, subsystem_A, L)

            # Entanglement measures
            S   = von_neumann_entropy(rho_A)
            CE  = capacity_of_entanglement(rho_A)
            ratio = S / CE if CE > 1e-15 else float("inf")

            # Cross-check: S_ref is in bits (benchmark); convert to nats
            S_ref_bits = s_ref.get(delta, {}).get(L)
            S_ref = S_ref_bits * math.log(2) if S_ref_bits is not None else None
            err   = abs(S - S_ref) if S_ref is not None else None

            results[delta][L] = {
                "L": L, "delta": delta, "E0": E0,
                "S": S, "S_ref": S_ref, "S_err": err,
                "CE": CE, "S_over_CE": ratio,
            }
            print(f"S={S:.6f} (err={err:.1e}), C_E={CE:.6f}, S/C_E={ratio:.4f}")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_json = VAULT / "ce_results.json"
    serial = {str(d): {str(L): v for L, v in rv.items()} for d, rv in results.items()}
    with open(out_json, "w") as f:
        json.dump(serial, f, indent=2)
    print(f"\n[saved] {out_json.name}")

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'Δ':>5} {'L':>4} {'S':>10} {'C_E':>10} {'S/C_E':>8}  Phase")
    print("-"*65)
    for delta in deltas:
        phase = "XY crit." if delta < 1 else "Néel gap."
        for L in Ls:
            r = results[delta][L]
            print(f"{delta:>5.1f} {L:>4d}  {r['S']:>9.6f}  {r['CE']:>9.6f}  "
                  f"{r['S_over_CE']:>7.4f}  {phase}")
    print("="*65)

    # ── Figure ─────────────────────────────────────────────────────────────────
    _make_figure(results, deltas, Ls)
    return results


def _make_figure(results, deltas, Ls):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = {0.5: "#1f77b4", 2.0: "#d62728"}
    LABELS = {0.5: r"$\Delta=0.5$ (XY, \textsc{out})", 2.0: r"$\Delta=2.0$ (Néel, \textsc{in})"}

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(r"Capacity of entanglement $C_E$ from ED ground states",
                 fontsize=11, fontfamily="serif")

    # Panel (a): S and C_E vs L
    ax = axes[0]
    for d in deltas:
        c  = COLORS[d]
        Lv = Ls
        Sv  = [results[d][L]["S"]  for L in Lv]
        CEv = [results[d][L]["CE"] for L in Lv]
        ax.plot(Lv, Sv,  "o-",  color=c, lw=1.8, ms=6, label=fr"$S$, $\Delta={d}$")
        ax.plot(Lv, CEv, "s--", color=c, lw=1.4, ms=6, alpha=0.75,
                label=fr"$C_E$, $\Delta={d}$")
    ax.set_xlabel(r"$L$", fontfamily="serif")
    ax.set_ylabel("Entropy (nats)", fontfamily="serif")
    ax.set_title("(a) $S$ and $C_E$ vs $L$", fontfamily="serif")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.tick_params(direction="in");  ax.grid(True, lw=0.3, alpha=0.4)

    # Panel (b): ratio S/C_E
    ax = axes[1]
    for d in deltas:
        c = COLORS[d]
        rv = [results[d][L]["S_over_CE"] for L in Ls]
        ax.plot(Ls, rv, "D-", color=c, lw=1.8, ms=6, label=fr"$\Delta={d}$")
    ax.axhline(1.0, lw=0.8, ls=":", color="k", label=r"$S/C_E=1$ (holo.)")
    ax.set_xlabel(r"$L$", fontfamily="serif")
    ax.set_ylabel(r"$S\,/\,C_E$", fontfamily="serif")
    ax.set_title(r"(b) Holographic ratio $S/C_E$", fontfamily="serif")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_ylim(0, 1.5)
    ax.tick_params(direction="in");  ax.grid(True, lw=0.3, alpha=0.4)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = VAULT / f"XXZ_Fig2_CE.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        print(f"[saved] {p.name}")


if __name__ == "__main__":
    run()
