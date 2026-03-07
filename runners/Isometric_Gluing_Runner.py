#!/usr/bin/env python3
"""
Isometric Gluing Diagnostics Runner

This script computes exact ground-state reduced density matrix diagnostics for a
bipartitioned spin chain. It does not claim a full MERA gluing verification.
Instead, it reports data relevant to later interpretation:

1. Exact ground-state energy
2. Reduced density matrices for A and B
3. Entropies derived from rho_A and rho_B
4. Density-matrix validity diagnostics
5. Entropic consistency checks:
   - subadditivity: S(AB) <= S(A) + S(B)
   - Araki-Lieb: |S(A) - S(B)| <= S(AB)

Outputs:
  metadata.json
  ed_reference.json
  measurements.json
  summary.json

Usage:
  python3 Isometric_Gluing_Runner.py --L 8 --A_size 4 \
    --model heisenberg_open --output <DIR>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np


def run_id() -> str:
    t = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    r = os.urandom(4).hex()
    return f"{t}_{r}"


def build_ising_hamiltonian(L: int, j: float = 1.0, h: float = 1.0) -> np.ndarray:
    """Build open-boundary transverse-field Ising Hamiltonian."""
    dim = 2**L
    H = np.zeros((dim, dim), dtype=np.float64)

    I = np.eye(2, dtype=np.float64)
    X = np.array([[0, 1], [1, 0]], dtype=np.float64)
    Z = np.array([[1, 0], [0, -1]], dtype=np.float64)

    for i in range(L - 1):
        ops = [I] * L
        ops[i] = Z
        ops[i + 1] = Z
        zz = ops[0]
        for op in ops[1:]:
            zz = np.kron(zz, op)
        H -= (j / 4.0) * zz

    for i in range(L):
        ops = [I] * L
        ops[i] = X
        x_i = ops[0]
        for op in ops[1:]:
            x_i = np.kron(x_i, op)
        H -= (h / 2.0) * x_i

    return H


def build_heisenberg_hamiltonian(L: int, J: float = 1.0) -> np.ndarray:
    """Build open-boundary Heisenberg Hamiltonian."""
    dim = 2**L
    H = np.zeros((dim, dim), dtype=np.complex128)

    I = np.eye(2, dtype=np.complex128)
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=np.complex128)

    for i in range(L - 1):
        for S1, S2 in ((Sx, Sx), (Sy, Sy), (Sz, Sz)):
            ops = [I] * L
            ops[i] = S1
            ops[i + 1] = S2
            s_dot_s = ops[0]
            for op in ops[1:]:
                s_dot_s = np.kron(s_dot_s, op)
            H += J * s_dot_s

    return H


def compute_entanglement_entropy(psi: np.ndarray, L: int, A_size: int) -> float:
    """Compute von Neumann entropy across the A|B cut."""
    dim_A = 2**A_size
    dim_B = 2 ** (L - A_size)
    psi_matrix = psi.reshape(dim_A, dim_B)
    _, s, _ = np.linalg.svd(psi_matrix, full_matrices=False)
    eigvals = s**2
    eigvals = eigvals[eigvals > 1e-12]
    return float(-np.sum(eigvals * np.log(eigvals)))


def get_reduced_density_matrix(psi: np.ndarray, L: int, A_size: int) -> np.ndarray:
    """Get reduced density matrix rho_A = Tr_B(|psi><psi|) for a contiguous cut."""
    dim_A = 2**A_size
    dim_B = 2 ** (L - A_size)
    psi_matrix = psi.reshape(dim_A, dim_B)
    rho_A = psi_matrix @ psi_matrix.conj().T
    rho_A = rho_A / np.trace(rho_A)
    return rho_A


def exact_diagonalization(
    L: int,
    model: str,
    A_size: int,
    j: float = 1.0,
    h: float = 1.0,
) -> Dict[str, Any]:
    """Perform exact diagonalization."""
    if model == "ising":
        H = build_ising_hamiltonian(L, j, h)
    elif model == "heisenberg":
        H = build_heisenberg_hamiltonian(L, j)
    else:
        raise ValueError(f"Unknown model: {model}")

    eigenvalues, eigenvectors = np.linalg.eigh(H)
    idx = int(np.argmin(eigenvalues))
    E0 = float(eigenvalues[idx])
    psi0 = eigenvectors[:, idx]
    S_cut = compute_entanglement_entropy(psi0, L, A_size)

    return {
        "ground_state_energy": E0,
        "ground_state_psi": psi0,
        "cut_entropy": S_cut,
        "n_sites": L,
    }


def check_subadditivity(S_A: float, S_B: float, S_AB: float, epsilon: float = 1e-6) -> bool:
    """Check S(AB) <= S(A) + S(B)."""
    return bool(S_AB <= S_A + S_B + epsilon)


def check_araki_lieb(S_A: float, S_B: float, S_AB: float, epsilon: float = 1e-6) -> bool:
    """Check |S(A) - S(B)| <= S(AB)."""
    return bool(abs(S_A - S_B) <= S_AB + epsilon)


def compute_rho_spectrum(rho: np.ndarray) -> np.ndarray:
    """Return filtered normalized eigenvalues for entropy calculations."""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    if len(eigenvalues) == 0:
        return eigenvalues
    return eigenvalues / np.sum(eigenvalues)


def density_matrix_diagnostics(rho: np.ndarray) -> Dict[str, Any]:
    """Compute validity diagnostics from the raw density matrix."""
    raw_eigs = np.linalg.eigvalsh(rho)
    hermitian = bool(np.allclose(rho, rho.conj().T, atol=1e-10))
    trace_one = bool(abs(np.trace(rho) - 1.0) < 1e-8)
    positive_semidefinite = bool(np.all(raw_eigs >= -1e-10))
    min_eigenvalue = float(np.min(raw_eigs))
    max_eigenvalue = float(np.max(raw_eigs))

    filtered = compute_rho_spectrum(rho)
    entropy = float(-np.sum(filtered * np.log(filtered))) if len(filtered) > 0 else 0.0

    return {
        "hermitian": hermitian,
        "trace_one": trace_one,
        "positive_semidefinite": positive_semidefinite,
        "min_eigenvalue": min_eigenvalue,
        "max_eigenvalue": max_eigenvalue,
        "rank_filtered": int(len(filtered)),
        "entropy": entropy,
        "spectrum_filtered": filtered.tolist(),
    }


def convert_numpy(obj: Any) -> Any:
    """Convert numpy types to Python-native JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    if isinstance(obj, tuple):
        return [convert_numpy(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.complexfloating):
        return {"real": float(np.real(obj)), "imag": float(np.imag(obj))}
    return obj


def run_isometric(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run density-matrix and entropy diagnostics for a bipartitioned exact ground state.

    Note:
      This is a data/diagnostic runner. It does not emit an ACCEPT/REJECT verdict
      for MERA isometric gluing.
    """
    L = cfg["L"]
    A_size = cfg["A_size"]
    B_size = L - A_size
    model = cfg["model"]

    print("[P3-ISO] Isometric Gluing Diagnostics")
    print(f"[P3-ISO] L={L}, A={A_size}, B={B_size}")
    print(f"[P3-ISO] Model: {model}")

    ed_model = "heisenberg" if "heisenberg" in model else "ising"
    ed_result = exact_diagonalization(
        L=L,
        model=ed_model,
        A_size=A_size,
        j=1.0,
        h=1.0 if ed_model == "ising" else 0.0,
    )

    psi_full = ed_result["ground_state_psi"]
    E0 = ed_result["ground_state_energy"]
    S_cut = ed_result["cut_entropy"]

    rho_A = get_reduced_density_matrix(psi_full, L, A_size)
    rho_B = get_reduced_density_matrix(psi_full, L, B_size)

    diag_A = density_matrix_diagnostics(rho_A)
    diag_B = density_matrix_diagnostics(rho_B)

    S_A = float(diag_A["entropy"])
    S_B = float(diag_B["entropy"])

    subadd = check_subadditivity(S_A, S_B, S_cut)
    araki_lieb = check_araki_lieb(S_A, S_B, S_cut)

    excision_consistency = bool(np.allclose(rho_A, get_reduced_density_matrix(psi_full, L, A_size), atol=1e-10))

    print("\n[P3-ISO] Results:")
    print(f"  E0={E0:.6f}, S_cut={S_cut:.6f}")
    print(f"  S_A={S_A:.6f}, S_B={S_B:.6f}")
    print(
        f"  rho_A valid: {diag_A['hermitian'] and diag_A['trace_one'] and diag_A['positive_semidefinite']}"
    )
    print(
        f"  rho_B valid: {diag_B['hermitian'] and diag_B['trace_one'] and diag_B['positive_semidefinite']}"
    )
    print(f"  Subadditivity: {subadd}")
    print(f"  Araki-Lieb: {araki_lieb}")

    return {
        "metadata": {
            "run_id": run_id(),
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "config": cfg,
            "test": "ISOMETRIC_GLUING_DIAGNOSTICS",
            "version": "5.0.0",
        },
        "ed_reference": {
            "energy": E0,
            "cut_entropy": S_cut,
            "model_used": ed_model,
        },
        "measurements": {
            "L": L,
            "A_size": A_size,
            "B_size": B_size,
            "S_cut": S_cut,
            "S_A": S_A,
            "S_B": S_B,
            "rho_A_diagnostics": diag_A,
            "rho_B_diagnostics": diag_B,
            "rho_A_shape": list(rho_A.shape),
            "rho_B_shape": list(rho_B.shape),
        },
        "derived_checks": {
            "rho_A_valid": bool(
                diag_A["hermitian"] and diag_A["trace_one"] and diag_A["positive_semidefinite"]
            ),
            "rho_B_valid": bool(
                diag_B["hermitian"] and diag_B["trace_one"] and diag_B["positive_semidefinite"]
            ),
            "excision_consistency": excision_consistency,
            "subadditivity": subadd,
            "araki_lieb": araki_lieb,
        },
    }


def write_out(res: Dict[str, Any], out_dir: Path) -> None:
    """Write results to output directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res_clean = convert_numpy(res)

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(res_clean["metadata"], f, indent=2)

    with open(out_dir / "ed_reference.json", "w") as f:
        json.dump(res_clean["ed_reference"], f, indent=2)

    with open(out_dir / "measurements.json", "w") as f:
        json.dump(res_clean["measurements"], f, indent=2)

    summary = {
        "metadata": res_clean["metadata"],
        "ed_reference": res_clean["ed_reference"],
        "derived_checks": res_clean["derived_checks"],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[P3-ISO] Results written to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="P3 Isometric Gluing Diagnostics")
    p.add_argument("--L", type=int, default=8)
    p.add_argument("--A_size", "--A-size", type=int, default=4)
    p.add_argument("--model", default="heisenberg_open")
    p.add_argument("--output", required=True)
    a = p.parse_args()

    cfg = {
        "L": a.L,
        "A_size": a.A_size,
        "model": a.model,
    }

    res = run_isometric(cfg)
    write_out(res, Path(a.output))


if __name__ == "__main__":
    main()
