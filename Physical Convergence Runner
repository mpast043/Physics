#!/usr/bin/env python3
"""
Claim 3P Physical Convergence Runner v2
FRAMEWORK_SPEC_v0.2.1 compliant

Phase 1: ED Gold Standard for Ising and Heisenberg
Phase 2: MERA variational optimization (energy minimization with local terms)
Phase 3: Compute entropy and fidelity to ED reference
Phase 4: Acceptance criteria P3.1-P3.4

Usage:
  python3 exp3_claim3_physical_convergence_runner_v2.py --L 8 --A_size 4 \\
    --model ising_open --j 1.0 --h 1.0 --chi_sweep 2,4,8,16 \\
    --restarts_per_chi 3 --fit_steps 100 --seed 42 --output <RUN_DIR>
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

# Import entanglement_utils - try relative first, then absolute
try:
    from ..entanglement_utils import (
        von_neumann_entropy,
        reduced_density_matrix,
        entanglement_spectrum,
        entanglement_gap,
    )
except ImportError:
    # For direct module execution, use absolute import
    from entanglement_utils import (
        von_neumann_entropy,
        reduced_density_matrix,
        entanglement_spectrum,
        entanglement_gap,
    )


# ============================================================
# Types and Configuration
# ============================================================


@dataclass
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
    eps_fid: float = 1e-4
    eps_S: float = 1e-3


@dataclass
class EDResult:
    ground_state_energy: float
    ground_state_psi: np.ndarray
    entanglement_entropy: float
    n_sites: int


@dataclass
class OptimizationResult:
    chi: int
    restart_idx: int
    fidelity: float
    entropy: float
    final_energy: float
    seed: int
    converged: bool = True
    num_steps: Optional[int] = None


# ============================================================
# Exact Diagonalization (SPARSE - quimb based)
# ============================================================


def compute_entanglement_entropy_from_rho_A(rho_A: np.ndarray) -> float:
    """Von Neumann entropy from reduced density matrix (small, already reduced)."""
    eigvals = np.linalg.eigvalsh(rho_A)
    eigvals = eigvals[eigvals > 1e-12]
    S = -np.sum(eigvals * np.log(eigvals))
    return float(S)


def compute_entanglement_entropy_sparse(psi: np.ndarray, L: int, A_size: int) -> float:
    """Von Neumann entropy using partial trace contraction.
    Only ever forms rho_A (size 2^A × 2^A), NEVER full rho.

    Note: This local implementation is kept for backward compatibility with existing
    MERA code paths. For new ED (Exact Diagonalization) reference data, the
    `entanglement_utils` module provides a more comprehensive interface with
    additional functionality like entanglement spectrum and gap computation.
    """
    dim_A = 2**A_size
    # psi has shape (2^L,) - reshape to tensor network
    psi_tensor = psi.reshape([2] * L)

    # Compute reduced density matrix by contracting over B sites
    # rho_A[i1,i2,...,j1,j2,...] = sum_{B} psi*[i1,...,B,...] psi[j1,...,B,...]
    psi_conj = psi_tensor.conj()

    # Build rho_A by contraction
    # Use numpy tensordot for contraction
    # Indices: A sites = 0..A_size-1, B sites = A_size..L-1
    psi_Ashape = psi_tensor.reshape(dim_A, -1)
    rho_A = psi_Ashape @ psi_Ashape.T.conj()

    return compute_entanglement_entropy_from_rho_A(rho_A)


def fidelity(psi1: np.ndarray, psi2: np.ndarray) -> float:
    """Compute |⟨ψ₁|ψ₂⟩|²"""
    overlap = np.vdot(psi1, psi2)
    return float(np.abs(overlap) ** 2)


def exact_diagonalization(
    L: int,
    model: str,
    A_size: int,
    j: float = 1.0,
    h: float = 1.0,
    cyclic: bool = False,
) -> EDResult:
    """ED using quimb sparse Hamiltonian + scipy.sparse.eigsh.
    Safe for L=16 (65K Hilbert space) using sparse eigensolver.
    """
    import quimb as qu
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla

    print(f"  [ED] Building SPARSE {model} Hamiltonian for L={L}...")

    if model in ["ising_open", "ising_cyclic"]:
        # Critical: sparse=True prevents densification
        # quimb uses jz, bx not j, h
        cyclic_flag = cyclic or model == "ising_cyclic"
        H = qu.ham_ising(L, jz=j, bx=h, sparse=True, cyclic=cyclic_flag)
    elif model in ["heisenberg_open", "heisenberg_cyclic"]:
        cyclic_flag = cyclic or model == "heisenberg_cyclic"
        H = qu.ham_heis(L, sparse=True, cyclic=cyclic_flag)
    else:
        raise ValueError(f"Unknown model: {model}")

    # ASSERTION: Must be sparse
    assert sp.issparse(H), f"ERROR: Hamiltonian is not sparse! Type: {type(H)}"
    print(
        f"  [ED] Sparse OK: shape {H.shape}, nnz={H.nnz}, sparsity={H.nnz / H.shape[0] ** 2:.4%}"
    )

    # Sparse eigensolver (only need k=1 ground state)
    print("  [ED] Sparse diagonalization (k=1, which='SA')...")
    evals, evecs = sla.eigsh(H, k=1, which="SA")

    E0 = float(evals[0].real)
    psi0 = np.asarray(evecs[:, 0], dtype=np.complex128)
    psi0 /= np.linalg.norm(psi0)  # Normalize

    print(f"  [ED] Ground state energy: {E0:.6f}")

    # Entropy from Schmidt decomposition
    S = compute_entanglement_entropy_sparse(psi0, L, A_size)
    print(f"  [ED] Entanglement entropy S={S:.6f}")

    return EDResult(
        ground_state_energy=E0, ground_state_psi=psi0, entanglement_entropy=S, n_sites=L
    )


# ============================================================
# MERA Optimization (Proper Quimb Pattern)
# ============================================================


def optimize_mera_for_model(
    L: int,
    chi: int,
    steps: int,
    seed: int,
    model: str,
    j: float = 1.0,
    h: float = 1.0,
    ed_psi: Optional[np.ndarray] = None,
) -> Tuple[Any, float]:
    """
    Optimize MERA for given Hamiltonian using local terms.
    Returns (optimized_mera, final_energy).
    """
    import quimb.tensor as qtn
    import quimb as qu

    np.random.seed(seed)

    # Build random MERA
    mera = qtn.MERA.rand(L, max_bond=chi, dtype="float64")

    # Build Hamiltonian terms
    if model in ["ising_open", "ising_cyclic"]:
        # Ising: -J Z_i Z_{i+1} - h X_i
        cyclic = model == "ising_cyclic"
        pairs = [(i, i + 1) for i in range(L - 1)]
        if cyclic:
            pairs.append((L - 1, 0))

        # Build local ZZ term: -J * Z ⊗ Z
        Z = np.array([[1, 0], [0, -1]], dtype=np.float64)
        ZZ = -j * np.kron(Z, Z)

        # Build local X term: -h * X
        X = np.array([[0, 1], [1, 0]], dtype=np.float64)
        Xlocal = -h * X

        terms = {}
        for i, j in pairs:
            terms[(i, j)] = ZZ
        for i in range(L):
            terms[(i,)] = Xlocal

    elif model in ["heisenberg_open", "heisenberg_cyclic"]:
        cyclic = model == "heisenberg_cyclic"
        H2 = qu.ham_heis(2).real

        pairs = [(i, (i + 1) % L) for i in range(L - 1)]
        if cyclic:
            pairs.append((L - 1, 0))

        terms = {pair: H2 for pair in pairs}
    else:
        raise ValueError(f"Unknown model: {model}")

    # Path optimizer
    try:
        import cotengra as ctg

        opt = ctg.ReusableHyperOptimizer(progbar=False, reconf_opts={})
    except Exception:
        opt = "auto-hq"

    # Norm function for MERA
    def norm_fn(m):
        return m.isometrize(method="exp")

    # Local expectation for 2-site terms
    def local_expectation_2site(m, terms, where, optimize="auto-hq"):
        tags = [m.site_tag(coo) for coo in where]
        m_ij = m.select(tags, which="any")
        m_ij_G = m_ij.gate(terms[where], where)
        ex = m_ij_G & m_ij.H
        return ex.contract(all, optimize=optimize)

    # Local expectation for 1-site terms (field)
    def local_expectation_1site(m, terms, where, optimize="auto-hq"):
        (site,) = where
        tag = m.site_tag(site)
        m_i = m.select([tag], which="any")
        m_i_G = m_i.gate(terms[where], [site])
        ex = m_i_G & m_i.H
        return ex.contract(all, optimize=optimize)

    # Loss function
    def loss_fn(m, terms, optimize="auto-hq"):
        total = 0.0
        for where in terms:
            if len(where) == 2:
                total += local_expectation_2site(m, terms, where, optimize=optimize)
            else:
                total += local_expectation_1site(m, terms, where, optimize=optimize)
        return total

    # Create optimizer
    tnopt = qtn.TNOptimizer(
        mera,
        loss_fn=loss_fn,
        norm_fn=norm_fn,
        loss_constants={"terms": terms},
        loss_kwargs={"optimize": opt},
        autodiff_backend="torch",
        device="cpu",
        jit_fn=False,
    )

    # Optimize
    tnopt.optimizer = "l-bfgs-b"
    try:
        mera_opt = tnopt.optimize(steps)
        final_energy = float(loss_fn(mera_opt, terms))
    except Exception as e:
        print(f"    Optimization failed: {e}")
        mera_opt = mera
        final_energy = float(loss_fn(mera, terms))

    return mera_opt, final_energy


def optimize_mera_for_fidelity(
    L: int,
    chi: int,
    ed_psi: Optional[np.ndarray],
    model: str,
    steps: int,
    seed: int,
    j: float = 1.0,
    h: float = 1.0,
) -> OptimizationResult:
    """
    Backwards-compatible API used by P2/P3 runners.

    Historically this function returned an OptimizationResult object with
    fidelity/entropy/final_energy/converged. Keep that contract while reusing
    the v2 optimizer implementation.
    """
    converged = True
    try:
        mera_opt, final_energy = optimize_mera_for_model(
            L=L,
            chi=chi,
            steps=steps,
            seed=seed,
            model=model,
            j=j,
            h=h,
            ed_psi=ed_psi,
        )
    except Exception:
        # Preserve legacy behavior for callers that continue on failed restarts.
        converged = False
        return OptimizationResult(
            chi=chi,
            restart_idx=0,
            fidelity=0.0,
            entropy=0.0,
            final_energy=0.0,
            seed=seed,
            converged=False,
            num_steps=steps,
        )

    # Entropy is always computed from the optimized MERA state.
    entropy = compute_entropy_from_mera(mera_opt, L, list(range(L // 2)))

    # Fidelity is only defined when an ED reference is provided.
    fidelity_value = 0.0
    if ed_psi is not None:
        psi_ref = np.asarray(ed_psi, dtype=np.complex128).reshape(-1)
        psi_ref = psi_ref / np.linalg.norm(psi_ref)
        psi_mera = mera_opt.to_dense().reshape(-1)
        psi_mera = psi_mera / np.linalg.norm(psi_mera)
        fidelity_value = fidelity(psi_mera, psi_ref)

    return OptimizationResult(
        chi=chi,
        restart_idx=0,
        fidelity=float(fidelity_value),
        entropy=float(entropy),
        final_energy=float(final_energy),
        seed=seed,
        converged=converged,
        num_steps=steps,
    )


def compute_entropy_from_mera(mera, L: int, A_sites: List[int]) -> float:
    """
    Compute S(A) from MERA by forming reduced density matrix.
    """

    A_size = len(A_sites)

    # Build bra with different index labels
    bra = mera.H.reindex_sites("b{}", A_sites)

    # Select causal cone
    tags = [mera.site_tag(i) for i in A_sites]
    rho_tn = bra.select(tags, which="any") & mera.select(tags, which="any")

    left_inds = tuple(f"k{i}" for i in A_sites)
    right_inds = tuple(f"b{i}" for i in A_sites)

    # Dense rho_A
    rho = rho_tn.to_dense(left_inds, right_inds)
    rho = np.array(rho, dtype=np.complex128)
    rho = 0.5 * (rho + rho.conj().T)

    # Normalize
    tr = float(np.real_if_close(np.trace(rho)))
    if tr > 0:
        rho = rho / tr

    # Eigenvalues
    evals = np.linalg.eigvalsh(rho)
    evals = np.real_if_close(evals).astype(float)
    evals = np.clip(evals, 0.0, 1.0)

    # Von Neumann entropy
    eps = 1e-15
    S = float(-np.sum(evals * np.log(evals + eps)))

    return S


def run_mera_with_restarts(
    L: int,
    chi: int,
    ed_psi: np.ndarray,
    ed_result: EDResult,
    model: str,
    steps: int,
    restarts: int,
    seed_base: int,
    j: float = 1.0,
    h: float = 1.0,
) -> List[OptimizationResult]:
    """Run MERA optimization with multiple restarts."""

    A_sites = list(range(L // 2))  # Contiguous partition

    results = []
    for restart in range(restarts):
        seed = seed_base + restart * 1000 + chi * 10000
        print(f"    [MERA] chi={chi}, restart={restart + 1}/{restarts}, seed={seed}")

        try:
            mera_opt, energy = optimize_mera_for_model(
                L=L,
                chi=chi,
                steps=steps,
                seed=seed,
                model=model,
                j=j,
                h=h,
                ed_psi=ed_psi,
            )

            # Compute entropy
            entropy = compute_entropy_from_mera(mera_opt, L, A_sites)

            # Compute fidelity to ED ground state
            psi_mera = mera_opt.to_dense().reshape(-1)
            psi_mera = psi_mera / np.linalg.norm(psi_mera)
            fid = fidelity(psi_mera, ed_psi)

            print(f"      fidelity={fid:.6f}, S={entropy:.4f}, E={energy:.4f}")

            results.append(
                OptimizationResult(
                    chi=chi,
                    restart_idx=restart,
                    fidelity=fid,
                    entropy=entropy,
                    final_energy=energy,
                    seed=seed,
                )
            )

        except Exception as e:
            print(f"      ERROR: {e}")
            results.append(
                OptimizationResult(
                    chi=chi,
                    restart_idx=restart,
                    fidelity=0.0,
                    entropy=0.0,
                    final_energy=0.0,
                    seed=seed,
                )
            )

    return results


# ============================================================
# Falsifiers (P3.1-P3.4)
# ============================================================


def falsifier_p31_nondecreasing_fidelity(
    best_fidelities: List[float], chi_values: List[int], eps_fid: float = 1e-4
) -> Dict[str, Any]:
    """P3.1: Best fidelity is nondecreasing in chi."""
    violations = []
    for i in range(len(best_fidelities) - 1):
        if best_fidelities[i + 1] < best_fidelities[i] - eps_fid:
            violations.append(
                {
                    "chi_i": chi_values[i],
                    "chi_i+1": chi_values[i + 1],
                    "fid_i": best_fidelities[i],
                    "fid_i+1": best_fidelities[i + 1],
                    "drop": best_fidelities[i] - best_fidelities[i + 1],
                }
            )

    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "n_checks": len(best_fidelities) - 1,
        "eps_fid": eps_fid,
    }


def falsifier_p32_entropy_convergence(
    S_best: List[float], S_ref: float, chi_values: List[int], eps_S: float = 1e-3
) -> Dict[str, Any]:
    """P3.2: Entropy error |S_best(chi) - S_ref| is nonincreasing."""
    errors = [abs(s - S_ref) for s in S_best]
    violations = []

    for i in range(len(errors) - 1):
        if errors[i + 1] > errors[i] + eps_S:
            violations.append(
                {
                    "chi_i": chi_values[i],
                    "chi_i+1": chi_values[i + 1],
                    "error_i": errors[i],
                    "error_i+1": errors[i + 1],
                }
            )

    return {
        "passed": len(violations) == 0,
        "errors": errors,
        "violations": violations,
        "n_checks": len(errors) - 1,
        "eps_S": eps_S,
    }


def falsifier_p33_final_thresholds(
    best_fidelity: float, final_entropy_error: float, L: int, model: str
) -> Dict[str, Any]:
    """P3.3: Final thresholds based on system size."""
    fid_threshold = 0.90 if L >= 16 else 0.95
    S_threshold = 0.15

    fid_passed = best_fidelity >= fid_threshold
    S_passed = final_entropy_error <= S_threshold

    return {
        "passed": fid_passed and S_passed,
        "fid_passed": fid_passed,
        "S_passed": S_passed,
        "best_fidelity": best_fidelity,
        "final_entropy_error": final_entropy_error,
        "fid_threshold": fid_threshold,
        "S_threshold": S_threshold,
    }


def fit_saturating_model(
    chi_values: List[int], S_values: List[float], S_ref: float
) -> Tuple[float, float, float]:
    """Fit S(chi) = S_inf * chi / (chi + c)"""
    chi_arr = np.array(chi_values, dtype=float)
    S_arr = np.array(S_values, dtype=float)

    best_rss = float("inf")
    best_params = (S_ref, 1.0)

    for S_inf in np.linspace(S_ref * 0.8, S_ref * 1.2, 30):
        for c in np.linspace(0.1, max(chi_arr) * 3, 30):
            S_pred = S_inf * chi_arr / (chi_arr + c)
            rss = np.sum((S_arr - S_pred) ** 2)
            if rss < best_rss:
                best_rss = rss
                best_params = (S_inf, c)

    return best_params[0], best_params[1], best_rss


def fit_log_linear_model(
    chi_values: List[int], S_values: List[float]
) -> Tuple[float, float, float]:
    """Fit S(chi) = a * log(chi) + b"""
    chi_arr = np.array(chi_values, dtype=float)
    S_arr = np.array(S_values, dtype=float)

    log_chi = np.log(chi_arr)
    X = np.column_stack([log_chi, np.ones(len(log_chi))])
    coef, _, _, _ = np.linalg.lstsq(X, S_arr, rcond=None)
    a, b = coef[0], coef[1]

    S_pred = a * log_chi + b
    rss = np.sum((S_arr - S_pred) ** 2)

    return a, b, rss


def aic_bic_from_rss(rss: float, n: int, k: int) -> Tuple[float, float]:
    rss = max(float(rss), 1e-12)
    aic = n * math.log(rss / n) + 2 * k
    bic = n * math.log(rss / n) + k * math.log(n)
    return float(aic), float(bic)


def falsifier_p34_model_selection(
    chi_values: List[int], S_best: List[float], S_ref: float
) -> Dict[str, Any]:
    """P3.4: Saturating model beats log-linear (ΔAIC/BIC >= 10)"""
    # Saturating fit
    S_inf_sat, c_sat, rss_sat = fit_saturating_model(chi_values, S_best, S_ref)
    aic_sat, bic_sat = aic_bic_from_rss(rss_sat, len(chi_values), 2)

    # Log-linear fit
    a_log, b_log, rss_log = fit_log_linear_model(chi_values, S_best)
    aic_log, bic_log = aic_bic_from_rss(rss_log, len(chi_values), 2)

    delta_aic = aic_log - aic_sat
    delta_bic = bic_log - bic_sat

    saturating_wins = delta_aic >= 10 and delta_bic >= 10

    return {
        "passed": saturating_wins,
        "delta_aic": delta_aic,
        "delta_bic": delta_bic,
        "S_inf_sat": S_inf_sat,
        "c_sat": c_sat,
        "aic_sat": aic_sat,
        "bic_sat": bic_sat,
        "a_log": a_log,
        "b_log": b_log,
        "aic_log": aic_log,
        "bic_log": bic_log,
    }


# ============================================================
# Main Runner
# ============================================================


def make_run_id() -> str:
    t = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    r = os.urandom(4).hex()
    return f"{t}_{r}"


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser(description="Claim 3P Physical Convergence v2")
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--A_size", type=int, required=True)
    ap.add_argument(
        "--model",
        choices=["ising_open", "heisenberg_open", "ising_cyclic", "heisenberg_cyclic"],
        required=True,
    )
    ap.add_argument("--j", type=float, default=1.0)
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--chi_sweep", default="2,4,8,16")
    ap.add_argument("--restarts_per_chi", type=int, default=3)
    ap.add_argument("--fit_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--eps_fid", type=float, default=1e-4)
    ap.add_argument("--eps_S", type=float, default=1e-3)

    args = ap.parse_args()

    chi_sweep = [int(x.strip()) for x in args.chi_sweep.split(",") if x.strip()]

    config = Config(
        L=args.L,
        A_size=args.A_size,
        model=args.model,
        chi_sweep=chi_sweep,
        restarts_per_chi=args.restarts_per_chi,
        fit_steps=args.fit_steps,
        seed=args.seed,
        output_dir=Path(args.output),
        j=args.j,
        h=args.h,
        eps_fid=args.eps_fid,
        eps_S=args.eps_S,
    )

    run_id = make_run_id()
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CLAIM 3P: Physical Hamiltonian Convergence Test (v2)")
    print("=" * 70)
    print(f"Model: {config.model}")
    print(f"System: L={config.L}, A={config.A_size}")
    print(f"Chi sweep: {config.chi_sweep}")
    print(f"Restarts: {config.restarts_per_chi}")
    print(f"Steps: {config.fit_steps}")
    print(f"Output: {run_dir}")
    print("=" * 70)

    # Phase 1: ED
    print("\n[Phase 1] Exact Diagonalization")
    ed_result = exact_diagonalization(
        L=config.L, model=config.model, A_size=config.A_size, j=config.j, h=config.h
    )

    # Compute entanglement entropy using entanglement_utils
    psi_ed = ed_result.ground_state_psi
    subsystem_A = list(range(config.A_size if hasattr(config, 'A_size') and config.A_size else config.L // 2))
    ed_entanglement = {}
    try:
        rho_A = reduced_density_matrix(psi_ed, subsystem_A, config.L)
        ed_entanglement["entanglement_entropy"] = von_neumann_entropy(rho_A)
        ed_entanglement["entanglement_spectrum"] = entanglement_spectrum(rho_A).tolist()
        ed_entanglement["entanglement_gap"] = entanglement_gap(rho_A)
    except Exception as e:
        print(f"  [ED] Warning: entanglement computation failed: {e}")
        ed_entanglement["entanglement_entropy"] = None
        ed_entanglement["entanglement_spectrum"] = None
        ed_entanglement["entanglement_gap"] = None
        ed_entanglement["entanglement_error"] = str(e)

    # Phase 2: MERA optimization
    print("\n[Phase 2] MERA Variational Optimization")

    all_results = []
    best_results = []

    for chi in config.chi_sweep:
        print(f"\n  Chi = {chi}")
        results = run_mera_with_restarts(
            L=config.L,
            chi=chi,
            ed_psi=ed_result.ground_state_psi,
            ed_result=ed_result,
            model=config.model,
            steps=config.fit_steps,
            restarts=config.restarts_per_chi,
            seed_base=config.seed,
            j=config.j,
            h=config.h,
        )

        best = max(results, key=lambda r: r.fidelity)
        best_results.append(best)
        all_results.extend(results)
        print(f"  Best: fidelity={best.fidelity:.6f}, S={best.entropy:.4f}")

    # Extract for falsifiers
    best_fids = [r.fidelity for r in best_results]
    best_S = [r.entropy for r in best_results]

    # Phase 3: Acceptance criteria
    print("\n[Phase 3] Acceptance Criteria")

    p31 = falsifier_p31_nondecreasing_fidelity(
        best_fids, config.chi_sweep, config.eps_fid
    )
    print(f"  P3.1 Fidelity nondecreasing: {'PASS' if p31['passed'] else 'FAIL'}")

    p32 = falsifier_p32_entropy_convergence(
        best_S, ed_result.entanglement_entropy, config.chi_sweep, config.eps_S
    )
    print(f"  P3.2 Entropy convergence: {'PASS' if p32['passed'] else 'FAIL'}")

    final_fid = best_fids[-1]
    final_S_err = abs(best_S[-1] - ed_result.entanglement_entropy)
    p33 = falsifier_p33_final_thresholds(final_fid, final_S_err, config.L, config.model)
    print(f"  P3.3 Final thresholds: {'PASS' if p33['passed'] else 'FAIL'}")
    print(f"        Fidelity: {final_fid:.4f} (threshold: {p33['fid_threshold']})")
    print(f"        S error: {final_S_err:.4f} (threshold: {p33['S_threshold']})")

    p34 = falsifier_p34_model_selection(
        config.chi_sweep, best_S, ed_result.entanglement_entropy
    )
    print(f"  P3.4 Model selection: {'PASS' if p34['passed'] else 'FAIL'}")
    print(f"        Delta AIC: {p34['delta_aic']:.2f}")
    print(f"        Delta BIC: {p34['delta_bic']:.2f}")

    # Final verdict
    all_passed = all([p31["passed"], p32["passed"], p33["passed"], p34["passed"]])
    verdict = "SUPPORTED" if all_passed else "REJECTED"

    print(f"\n{'=' * 70}")
    print(f"FINAL VERDICT: {verdict}")
    print(f"{'=' * 70}")

    # Save artifacts
    print(f"\n[Phase 4] Saving artifacts to {run_dir}")

    # Metrics
    write_json(
        run_dir / "metrics.json",
        {
            "chi_sweep": config.chi_sweep,
            "ed_reference": {
                "energy": float(ed_result.ground_state_energy),
                "entropy": float(ed_result.entanglement_entropy),
                "entanglement": ed_entanglement,
            },
            "best_per_chi": [
                {"chi": r.chi, "fidelity": r.fidelity, "entropy": r.entropy}
                for r in best_results
            ],
        },
    )

    # Verdict
    write_json(
        run_dir / "verdict.json",
        {
            "claim": "3P",
            "model": config.model,
            "L": config.L,
            "verdict": verdict,
            "falsifiers": {"P3.1": p31, "P3.2": p32, "P3.3": p33, "P3.4": p34},
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )

    # Manifest
    write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "config": {
                "L": config.L,
                "A_size": config.A_size,
                "model": config.model,
                "chi_sweep": config.chi_sweep,
                "restarts": config.restarts_per_chi,
                "steps": config.fit_steps,
                "seed": config.seed,
            },
            "ed_reference": {
                "energy": float(ed_result.ground_state_energy),
                "entropy": float(ed_result.entanglement_entropy),
            },
            "verdict": verdict,
        },
    )

    # CSV
    with open(run_dir / "raw_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chi", "restart", "seed", "fidelity", "entropy", "energy"])
        for r in all_results:
            writer.writerow(
                [r.chi, r.restart_idx, r.seed, r.fidelity, r.entropy, r.final_energy]
            )

    print("  Saved: metrics.json, verdict.json, manifest.json, raw_results.csv")

    return 0 if verdict == "SUPPORTED" else 1


if __name__ == "__main__":
    sys.exit(main())
