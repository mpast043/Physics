#!/usr/bin/env python3
"""
Physical Convergence Runner v3 (data collection only)

Purpose:
- Compute ED reference data for Ising / Heisenberg models
- Run MERA optimization across chi values and restarts
- Collect raw per-restart data with strong reproducibility safeguards
- Save data artifacts only (no falsifiers, verdicts, or acceptance criteria)

Usage:
  python3 physical_convergence_runner.py --L 8 --A_size 4 \
    --model ising_open --j 1.0 --h 1.0 --chi_sweep 2,4,8,16 \
    --restarts_per_chi 3 --fit_steps 100 --seed 42 --output <RUN_DIR>
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import socket
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Import entanglement_utils - try relative first, then absolute.
# If unavailable, fall back to local implementations.
HAVE_ENTANGLEMENT_UTILS = True
try:
    from ..entanglement_utils import (
        von_neumann_entropy,
        reduced_density_matrix,
        entanglement_spectrum,
        entanglement_gap,
    )
except ImportError:
    try:
        from entanglement_utils import (
            von_neumann_entropy,
            reduced_density_matrix,
            entanglement_spectrum,
            entanglement_gap,
        )
    except ImportError:
        HAVE_ENTANGLEMENT_UTILS = False
        von_neumann_entropy = None
        reduced_density_matrix = None
        entanglement_spectrum = None
        entanglement_gap = None


# ============================================================
# Types and Configuration
# ============================================================


@dataclass(frozen=True)
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

    def validate(self) -> None:
        if self.L <= 1:
            raise ValueError("--L must be > 1")
        if self.A_size <= 0 or self.A_size >= self.L:
            raise ValueError("--A_size must satisfy 1 <= A_size < L")
        if not self.chi_sweep:
            raise ValueError("--chi_sweep must contain at least one chi value")
        if any(chi <= 0 for chi in self.chi_sweep):
            raise ValueError("All chi values must be positive")
        if self.restarts_per_chi <= 0:
            raise ValueError("--restarts_per_chi must be > 0")
        if self.fit_steps <= 0:
            raise ValueError("--fit_steps must be > 0")
        if self.model not in {
            "ising_open",
            "ising_cyclic",
            "heisenberg_open",
            "heisenberg_cyclic",
        }:
            raise ValueError(f"Unsupported model: {self.model}")


@dataclass
class EDResult:
    ground_state_energy: float
    ground_state_psi: np.ndarray
    entanglement_entropy: float
    entanglement_spectrum: Optional[np.ndarray]
    entanglement_gap: Optional[float]
    n_sites: int


@dataclass
class OptimizationResult:
    chi: int
    restart_idx: int
    fidelity: float
    entropy: float
    final_energy: float
    seed: int
    converged: bool
    num_steps: int
    elapsed_sec: float
    error_message: Optional[str] = None


# ============================================================
# Utilities
# ============================================================


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_chi_sweep(text: str) -> List[int]:
    values = []
    for raw in text.split(","):
        raw = raw.strip()
        if raw:
            values.append(int(raw))
    return sorted(set(values))


def subsystem_sites(A_size: int) -> List[int]:
    return list(range(A_size))


def normalize_state(psi: np.ndarray) -> np.ndarray:
    vec = np.asarray(psi, dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(vec)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("State vector has invalid norm")
    return vec / norm


def fidelity(psi1: np.ndarray, psi2: np.ndarray) -> float:
    """Compute |<psi1|psi2>|^2."""
    v1 = normalize_state(psi1)
    v2 = normalize_state(psi2)
    overlap = np.vdot(v1, v2)
    return float(np.abs(overlap) ** 2)


def make_run_id() -> str:
    return f"RUN_{utc_now().strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}_{os.urandom(4).hex()}"


def create_unique_run_dir(base_output: Path) -> Tuple[str, Path]:
    base_output.mkdir(parents=True, exist_ok=True)
    for _ in range(16):
        run_id = make_run_id()
        run_dir = base_output / run_id
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
            return run_id, run_dir
        except FileExistsError:
            continue
    raise RuntimeError("Could not create a unique run directory after multiple attempts")


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dt.datetime):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=json_default)
    os.replace(tmp_path, path)


def write_csv_atomic(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(tmp_path, path)


def snapshot_runtime_environment() -> Dict[str, Any]:
    return {
        "timestamp_utc": utc_now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "argv": sys.argv,
        "pid": os.getpid(),
    }


def best_result_for_chi(results: Sequence[OptimizationResult]) -> OptimizationResult:
    if not results:
        raise ValueError("No results available for chi")

    def key(r: OptimizationResult) -> Tuple[int, float, float]:
        # Prefer converged results, then higher fidelity, then lower energy.
        return (
            1 if r.converged else 0,
            float(r.fidelity),
            float(-r.final_energy),
        )

    return max(results, key=key)


# ============================================================
# Entanglement helpers
# ============================================================


def reduced_density_matrix_manual(psi: np.ndarray, subsystem_A: Sequence[int], L: int) -> np.ndarray:
    psi = normalize_state(psi)
    subsystem_A = sorted(set(int(i) for i in subsystem_A))
    if len(subsystem_A) == 0 or len(subsystem_A) >= L:
        raise ValueError("Subsystem A must satisfy 1 <= len(A) < L")
    if any(i < 0 or i >= L for i in subsystem_A):
        raise ValueError("Subsystem A indices out of range")

    subsystem_B = [i for i in range(L) if i not in subsystem_A]
    psi_tensor = psi.reshape([2] * L)
    perm = subsystem_A + subsystem_B
    psi_matrix = np.transpose(psi_tensor, axes=perm).reshape(2 ** len(subsystem_A), -1)
    rho_A = psi_matrix @ psi_matrix.conj().T
    return np.asarray(rho_A, dtype=np.complex128)


def compute_entanglement_entropy_from_rho_A(rho_A: np.ndarray) -> float:
    rho = np.asarray(rho_A, dtype=np.complex128)
    rho = 0.5 * (rho + rho.conj().T)

    tr = float(np.real_if_close(np.trace(rho)))
    if not np.isfinite(tr) or tr <= 0.0:
        raise ValueError("Reduced density matrix has invalid trace")

    rho = rho / tr
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.real_if_close(eigvals).astype(float)
    eigvals = np.clip(eigvals, 0.0, 1.0)
    eigvals = eigvals[eigvals > 1e-12]
    if eigvals.size == 0:
        return 0.0

    return float(-np.sum(eigvals * np.log(eigvals)))


def compute_entanglement_metadata(
    psi: np.ndarray,
    L: int,
    A_sites: Sequence[int],
) -> Dict[str, Any]:
    rho_A = None
    entropy_value = None
    spectrum_value = None
    gap_value = None
    error_message = None

    try:
        if HAVE_ENTANGLEMENT_UTILS:
            rho_A = reduced_density_matrix(psi, list(A_sites), L)
            entropy_value = float(von_neumann_entropy(rho_A))
            spectrum_arr = np.asarray(entanglement_spectrum(rho_A), dtype=float)
            spectrum_value = spectrum_arr
            gap_raw = entanglement_gap(rho_A)
            gap_value = None if gap_raw is None else float(gap_raw)
        else:
            raise RuntimeError("entanglement_utils not available")
    except Exception as exc:
        try:
            rho_A = reduced_density_matrix_manual(psi, A_sites, L)
            entropy_value = compute_entanglement_entropy_from_rho_A(rho_A)
            spectrum_arr = np.linalg.eigvalsh(rho_A)
            spectrum_arr = np.real_if_close(spectrum_arr).astype(float)
            spectrum_arr = np.sort(spectrum_arr)[::-1]
            spectrum_value = spectrum_arr
            gap_value = (
                float(spectrum_arr[0] - spectrum_arr[1])
                if spectrum_arr.size >= 2
                else None
            )
            error_message = f"Used local entanglement fallback: {exc}"
        except Exception as fallback_exc:
            entropy_value = None
            spectrum_value = None
            gap_value = None
            error_message = (
                f"Entanglement computation failed. Primary: {exc}. Fallback: {fallback_exc}"
            )

    return {
        "entropy": entropy_value,
        "spectrum": spectrum_value,
        "gap": gap_value,
        "error_message": error_message,
    }


# ============================================================
# Exact Diagonalization (sparse)
# ============================================================


def exact_diagonalization(
    L: int,
    model: str,
    A_size: int,
    j: float = 1.0,
    h: float = 1.0,
    cyclic: bool = False,
) -> EDResult:
    """
    ED using quimb sparse Hamiltonian + scipy.sparse.linalg.eigsh.
    """
    import quimb as qu
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla

    print(f"  [ED] Building sparse {model} Hamiltonian for L={L}...")

    if model in {"ising_open", "ising_cyclic"}:
        cyclic_flag = cyclic or model == "ising_cyclic"
        H = qu.ham_ising(L, jz=j, bx=h, sparse=True, cyclic=cyclic_flag)
    elif model in {"heisenberg_open", "heisenberg_cyclic"}:
        cyclic_flag = cyclic or model == "heisenberg_cyclic"
        H = qu.ham_heis(L, sparse=True, cyclic=cyclic_flag)
    else:
        raise ValueError(f"Unknown model: {model}")

    if not sp.issparse(H):
        raise TypeError(f"Hamiltonian unexpectedly not sparse: {type(H)}")

    print(
        f"  [ED] Sparse OK: shape={H.shape}, nnz={H.nnz}, "
        f"sparsity={H.nnz / (H.shape[0] ** 2):.6%}"
    )

    print("  [ED] Sparse diagonalization (k=1, which='SA')...")
    evals, evecs = sla.eigsh(H, k=1, which="SA")

    E0 = float(np.real(evals[0]))
    psi0 = normalize_state(np.asarray(evecs[:, 0], dtype=np.complex128))

    ent = compute_entanglement_metadata(psi0, L, subsystem_sites(A_size))
    if ent["entropy"] is None:
        raise RuntimeError(f"ED entanglement calculation failed: {ent['error_message']}")

    print(f"  [ED] Ground state energy: {E0:.10f}")
    print(f"  [ED] Entanglement entropy S={ent['entropy']:.10f}")

    return EDResult(
        ground_state_energy=E0,
        ground_state_psi=psi0,
        entanglement_entropy=float(ent["entropy"]),
        entanglement_spectrum=None
        if ent["spectrum"] is None
        else np.asarray(ent["spectrum"], dtype=float),
        entanglement_gap=None if ent["gap"] is None else float(ent["gap"]),
        n_sites=L,
    )


# ============================================================
# MERA Optimization
# ============================================================


def build_local_terms(
    L: int,
    model: str,
    j_coupling: float = 1.0,
    h_field: float = 1.0,
) -> Tuple[List[Tuple[Tuple[int, ...], np.ndarray]], List[Tuple[Tuple[int, ...], np.ndarray]]]:
    import quimb as qu

    pair_terms: List[Tuple[Tuple[int, ...], np.ndarray]] = []
    single_terms: List[Tuple[Tuple[int, ...], np.ndarray]] = []

    if model in {"ising_open", "ising_cyclic"}:
        cyclic = model == "ising_cyclic"
        neighbor_pairs = [(site, site + 1) for site in range(L - 1)]
        if cyclic:
            neighbor_pairs.append((L - 1, 0))

        Z = np.array([[1, 0], [0, -1]], dtype=np.float64)
        X = np.array([[0, 1], [1, 0]], dtype=np.float64)

        zz_term = -j_coupling * np.kron(Z, Z)
        x_term = -h_field * X

        pair_terms = [((left, right), zz_term) for left, right in neighbor_pairs]
        single_terms = [((site,), x_term) for site in range(L)]

    elif model in {"heisenberg_open", "heisenberg_cyclic"}:
        cyclic = model == "heisenberg_cyclic"
        neighbor_pairs = [(site, site + 1) for site in range(L - 1)]
        if cyclic:
            neighbor_pairs.append((L - 1, 0))

        h2 = np.asarray(qu.ham_heis(2).real, dtype=np.float64)
        pair_terms = [((left, right), h2) for left, right in neighbor_pairs]

    else:
        raise ValueError(f"Unknown model: {model}")

    return pair_terms, single_terms


def optimize_mera_for_model(
    L: int,
    chi: int,
    steps: int,
    seed: int,
    model: str,
    j: float = 1.0,
    h: float = 1.0,
) -> Tuple[Any, float]:
    """
    Optimize MERA for the given Hamiltonian using local terms.
    Returns (optimized_mera, final_energy).
    """
    import quimb.tensor as qtn

    pair_terms, single_terms = build_local_terms(L=L, model=model, j_coupling=j, h_field=h)

    # Avoid polluting global RNG state.
    rng_state = np.random.get_state()
    try:
        np.random.seed(seed)
        mera = qtn.MERA.rand(L, max_bond=chi, dtype="float64")
    finally:
        np.random.set_state(rng_state)

    try:
        import cotengra as ctg

        contract_opt = ctg.ReusableHyperOptimizer(progbar=False, reconf_opts={})
    except Exception:
        contract_opt = "auto-hq"

    site_tags = {site: mera.site_tag(site) for site in range(L)}

    def norm_fn(m):
        return m.isometrize(method="exp")

    def local_expectation(m, where: Tuple[int, ...], operator: np.ndarray, optimize: Any) -> Any:
        tags = [site_tags[site] for site in where]
        local_tn = m.select(tags, which="any")
        gated = local_tn.gate(operator, where)
        expectation_tn = gated & local_tn.H
        return expectation_tn.contract(all, optimize=optimize)

    def loss_fn(
        m,
        pair_terms: List[Tuple[Tuple[int, ...], np.ndarray]],
        single_terms: List[Tuple[Tuple[int, ...], np.ndarray]],
        optimize: Any = "auto-hq",
    ):
        total = 0.0
        for where, op in pair_terms:
            total += local_expectation(m, where, op, optimize=optimize)
        for where, op in single_terms:
            total += local_expectation(m, where, op, optimize=optimize)
        return total

    tnopt = qtn.TNOptimizer(
        mera,
        loss_fn=loss_fn,
        norm_fn=norm_fn,
        loss_constants={
            "pair_terms": pair_terms,
            "single_terms": single_terms,
        },
        loss_kwargs={"optimize": contract_opt},
        autodiff_backend="torch",
        device="cpu",
        jit_fn=False,
    )

    tnopt.optimizer = "l-bfgs-b"
    mera_opt = tnopt.optimize(steps)
    final_energy = float(np.real(loss_fn(mera_opt, pair_terms, single_terms, optimize=contract_opt)))

    return mera_opt, final_energy


def compute_entropy_from_mera(mera: Any, A_sites: Sequence[int]) -> float:
    """
    Compute S(A) from MERA by forming the reduced density matrix.
    """
    A_sites = list(A_sites)

    bra = mera.H.reindex_sites("b{}", A_sites)
    tags = [mera.site_tag(i) for i in A_sites]

    rho_tn = bra.select(tags, which="any") & mera.select(tags, which="any")
    left_inds = tuple(f"k{i}" for i in A_sites)
    right_inds = tuple(f"b{i}" for i in A_sites)

    rho = rho_tn.to_dense(left_inds, right_inds)
    rho = np.asarray(rho, dtype=np.complex128)

    return compute_entanglement_entropy_from_rho_A(rho)


def run_mera_with_restarts(
    L: int,
    A_size: int,
    chi: int,
    ed_psi: np.ndarray,
    model: str,
    steps: int,
    restarts: int,
    seed_base: int,
    j: float = 1.0,
    h: float = 1.0,
) -> List[OptimizationResult]:
    results: List[OptimizationResult] = []
    A_sites = subsystem_sites(A_size)

    for restart in range(restarts):
        seed = seed_base + restart * 1000 + chi * 10000
        print(f"    [MERA] chi={chi}, restart={restart + 1}/{restarts}, seed={seed}")

        t0 = time.perf_counter()
        try:
            mera_opt, energy = optimize_mera_for_model(
                L=L,
                chi=chi,
                steps=steps,
                seed=seed,
                model=model,
                j=j,
                h=h,
            )

            entropy_value = compute_entropy_from_mera(mera_opt, A_sites)
            psi_mera = normalize_state(mera_opt.to_dense().reshape(-1))
            fid = fidelity(psi_mera, ed_psi)

            elapsed = time.perf_counter() - t0
            print(
                f"      converged=True, fidelity={fid:.8f}, "
                f"S={entropy_value:.8f}, E={energy:.8f}, t={elapsed:.2f}s"
            )

            results.append(
                OptimizationResult(
                    chi=chi,
                    restart_idx=restart,
                    fidelity=float(fid),
                    entropy=float(entropy_value),
                    final_energy=float(energy),
                    seed=seed,
                    converged=True,
                    num_steps=steps,
                    elapsed_sec=float(elapsed),
                    error_message=None,
                )
            )

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            err = f"{type(exc).__name__}: {exc}"
            print(f"      converged=False, error={err}, t={elapsed:.2f}s")

            results.append(
                OptimizationResult(
                    chi=chi,
                    restart_idx=restart,
                    fidelity=0.0,
                    entropy=0.0,
                    final_energy=0.0,
                    seed=seed,
                    converged=False,
                    num_steps=steps,
                    elapsed_sec=float(elapsed),
                    error_message=err,
                )
            )

    return results


# ============================================================
# Main Runner
# ============================================================


def build_config_from_args(args: argparse.Namespace) -> Config:
    config = Config(
        L=args.L,
        A_size=args.A_size,
        model=args.model,
        chi_sweep=parse_chi_sweep(args.chi_sweep),
        restarts_per_chi=args.restarts_per_chi,
        fit_steps=args.fit_steps,
        seed=args.seed,
        output_dir=Path(args.output),
        j=args.j,
        h=args.h,
    )
    config.validate()
    return config


def main() -> int:
    ap = argparse.ArgumentParser(description="Physical convergence data runner")
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--A_size", type=int, required=True)
    ap.add_argument(
        "--model",
        choices=["ising_open", "heisenberg_open", "ising_cyclic", "heisenberg_cyclic"],
        required=True,
    )
    ap.add_argument("--j", type=float, default=1.0)
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--chi_sweep", type=str, default="8,16,32,64")
    ap.add_argument("--restarts_per_chi", type=int, default=2)
    ap.add_argument("--fit_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, required=True)

    args = ap.parse_args()
    config = build_config_from_args(args)

    run_id, run_dir = create_unique_run_dir(config.output_dir)
    env = snapshot_runtime_environment()

    print("=" * 72)
    print("PHYSICAL CONVERGENCE DATA RUNNER")
    print("=" * 72)
    print(f"Model:       {config.model}")
    print(f"System:      L={config.L}, A_size={config.A_size}")
    print(f"Chi sweep:   {config.chi_sweep}")
    print(f"Restarts:    {config.restarts_per_chi}")
    print(f"Fit steps:   {config.fit_steps}")
    print(f"Seed:        {config.seed}")
    print(f"Output dir:  {run_dir}")
    print("=" * 72)

    total_t0 = time.perf_counter()

    # Phase 1: ED reference
    print("\n[Phase 1] Exact Diagonalization")
    ed_t0 = time.perf_counter()
    ed_result = exact_diagonalization(
        L=config.L,
        model=config.model,
        A_size=config.A_size,
        j=config.j,
        h=config.h,
    )
    ed_elapsed = time.perf_counter() - ed_t0

    # Phase 2: MERA sweep
    print("\n[Phase 2] MERA Variational Optimization")

    all_results: List[OptimizationResult] = []
    best_results: List[OptimizationResult] = []

    for chi in config.chi_sweep:
        print(f"\n  Chi = {chi}")
        chi_results = run_mera_with_restarts(
            L=config.L,
            A_size=config.A_size,
            chi=chi,
            ed_psi=ed_result.ground_state_psi,
            model=config.model,
            steps=config.fit_steps,
            restarts=config.restarts_per_chi,
            seed_base=config.seed,
            j=config.j,
            h=config.h,
        )

        best = best_result_for_chi(chi_results)
        best_results.append(best)
        all_results.extend(chi_results)

        print(
            f"  Best: converged={best.converged}, "
            f"fidelity={best.fidelity:.8f}, S={best.entropy:.8f}, E={best.final_energy:.8f}"
        )

    total_elapsed = time.perf_counter() - total_t0

    print(f"\n{'=' * 72}")
    print(f"RUN COMPLETE")
    print(f"TOTAL ELAPSED: {total_elapsed:.2f}s")
    print(f"{'=' * 72}")

    # ========================================================
    # Save artifacts (data only)
    # ========================================================

    print(f"\n[Phase 3] Saving artifacts to {run_dir}")

    config_payload = {
        "run_id": run_id,
        "config": {
            "L": config.L,
            "A_size": config.A_size,
            "model": config.model,
            "chi_sweep": config.chi_sweep,
            "restarts_per_chi": config.restarts_per_chi,
            "fit_steps": config.fit_steps,
            "seed": config.seed,
            "output_dir": str(config.output_dir),
            "j": config.j,
            "h": config.h,
        },
    }

    ed_payload = {
        "run_id": run_id,
        "energy": float(ed_result.ground_state_energy),
        "entropy": float(ed_result.entanglement_entropy),
        "entanglement_spectrum": None
        if ed_result.entanglement_spectrum is None
        else ed_result.entanglement_spectrum.tolist(),
        "entanglement_gap": ed_result.entanglement_gap,
        "n_sites": ed_result.n_sites,
        "elapsed_sec": float(ed_elapsed),
    }

    summary_payload = {
        "run_id": run_id,
        "model": config.model,
        "L": config.L,
        "A_size": config.A_size,
        "chi_sweep": config.chi_sweep,
        "ed_energy": float(ed_result.ground_state_energy),
        "ed_entropy": float(ed_result.entanglement_entropy),
        "num_total_restarts": len(all_results),
        "num_failed_restarts": sum(0 if r.converged else 1 for r in all_results),
        "num_successful_restarts": sum(1 if r.converged else 0 for r in all_results),
        "best_per_chi": [
            {
                "chi": r.chi,
                "restart_idx": r.restart_idx,
                "seed": r.seed,
                "converged": r.converged,
                "fidelity": r.fidelity,
                "entropy": r.entropy,
                "final_energy": r.final_energy,
                "elapsed_sec": r.elapsed_sec,
                "error_message": r.error_message,
            }
            for r in best_results
        ],
        "timing": {
            "ed_elapsed_sec": float(ed_elapsed),
            "total_elapsed_sec": float(total_elapsed),
        },
    }

    failures_payload = {
        "run_id": run_id,
        "failures": [
            {
                "chi": r.chi,
                "restart_idx": r.restart_idx,
                "seed": r.seed,
                "num_steps": r.num_steps,
                "elapsed_sec": r.elapsed_sec,
                "error_message": r.error_message,
            }
            for r in all_results
            if not r.converged
        ],
    }

    manifest_payload = {
        "run_id": run_id,
        "environment": env,
        "artifacts": [
            "config.json",
            "ed_reference.json",
            "summary.json",
            "manifest.json",
            "raw_results.csv",
            "best_per_chi.csv",
            "failures.json",
        ],
    }

    write_json_atomic(run_dir / "config.json", config_payload)
    write_json_atomic(run_dir / "ed_reference.json", ed_payload)
    write_json_atomic(run_dir / "summary.json", summary_payload)
    write_json_atomic(run_dir / "failures.json", failures_payload)
    write_json_atomic(run_dir / "manifest.json", manifest_payload)

    raw_rows = [
        [
            r.chi,
            r.restart_idx,
            r.seed,
            int(r.converged),
            r.num_steps,
            r.elapsed_sec,
            r.fidelity,
            r.entropy,
            r.final_energy,
            r.error_message or "",
        ]
        for r in all_results
    ]
    write_csv_atomic(
        run_dir / "raw_results.csv",
        header=[
            "chi",
            "restart_idx",
            "seed",
            "converged",
            "num_steps",
            "elapsed_sec",
            "fidelity",
            "entropy",
            "final_energy",
            "error_message",
        ],
        rows=raw_rows,
    )

    best_rows = [
        [
            r.chi,
            r.restart_idx,
            r.seed,
            int(r.converged),
            r.elapsed_sec,
            r.fidelity,
            r.entropy,
            r.final_energy,
            r.error_message or "",
        ]
        for r in best_results
    ]
    write_csv_atomic(
        run_dir / "best_per_chi.csv",
        header=[
            "chi",
            "best_restart_idx",
            "seed",
            "converged",
            "elapsed_sec",
            "fidelity",
            "entropy",
            "final_energy",
            "error_message",
        ],
        rows=best_rows,
    )

    print(
        "  Saved: config.json, ed_reference.json, summary.json, "
        "manifest.json, raw_results.csv, best_per_chi.csv, failures.json"
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)
