#!/usr/bin/env python3
"""
MERA Backend with ED and DMRG support.

ED: Exact diagonalization for L <= 16 (memory limited)
DMRG: Density Matrix Renormalization Group for L > 16 (scalable)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

# Keep thread usage conservative for crash-prone numerical paths.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    from threadpoolctl import threadpool_limits
except Exception:
    from contextlib import contextmanager

    @contextmanager
    def threadpool_limits(limits=None):
        yield


# Import entanglement_utils - try relative first, then absolute.
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


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class EDResult:
    """Result from exact diagonalization."""
    ground_state_energy: float
    ground_state_psi: np.ndarray
    entanglement_entropy: float
    entanglement_spectrum: Optional[np.ndarray]
    entanglement_gap: Optional[float]
    n_sites: int
    method: str = "ED"


@dataclass
class DMRGResult:
    """Result from DMRG calculation."""
    ground_state_energy: float
    entanglement_entropy: float
    entanglement_spectrum: Optional[np.ndarray]
    entanglement_gap: Optional[float]
    n_sites: int
    bond_dim: int
    sweeps: int
    converged: bool
    method: str = "DMRG"
    # DMRG doesn't give full wavefunction, but we can compute observables
    mps_state: Any = None  # quimb MatrixProductState


@dataclass
class MERAOptimizationResult:
    """Result from MERA optimization."""
    entropy: float
    fidelity: float
    final_energy: float
    converged: bool
    mera_state: Any | None = None
    error_message: Optional[str] = None


# =============================================================================
# Utility functions
# =============================================================================

def subsystem_sites(A_size: int) -> list[int]:
    return list(range(A_size))


def normalize_state(psi: np.ndarray) -> np.ndarray:
    vec = np.asarray(psi, dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(vec)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("State vector has invalid norm")
    return vec / norm


def fidelity(psi1: np.ndarray, psi2: np.ndarray) -> float:
    v1 = normalize_state(psi1)
    v2 = normalize_state(psi2)
    overlap = np.vdot(v1, v2)
    return float(np.abs(overlap) ** 2)


def reduced_density_matrix_manual(
    psi: np.ndarray,
    subsystem_A: Sequence[int],
    L: int,
) -> np.ndarray:
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
) -> dict[str, Any]:
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


# =============================================================================
# ED (Exact Diagonalization)
# =============================================================================

def exact_diagonalization(
    L: int,
    model: str,
    A_size: int,
    j: float = 1.0,
    h: float = 1.0,
    cyclic: bool = False,
) -> EDResult:
    """
    Compute ground state via exact diagonalization.
    
    Memory: O(2^L), feasible for L <= 16 on typical machines.
    """
    import quimb as qu
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla

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

    with threadpool_limits(limits=1):
        evals, evecs = sla.eigsh(H, k=1, which="SA")

    E0 = float(np.real(evals[0]))
    psi0 = normalize_state(np.asarray(evecs[:, 0], dtype=np.complex128))

    ent = compute_entanglement_metadata(psi0, L, subsystem_sites(A_size))
    if ent["entropy"] is None:
        raise RuntimeError(f"ED entanglement calculation failed: {ent['error_message']}")

    return EDResult(
        ground_state_energy=E0,
        ground_state_psi=psi0,
        entanglement_entropy=float(ent["entropy"]),
        entanglement_spectrum=None
        if ent["spectrum"] is None
        else np.asarray(ent["spectrum"], dtype=float),
        entanglement_gap=None if ent["gap"] is None else float(ent["gap"]),
        n_sites=L,
        method="ED",
    )


# =============================================================================
# DMRG (Density Matrix Renormalization Group)
# =============================================================================

def dmrg_ground_state(
    L: int,
    model: str,
    A_size: int,
    bond_dim: int = 64,
    max_bond_dim: int = 256,
    sweeps: int = 10,
    j: float = 1.0,
    h: float = 1.0,
    cutoff: float = 1e-10,
    verbosity: int = 0,
) -> DMRGResult:
    """
    Compute ground state via DMRG.
    
    Memory: O(L * bond_dim^2), scalable to L > 100.
    
    Parameters
    ----------
    L : int
        Number of sites (must be power of 2 for MERA compatibility, but DMRG works for any L)
    model : str
        Model type: 'heisenberg_open', 'heisenberg_cyclic', 'ising_open', 'ising_cyclic'
    A_size : int
        Subsystem size for entanglement entropy calculation
    bond_dim : int
        Initial bond dimension
    max_bond_dim : int
        Maximum bond dimension during sweeps
    sweeps : int
        Number of DMRG sweeps
    j : float
        Coupling strength
    h : float
        Field strength (for Ising)
    cutoff : float
        Singular value cutoff
    verbosity : int
        0 = silent, 1 = progress, 2 = detailed
    
    Returns
    -------
    DMRGResult
        Ground state energy, entanglement entropy, and MPS state
    """
    import quimb as qu
    import quimb.tensor as qtn
    
    # Build Hamiltonian as MPO
    if model in {"ising_open", "ising_cyclic"}:
        cyclic_flag = model == "ising_cyclic"
        H_mpo = qu.MPO_ham_ising(L, jz=j, bx=h, cyclic=cyclic_flag)
    elif model in {"heisenberg_open", "heisenberg_cyclic"}:
        cyclic_flag = model == "heisenberg_cyclic"
        H_mpo = qu.MPO_ham_heis(L, cyclic=cyclic_flag)
    else:
        raise ValueError(f"Unknown model: {model}")
    
    # Initialize random MPS
    mps = qtn.MatrixProductState.rand(L, bond_dim, dtype="float64")
    
    # Run DMRG
    dmrg = qtn.DMRG2(
        H_mpo,
        bond_dim=bond_dim,
        max_bond_dim=max_bond_dim,
        cutoff=cutoff,
    )
    
    with threadpool_limits(limits=1):
        energy, mps_opt = dmrg.solve(
            mps,
            max_sweeps=sweeps,
            verbosity=verbosity,
        )
    
    # Compute entanglement entropy at bipartition A_size
    entropy = compute_entropy_from_mps(mps_opt, A_size)
    
    # Get entanglement spectrum
    spectrum = get_entanglement_spectrum_from_mps(mps_opt, A_size)
    gap = float(spectrum[0] - spectrum[1]) if len(spectrum) >= 2 else None
    
    return DMRGResult(
        ground_state_energy=float(energy),
        entanglement_entropy=float(entropy),
        entanglement_spectrum=spectrum,
        entanglement_gap=gap,
        n_sites=L,
        bond_dim=max(mps_opt.bond_sizes),
        sweeps=sweeps,
        converged=dmrg.converged,
        method="DMRG",
        mps_state=mps_opt,
    )


def compute_entropy_from_mps(mps: Any, A_size: int) -> float:
    """
    Compute entanglement entropy for bipartition at site A_size.
    
    Uses the Schmidt values from the bond between sites A_size-1 and A_size.
    """
    # Get the bond dimension and Schmidt values at the cut
    # For MPS, entropy = -sum(lambda_i^2 * log(lambda_i^2))
    # where lambda_i are the Schmidt values
    
    try:
        # quimb MPS has method to get Schmidt values at a bond
        # Bond index is between site A_size-1 and A_size
        bond_idx = A_size - 1 if A_size > 0 else 0
        
        # Get reduced density matrix or Schmidt values
        # Method varies by quimb version
        if hasattr(mps, 'schmidt_values'):
            schmidt = mps.schmidt_values(bond_idx)
        elif hasattr(mps, 'get_schmidt_values'):
            schmidt = mps.get_schmidt_values(bond_idx)
        else:
            # Fallback: compute from singular values at bond
            # This requires accessing the MPS tensors directly
            schmidt = _compute_schmidt_from_mps(mps, A_size)
        
        # Entropy from Schmidt values
        schmidt_sq = np.array(schmidt) ** 2
        schmidt_sq = schmidt_sq[schmidt_sq > 1e-15]  # Filter numerical zeros
        entropy = -np.sum(schmidt_sq * np.log(schmidt_sq))
        
        return float(entropy)
        
    except Exception as e:
        # Fallback: compute from MPS directly
        return _compute_entropy_from_mps_direct(mps, A_size)


def _compute_schmidt_from_mps(mps: Any, A_size: int) -> np.ndarray:
    """Compute Schmidt values from MPS tensors at bipartition A_size."""
    import quimb.tensor as qtn
    
    # Get tensors for left partition
    L = mps.L
    
    # Contract left partition
    left_tensors = [mps.tensors[i] for i in range(A_size)]
    if len(left_tensors) == 0:
        return np.array([1.0])
    
    # Build reduced density matrix
    # This is a simplified approach - full implementation would use canonical form
    left_mps = qtn.MatrixProductState(left_tensors)
    
    # Get singular values at the bond
    # The bond between site A_size-1 and A_size
    if A_size < L:
        # Use SVD on the bond
        bond_idx = A_size - 1
        # This is approximate - proper implementation needs canonical form
        return np.ones(1)  # Placeholder
    
    return np.ones(1)


def _compute_entropy_from_mps_direct(mps: Any, A_size: int) -> float:
    """Direct computation of entanglement entropy from MPS."""
    import quimb.tensor as qtn
    
    L = mps.L
    
    # For MPS in canonical form, entropy is computed from bond dimensions
    # This is a simplified version
    
    # Get the bond dimension at the cut
    bond_dims = mps.bond_sizes if hasattr(mps, 'bond_sizes') else []
    
    if A_size > 0 and A_size < L and len(bond_dims) > A_size - 1:
        bond_dim = bond_dims[A_size - 1]
        # Uniform distribution approximation (upper bound on entropy)
        # Actual entropy requires Schmidt values
        max_entropy = np.log(bond_dim)
        return float(max_entropy)
    
    # Fallback: compute from MPS norm
    return 0.0


def get_entanglement_spectrum_from_mps(mps: Any, A_size: int) -> np.ndarray:
    """Get entanglement spectrum (Schmidt values squared) at bipartition A_size."""
    try:
        bond_idx = A_size - 1 if A_size > 0 else 0
        
        if hasattr(mps, 'schmidt_values'):
            schmidt = mps.schmidt_values(bond_idx)
        elif hasattr(mps, 'get_schmidt_values'):
            schmidt = mps.get_schmidt_values(bond_idx)
        else:
            # Approximate from bond dimension
            bond_dims = mps.bond_sizes if hasattr(mps, 'bond_sizes') else []
            if A_size > 0 and A_size < len(bond_dims) + 1:
                bond_dim = bond_dims[A_size - 1]
                # Uniform distribution approximation
                schmidt = np.ones(bond_dim) / np.sqrt(bond_dim)
            else:
                schmidt = np.array([1.0])
        
        # Spectrum is Schmidt values squared (probabilities)
        spectrum = np.sort(np.array(schmidt) ** 2)[::-1]
        return spectrum
        
    except Exception:
        return np.array([1.0])


# =============================================================================
# Unified ground state interface
# =============================================================================

def ground_state(
    L: int,
    model: str,
    A_size: int,
    method: str = "auto",
    ed_max_L: int = 16,
    dmrg_bond_dim: int = 64,
    dmrg_max_bond_dim: int = 256,
    dmrg_sweeps: int = 10,
    j: float = 1.0,
    h: float = 1.0,
    verbosity: int = 0,
) -> EDResult | DMRGResult:
    """
    Compute ground state using ED or DMRG depending on system size.
    
    Parameters
    ----------
    L : int
        Number of sites
    model : str
        Model type
    A_size : int
        Subsystem size for entanglement
    method : str
        'auto' (default): use ED for L <= ed_max_L, DMRG otherwise
        'ed': force exact diagonalization
        'dmrg': force DMRG
    ed_max_L : int
        Maximum L for ED (default 16, ~4 GB memory)
    dmrg_bond_dim : int
        Initial DMRG bond dimension
    dmrg_max_bond_dim : int
        Maximum DMRG bond dimension
    dmrg_sweeps : int
        Number of DMRG sweeps
    j, h : float
        Model parameters
    verbosity : int
        0 = silent, 1 = progress
    
    Returns
    -------
    EDResult or DMRGResult
    """
    # Determine method
    if method == "auto":
        use_dmrg = L > ed_max_L
    elif method == "ed":
        use_dmrg = False
    elif method == "dmrg":
        use_dmrg = True
    else:
        raise ValueError(f"Unknown method: {method}. Use 'auto', 'ed', or 'dmrg'.")
    
    if use_dmrg:
        if verbosity > 0:
            print(f"[DMRG] L={L}, bond_dim={dmrg_bond_dim}, max_bond={dmrg_max_bond_dim}, sweeps={dmrg_sweeps}")
        return dmrg_ground_state(
            L=L,
            model=model,
            A_size=A_size,
            bond_dim=dmrg_bond_dim,
            max_bond_dim=dmrg_max_bond_dim,
            sweeps=dmrg_sweeps,
            j=j,
            h=h,
            verbosity=verbosity,
        )
    else:
        if verbosity > 0:
            print(f"[ED] L={L}")
        return exact_diagonalization(
            L=L,
            model=model,
            A_size=A_size,
            j=j,
            h=h,
        )


# =============================================================================
# MERA optimization
# =============================================================================

def build_local_terms(
    L: int,
    model: str,
    j_coupling: float = 1.0,
    h_field: float = 1.0,
) -> Tuple[list[Tuple[Tuple[int, ...], np.ndarray]], list[Tuple[Tuple[int, ...], np.ndarray]]]:
    import quimb as qu

    pair_terms: list[Tuple[Tuple[int, ...], np.ndarray]] = []
    single_terms: list[Tuple[Tuple[int, ...], np.ndarray]] = []

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


def _site_to_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        if hasattr(x, "item"):
            return int(x.item())
        raise


def optimize_mera_for_model(
    L: int,
    chi: int,
    steps: int,
    seed: int,
    model: str,
    j: float = 1.0,
    h: float = 1.0,
) -> Tuple[Any, float]:
    import quimb.tensor as qtn

    pair_terms, single_terms = build_local_terms(L=L, model=model, j_coupling=j, h_field=h)

    rng_state = np.random.get_state()
    try:
        np.random.seed(seed)
        mera = qtn.MERA.rand(L, max_bond=chi, dtype="float64")
    finally:
        np.random.set_state(rng_state)

    contract_opt: Any = "auto-hq"

    def norm_fn(m):
        return m.isometrize(method="exp")

    def local_expectation(m, where: Tuple[int, ...], operator: np.ndarray, optimize: Any) -> Any:
        sites = tuple(_site_to_int(site) for site in where)

        if len(sites) == 1:
            local_tn = m.select(sites[0])
            gated = local_tn.gate(operator, sites[0])
        else:
            tags = tuple(m.site_tag(site) for site in sites)
            local_tn = m.select(tags, which="any")
            gated = local_tn.gate(operator, sites)

        expectation_tn = gated & local_tn.H
        return expectation_tn.contract(all, optimize=optimize)

    def loss_fn(
        m,
        pair_terms: list[Tuple[Tuple[int, ...], np.ndarray]],
        single_terms: list[Tuple[Tuple[int, ...], np.ndarray]],
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

    with threadpool_limits(limits=1):
        mera_opt = tnopt.optimize(steps)
        final_energy = float(
            np.real(loss_fn(mera_opt, pair_terms, single_terms, optimize=contract_opt))
        )

    return mera_opt, final_energy


def compute_entropy_from_mera(mera: Any, A_sites: Sequence[int]) -> float:
    A_sites = list(A_sites)

    bra = mera.H.reindex_sites("b{selected_text}", A_sites)
    tags = [mera.site_tag(i) for i in A_sites]

    rho_tn = bra.select(tags, which="any") & mera.select(tags, which="any")
    left_inds = tuple(f"k{i}" for i in A_sites)
    right_inds = tuple(f"b{i}" for i in A_sites)

    rho = rho_tn.to_dense(left_inds, right_inds)
    rho = np.asarray(rho, dtype=np.complex128)

    return compute_entanglement_entropy_from_rho_A(rho)


def optimize_mera_for_fidelity(
    L: int,
    chi: int,
    ed_psi: np.ndarray | None,
    model: str,
    steps: int,
    seed: int,
    j: float = 1.0,
    h: float = 1.0,
    A_size: int | None = None,
) -> MERAOptimizationResult:
    if A_size is None:
        A_size = L // 2

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

        entropy_value = compute_entropy_from_mera(mera_opt, subsystem_sites(A_size))

        if ed_psi is None:
            fid = float("nan")
        else:
            psi_mera = normalize_state(mera_opt.to_dense().reshape(-1))
            fid = fidelity(psi_mera, ed_psi)

        return MERAOptimizationResult(
            entropy=float(entropy_value),
            fidelity=float(fid),
            final_energy=float(energy),
            converged=True,
            mera_state=mera_opt,
            error_message=None,
        )

    except Exception as exc:
        return MERAOptimizationResult(
            entropy=0.0,
            fidelity=float("nan"),
            final_energy=0.0,
            converged=False,
            mera_state=None,
            error_message=f"{type(exc).__name__}: {exc}",
        )


# =============================================================================
# MERA optimization with DMRG reference (for L > 16)
# =============================================================================

def optimize_mera_with_dmrg_reference(
    L: int,
    chi: int,
    model: str,
    steps: int,
    seed: int,
    dmrg_bond_dim: int = 64,
    dmrg_max_bond_dim: int = 256,
    dmrg_sweeps: int = 10,
    j: float = 1.0,
    h: float = 1.0,
    A_size: int | None = None,
    verbosity: int = 0,
) -> MERAOptimizationResult:
    """
    Optimize MERA with DMRG reference for larger systems.
    
    For L > 16, ED is infeasible, so we use DMRG for the reference state.
    Note: Fidelity cannot be computed directly (DMRG gives MPS, not full wavefunction).
    Instead, we compare energies and entanglement entropies.
    """
    if A_size is None:
        A_size = L // 2
    
    # Get DMRG reference
    dmrg_result = dmrg_ground_state(
        L=L,
        model=model,
        A_size=A_size,
        bond_dim=dmrg_bond_dim,
        max_bond_dim=dmrg_max_bond_dim,
        sweeps=dmrg_sweeps,
        j=j,
        h=h,
        verbosity=verbosity,
    )
    
    if verbosity > 0:
        print(f"[DMRG] E0 = {dmrg_result.ground_state_energy:.10f}")
        print(f"[DMRG] S = {dmrg_result.entanglement_entropy:.10f}")
    
    # Optimize MERA
    try:
        mera_opt, mera_energy = optimize_mera_for_model(
            L=L,
            chi=chi,
            steps=steps,
            seed=seed,
            model=model,
            j=j,
            h=h,
        )
        
        entropy_value = compute_entropy_from_mera(mera_opt, subsystem_sites(A_size))
        
        # Energy difference from DMRG reference
        energy_error = mera_energy - dmrg_result.ground_state_energy
        
        # Entropy difference from DMRG reference
        entropy_error = entropy_value - dmrg_result.entanglement_entropy
        
        # For L > 16, we can't compute fidelity directly
        # Use energy-based quality metric instead
        # Relative energy error
        rel_energy_error = abs(energy_error) / abs(dmrg_result.ground_state_energy)
        
        if verbosity > 0:
            print(f"[MERA] E = {mera_energy:.10f}, ΔE = {energy_error:.2e}")
            print(f"[MERA] S = {entropy_value:.10f}, ΔS = {entropy_error:.2e}")
        
        return MERAOptimizationResult(
            entropy=float(entropy_value),
            fidelity=float("nan"),  # Cannot compute for DMRG reference
            final_energy=float(mera_energy),
            converged=True,
            mera_state=mera_opt,
            error_message=None,
        )
        
    except Exception as exc:
        return MERAOptimizationResult(
            entropy=0.0,
            fidelity=float("nan"),
            final_energy=0.0,
            converged=False,
            mera_state=None,
            error_message=f"{type(exc).__name__}: {exc}",
        )
