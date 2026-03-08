#!/usr/bin/env python3
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


@dataclass
class EDResult:
    ground_state_energy: float
    ground_state_psi: np.ndarray
    entanglement_entropy: float
    entanglement_spectrum: Optional[np.ndarray]
    entanglement_gap: Optional[float]
    n_sites: int


@dataclass
class MERAOptimizationResult:
    entropy: float
    fidelity: float
    final_energy: float
    converged: bool
    mera_state: Any | None = None
    error_message: Optional[str] = None


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


def exact_diagonalization(
    L: int,
    model: str,
    A_size: int,
    j: float = 1.0,
    h: float = 1.0,
    cyclic: bool = False,
) -> EDResult:
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
    )


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
            # quimb docs: MERA.select(i) for one-site lightcone
            local_tn = m.select(sites[0])
            gated = local_tn.gate(operator, sites[0])
        else:
            # quimb docs: MERA.select((mera.site_tag(i), mera.site_tag(j)), which="any")
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

    bra = mera.H.reindex_sites("b{}", A_sites)
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