import argparse
import csv
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh

SIERPINSKI_D_S = 2.0 * np.log(3.0) / np.log(5.0)


def make_run_dir(base: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"RUN_{stamp}_{uuid.uuid4().hex[:8]}"
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def sierpinski_gasket_edges(level: int) -> Tuple[int, List[Tuple[int, int]]]:
    if level < 0:
        raise ValueError("level must be >= 0")
    if level == 0:
        return 3, [(0, 1), (0, 2), (1, 2)]

    n_prev, edges_prev = sierpinski_gasket_edges(level - 1)

    def offset_edges(edges: List[Tuple[int, int]], offset: int) -> List[Tuple[int, int]]:
        return [(u + offset, v + offset) for u, v in edges]

    off_t = 0
    off_l = n_prev
    off_r = 2 * n_prev

    edges: List[Tuple[int, int]] = []
    edges += offset_edges(edges_prev, off_t)
    edges += offset_edges(edges_prev, off_l)
    edges += offset_edges(edges_prev, off_r)

    parent = list(range(3 * n_prev))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    a_t, b_t, c_t = 0, 1, 2

    t_b = off_t + b_t
    t_c = off_t + c_t
    l_a = off_l + a_t
    l_c = off_l + c_t
    r_a = off_r + a_t
    r_b = off_r + b_t

    union(t_b, l_a)
    union(t_c, r_a)
    union(l_c, r_b)

    rep_to_new: Dict[int, int] = {}
    new_id = 0

    def get_new(x: int) -> int:
        nonlocal new_id
        r = find(x)
        if r not in rep_to_new:
            rep_to_new[r] = new_id
            new_id += 1
        return rep_to_new[r]

    edge_set = set()
    for u, v in edges:
        nu, nv = get_new(u), get_new(v)
        if nu == nv:
            continue
        a, b = (nu, nv) if nu < nv else (nv, nu)
        edge_set.add((a, b))

    return new_id, sorted(edge_set)


def sparse_laplacian_from_edges(n: int, edges: List[Tuple[int, int]]):
    if n <= 0:
        raise ValueError("n must be positive")

    rows = []
    cols = []
    data = []

    degree = np.zeros(n, dtype=np.int64)

    for u, v in edges:
        rows.extend([u, v])
        cols.extend([v, u])
        data.extend([1.0, 1.0])
        degree[u] += 1
        degree[v] += 1

    adjacency = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    laplacian = diags(degree.astype(float), offsets=0, format="csr") - adjacency
    return laplacian, degree


def default_capacities(n: int) -> List[int]:
    base = [3, 5, 8, 13, 21, 34, 55, 89]
    fracs = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
    caps = base + [max(3, int(frac * n)) for frac in fracs]
    return sorted(set(c for c in caps if 2 <= c <= n))


def choose_k_eigs(n: int, caps: List[int], k_extra: int = 16) -> int:
    """
    Compute only as many low eigenvalues as needed for the largest requested capacity,
    plus a small buffer.
    """
    if not caps:
        raise ValueError("caps must not be empty")
    k_target = max(caps) + k_extra
    k_target = min(k_target, n - 1)  # eigsh requires k < N for sparse Hermitian solve
    k_target = max(k_target, 2)
    return k_target


def compute_low_spectrum(laplacian, k: int) -> np.ndarray:
    """
    Smallest algebraic eigenvalues of sparse symmetric Laplacian.
    """
    vals = eigsh(laplacian, k=k, which="SM", return_eigenvectors=False)
    vals = np.sort(np.real(vals))
    vals[np.abs(vals) < 1e-12] = 0.0
    return vals


def compute_capacity_curves(taus: np.ndarray, eigenvalues: np.ndarray, caps: List[int]) -> Dict[int, np.ndarray]:
    """
    Vectorized low-mode curves using only the first max(caps) eigenvalues.
    """
    exp_matrix = np.exp(-np.outer(taus, eigenvalues))  # shape: (n_tau, k)
    prefix = np.cumsum(exp_matrix, axis=1)

    curves: Dict[int, np.ndarray] = {}
    for c in caps:
        curves[c] = prefix[:, c - 1] / float(c)
    return curves


def compute_tau_crit(eigenvalues: np.ndarray, c_obs: int) -> float:
    idx = min(max(c_obs - 1, 0), len(eigenvalues) - 1)
    lam = float(eigenvalues[idx])
    if lam <= 0:
        return float("inf")
    return 1.0 / lam


def run_experiment(
    levels: List[int],
    n_tau: int = 90,
    capacities: List[int] | None = None,
    k_extra: int = 16,
) -> Tuple[Dict, List[List[float]]]:
    all_data: Dict = {
        "meta": {
            "expected_spectral_dimension": float(SIERPINSKI_D_S),
            "mode": "sparse_low_spectrum_capacity_runner",
        },
        "levels": {},
    }
    summary_rows: List[List[float]] = []

    for level in levels:
        print(f"Running level {level}...")

        n, edges = sierpinski_gasket_edges(level)
        caps = capacities if capacities is not None else default_capacities(n)
        caps = [c for c in caps if 2 <= c <= n]

        if not caps:
            raise ValueError(f"No valid capacities for level {level} with N={n}")

        laplacian, degree = sparse_laplacian_from_edges(n, edges)

        k_eigs = choose_k_eigs(n, caps, k_extra=k_extra)
        eigenvalues = compute_low_spectrum(laplacian, k=k_eigs)

        lam_max_low = float(max(eigenvalues[-1], 1.0))
        taus = np.logspace(-3, np.log10(5e3 / lam_max_low), n_tau)

        curves = compute_capacity_curves(taus, eigenvalues, caps)

        level_payload = {
            "N": int(n),
            "edge_count": int(len(edges)),
            "degrees": {
                "min": int(np.min(degree)),
                "max": int(np.max(degree)),
                "mean": float(np.mean(degree)),
            },
            "taus": taus.tolist(),
            "low_spectrum_k": int(k_eigs),
            "low_eigenvalues": eigenvalues.tolist(),
            "capacities": {},
        }

        for c in caps:
            p_c = curves[c]
            tau_crit = compute_tau_crit(eigenvalues, c)

            level_payload["capacities"][str(c)] = {
                "P_C": p_c.tolist(),
                "tau_crit": float(tau_crit),
            }

            summary_rows.append(
                [
                    int(level),
                    int(n),
                    int(len(edges)),
                    int(k_eigs),
                    int(c),
                    float(c / n),
                    float(tau_crit),
                ]
            )

        all_data["levels"][str(level)] = level_payload

    return all_data, summary_rows


def save_outputs(run_dir: Path, payload: Dict, summary_rows: List[List[float]]) -> None:
    with open(run_dir / "data.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(run_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "level",
                "N",
                "edge_count",
                "low_spectrum_k",
                "C_obs",
                "C_ratio",
                "tau_crit",
            ]
        )
        writer.writerows(summary_rows)


def save_manifest(
    run_dir: Path,
    args: argparse.Namespace,
    payload: Dict,
) -> None:
    manifest = {
        "runner": "exp1_sparse_capacity_runner",
        "levels": args.levels,
        "n_tau": args.n_tau,
        "k_extra": args.k_extra,
        "capacities": args.capacities,
        "expected_spectral_dimension": payload["meta"]["expected_spectral_dimension"],
        "notes": [
            "This runner computes only low Laplacian eigenvalues using scipy.sparse.linalg.eigsh.",
            "Outputs are capacity-limited observables only; no exact full-spectrum P_full is computed.",
            "Run directory is unique and immutable by construction.",
        ],
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sparse capacity-only spectral runner")
    parser.add_argument("--output", default="./results")
    parser.add_argument("--levels", type=int, nargs="*", default=[2, 3, 4, 5])
    parser.add_argument("--n-tau", type=int, default=90)
    parser.add_argument("--k-extra", type=int, default=16)
    parser.add_argument("--capacities", type=int, nargs="*", default=None)
    args = parser.parse_args()

    base = Path(args.output)
    run_dir = make_run_dir(base)

    print(f"Run directory: {run_dir}")

    payload, summary_rows = run_experiment(
        levels=args.levels,
        n_tau=args.n_tau,
        capacities=args.capacities,
        k_extra=args.k_extra,
    )

    save_outputs(run_dir, payload, summary_rows)
    save_manifest(run_dir, args, payload)

    print("Finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
