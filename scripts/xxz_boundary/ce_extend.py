#!/usr/bin/env python3
"""
Extend C_E (kappa2) computation to L=20,24 via sparse Lanczos + SVD.
No full RDM needed — Schmidt values come directly from SVD of reshaped psi.
"""
import math, sys, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

VAULT = Path("/sessions/beautiful-modest-einstein/mnt/Obsidian Vault")

def xxz_sparse(L, delta):
    n = 2**L
    rows, cols, vals = [], [], []
    for i in range(L-1):
        for s in range(n):
            si = (s >> i) & 1; sj = (s >> (i+1)) & 1
            rows.append(s); cols.append(s)
            vals.append(delta * (si-0.5)*(sj-0.5))
            if si != sj:
                sf = s ^ (1<<i) ^ (1<<(i+1))
                rows.append(s); cols.append(sf); vals.append(0.5)
    return sp.coo_matrix((vals,(rows,cols)),(n,n)).tocsr()

def ce_from_psi(psi, L):
    """Compute S (nats) and C_E from ground state via SVD of reshaped psi."""
    A = L // 2
    psi_mat = psi.reshape(2**A, 2**(L-A))
    sv = np.linalg.svd(psi_mat, compute_uv=False)
    lam = sv**2                       # Schmidt eigenvalues
    lam = lam[lam > 1e-15]
    lam /= lam.sum()                  # normalise
    ln_lam = np.log(lam)
    S  = float(-np.dot(lam, ln_lam))  # nats
    CE = float(np.dot(lam, ln_lam**2)) - S**2
    return S, CE

# Known ED values from JSON (kappa2 = CE in nats; S converted to nats)
known = {
    0.5: {8:  (0.69709233*math.log(2), 0.9714546821105731),
          12: (0.81831258*math.log(2), 1.0056553933887973),
          16: (0.90232597*math.log(2), 1.0230682105313902)},
    2.0: {8:  (0.75923906*math.log(2), 0.8903337929185299),
          12: (0.91172657*math.log(2), 0.8709861362668836),
          16: (1.02442827*math.log(2), 0.8350122617837537)},
}

results = {d: dict(known[d]) for d in [0.5, 2.0]}

for delta in [0.5, 2.0]:
    for L in [20, 24]:
        print(f"Δ={delta}, L={L}: building sparse H ({2**L} dim)...", end=" ", flush=True)
        H = xxz_sparse(L, delta)
        print(f"Lanczos...", end=" ", flush=True)
        evals, evecs = spla.eigsh(H, k=1, which="SA", tol=1e-12, maxiter=10000)
        psi0 = evecs[:,0].real; psi0 /= np.linalg.norm(psi0)
        S, CE = ce_from_psi(psi0, L)
        ratio = S/CE
        print(f"S={S:.6f}, C_E={CE:.6f}, S/C_E={ratio:.4f}")
        results[delta][L] = (S, CE)

# Save
out = {str(d): {str(L): {"S_nats": v[0], "CE": v[1], "S_over_CE": v[0]/v[1]} 
                for L,v in Ls.items()}
       for d, Ls in results.items()}
with open(VAULT/"ce_extended.json","w") as f: json.dump(out, f, indent=2)
print(f"\nSaved ce_extended.json")

# Summary
print("\n" + "="*60)
print(f"{'Δ':>5} {'L':>4} {'S (nats)':>10} {'C_E':>10} {'S/C_E':>8}")
print("-"*60)
for d in [0.5, 2.0]:
    for L in sorted(results[d]):
        S, CE = results[d][L]
        print(f"{d:>5.1f} {L:>4d} {S:>10.6f} {CE:>10.6f} {S/CE:>8.4f}")
print("="*60)
