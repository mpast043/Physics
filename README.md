# Physics Research Repository

This repository contains computational physics research including runner suites,
data outputs, and the XXZ boundary entropy scope-gate paper.

---

## Repository Layout

```
Physics/
├── papers/
│   └── xxz_boundary/
│       ├── XXZ_Scope_Gate_Paper.tex   — arXiv-ready LaTeX source
│       └── XXZ_Scope_Gate_Paper.pdf   — Compiled PDF (11 pages)
│
├── figures/
│   └── xxz_boundary/
│       ├── XXZ_Fig1_SL_scaling.{pdf,png}  — S(L) scaling + AICc fits
│       └── XXZ_Fig2_CE.{pdf,png}           — C_E analysis (3-panel)
│
├── results/
│   └── xxz_boundary/
│       ├── ce_results.json     — C_E at L=8,12,16 (exact ED)
│       └── ce_extended.json    — C_E at L=8–44 (ED + extrapolation)
│
├── scripts/
│   └── xxz_boundary/
│       ├── ce_pipeline.py      — Compute C_E via ED (L≤16)
│       ├── ce_extend.py        — Extend to L=20 via sparse Lanczos
│       └── ce_fig2_final.py    — Fit models + generate Fig 2
│
├── runners/                    — MERA/DMRG runner scripts
│   ├── XXZ_Boundary_Runner.py
│   ├── Capacity_Plateau_Runner.py
│   ├── Isometric_Gluing_Runner.py
│   ├── Spectral_Dimension_Runner.py
│   ├── physical_convergence_runner.py
│   ├── shared/
│   │   └── mera_backend.py
│   └── results/                — Timestamped runner output dirs
│
├── outputs/                    — Outputs from named experiments
├── data/                       — Raw / curated data files
├── docs/                       — Design documents
│   ├── plan.md
│   └── runner_contract.md
├── Keep/                       — Hand-selected best runs
└── Archive/                    — Retired run directories
```

---

## XXZ Scope Gate Paper

**Title:** *Entanglement Entropy Scaling as an Automatic Scope Gate:
Validation on the Spin-½ XXZ Chain*

**Summary:** We introduce and validate a scope gate that automatically classifies
a quantum many-body system as amenable to holographic tensor-network methods
(IN-SCOPE) or not (OUT-OF-SCOPE) based on AICc model selection of finite-size
entanglement entropy scaling.

| Δ    | Phase          | ΔAICc  | Decision      |
|------|----------------|--------|---------------|
| 0.5  | XY critical    | +20.13 | OUT-OF-SCOPE  |
| 1.0  | KT transition  | ∞      | INCONCLUSIVE  |
| 2.0  | Néel gapped    | −21.06 | IN-SCOPE      |

Key secondary result: the capacity of entanglement C_E = Var(H_A) was
computed at L = 8–20 (exact ED) and extrapolated to L = 44 via finite-size
model fits.  Critical and gapped phases show opposite S/C_E trajectories,
providing a complementary holographic diagnostic.

**Source files:** `papers/xxz_boundary/`
**Figures:** `figures/xxz_boundary/`
**C_E data:** `results/xxz_boundary/ce_extended.json`

---

## Runners

The `runners/` directory contains MERA/DMRG diagnostic runners:

- **XXZ_Boundary_Runner.py** — XXZ boundary entropy scaling fits
- **Capacity_Plateau_Runner.py** — MERA entanglement capacity plateau
- **Isometric_Gluing_Runner.py** — Isometric tensor gluing diagnostics
- **Spectral_Dimension_Runner.py** — Spectral dimension measurement
- **physical_convergence_runner.py** — Physical convergence testing

The canonical gate runner lives in the `host-adapters` repo:
`PHYS_BORDER_XXZ_ED_runner_v1.py`

---

## Dependencies

```
numpy, scipy, matplotlib    # core numerics and plotting
pdflatex                    # LaTeX compilation (for paper)
```

For the ED scripts:
```
pip install scipy numpy matplotlib
```
