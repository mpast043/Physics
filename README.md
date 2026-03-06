# Physics — Tensor Network Simulations for Quantum Many-Body Systems

This repository contains experimental research on tensor network methods for quantum many-body physics, with a focus on the **MERA** (Multi-scale Entanglement Renormalization Ansatz) approach to entanglement structure and holographic emergence.

Each test (`P1`–`P4` + XXZ) is implemented as a standalone runner script that:
- Accepts a few configuration arguments (system size, bond dimension, seed, etc.)
- Produces a timestamped output directory with `config.json`, `ed_reference.json`, `summary.json`, `raw_results.csv`, and optional plots
- Can be run independently or as part of a batch suite

## Directory Structure

```
Physics/
├── runners/                # Runner scripts grouped by test
│   ├── p1_spectral_dimension/
│   │   ├── Spectral_Dimension_Runner.py          # Synthetic return-probability scaling
│   │   └── new_spectral_dimension_sparse_runner.py  # Sierpinski gasket (real graph)
│   │
│   ├── p2_isometric_gluing/
│   │   └── Isometric_Gluing_Runner.py            # ED bipartition entanglement
│   │
│   ├── p3_physical_convergence/
│   │   ├── physical_convergence_runner.py        # MERA vs ED (small systems)
│   │   └── Capacity_Plateau_Runner.py            # MERA entropy plateau across χ (needs claim3)
│   │
│   ├── p4_capacity_comparisons/
│   │   └── Unstable_MERA_Capacity_Allocator.py   # MERA vs random TN proxy capacity
│   │
│   └── xxz_boundary_fits/
│       └── XXZ_Boundary_Runner.py                # CFT log-form fits (open/periodic BCs)
│
├── data/                   # Input data and raw outputs
│   ├── input/              # Preprocessed input CSVs (e.g., XXZ entropy measurements)
│   └── raw/                # Timestamped output directories from runners
│
├── outputs/                # Aggregated results
│   ├── summary/            # JSON/CSV summaries (per-test)
│   └── plots/              # Visualization artifacts
│
├── docs/                   # Documentation
│   ├── claims/
│   │   ├── CLAIM_P1.md       # Spectral dimension evidence
│   │   ├── CLAIM_P2.md       # Isometric gluing evidence
│   │   ├── CLAIM_P3.md       # Physical convergence evidence
│   │   └── CLAIM_XXZ.md      # XXZ boundary scaling evidence
│   └── specifications/
│       ├── RUNNER_SPECS.md   # Runner interfaces and arguments
│       └── OUTPUT_FORMAT.md  # JSON/CSV schema documentation
│
├── tests/                  # Falsifiers, unit tests, integration tests (future)
├── scripts/                # Convenience helpers (run_all_tests.sh, summarize_outputs.py)
└── README.md
```

## Installation & Dependencies

```bash
# Optional: create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install numpy scipy quimb

# Optional: for tensor-optimization acceleration
pip install cotengra

# Optional: for plotting
pip install matplotlib
```

## Usage

Each runner accepts similar command-line arguments and writes outputs to a timestamped directory.

### Example: Run Spectral Dimension (P1)

```bash
cd runners/p1_spectral_dimension

# Synthetic power-law scaling test
python3 Spectral_Dimension_Runner.py \
  --true-ds 1.333 \
  --noise-level 0.01 \
  --L 16 \
  --A_size 8 \
  --seed 42 \
  --output ../outputs/p1_spectral_dimension

# Real Sierpinski gasket graph test
python3 new_spectral_dimension_sparse_runner.py \
  --graph sierpinski \
  --L 16 \
  --A_size 8 \
  --seed 42 \
  --output ../outputs/p1_spectral_dimension
```

### Example: Run Isometric Gluing (P2)

```bash
cd runners/p2_isometric_gluing

python3 Isometric_Gluing_Runner.py \
  --model ising_cyclic \
  --L 8 \
  --A_size 4 \
  --seed 42 \
  --output ../outputs/p2_isometric_gluing
```

### Example: Run Physical Convergence (P3)

```bash
cd runners/p3_physical_convergence

python3 physical_convergence_runner.py \
  --model ising_cyclic \
  --chi_sweep 2,4,8,16 \
  --L 8 \
  --A_size 4 \
  --seed 42 \
  --output ../outputs/p3_physical_convergence

# Capacity plateau scan (needs external `claim3.PHYS_PHYSICAL_CONVERGENCE_runner_v2`)
# python3 Capacity_Plateau_Runner.py --model ising_cyclic --chi_sweep 2,4,8,16 --L 8 --A_size 4 --seed 42 --output ../outputs/p3_capacity_plateau
```

### Example: Run XXZ Boundary Fits

```bash
cd runners/xxz_boundary_fits

# Provide CSV with columns: delta, boundary, L, ell, entropy
python3 XXZ_Boundary_Runner.py \
  --input ../data/input/xxz_entropy_data.csv \
  --output ../outputs/xxz_boundary_fits
```

## Output Format

Each run creates a directory like `outputs/p1_spectral_dimension/2026-03-06_123456/` containing:

| File | Purpose |
|------|---------|
| `config.json` | Full runner configuration (seed, L, A_size, model, etc.) |
| `ed_reference.json` | Exact diagonalization reference values (energy, entropy, etc.) |
| `summary.json` | Key metrics and verdict (e.g., `d_s_estimated`, `fidelity`, `c_eff`) |
| `raw_results.csv` | Full data points used for fitting/scaling |
| `*.png` (optional) | Visualization of scaling fit or convergence |

All JSON files are valid UTF-8 and parseable with `json.load()`.

## Current Status

| Test | Status | Evidence |
|------|--------|----------|
| **P1** — Spectral Dimension | ✅ SUPPORTED | d_s ≈ 1.365 matches CFT expectation (power-law fit) |
| **P2** — Isometric Gluing | ✅ SCOPE_CORRECT | ED reduced density matrix construction validated; S(A), S(B) accurate |
| **P3** — Physical Convergence | ✅ ACCEPTED | MERA fidelity → 1 as χ increases (Ising & Heisenberg) |
| **P4** — Capacity Comparisons | ⚠️ PROXY | MERA vs random TN comparison implemented; use as feasibility check only |
| **XXZ Boundary** | ✅ SCOPE_VALIDATED | CFT log-form fits distinguish open vs periodic boundary conditions |

## Extending

1. **Add a new runner** — Copy an existing runner, modify `RunnerConfig`, and adjust the output format.
2. **Add a falsifier** — Create a test in `tests/` that checks whether an evidence metric passes/fails.
3. **Standardize outputs** — Reference `docs/specifications/OUTPUT_FORMAT.md` for schema guidance.

## License

This is experimental research code — use as a reference, not production software.
