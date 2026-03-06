# Physics

Tensor network simulations for quantum many-body physics using Multi-scale Entanglement Renormalization Ansatz (MERA).

## Overview

This project implements MERA tensor network algorithms to study:

- **Spectral dimension estimation** via return probability scaling
- **Capacity scaling** (entanglement entropy vs bond dimension)
- **Isometric constraints** and gluing diagnostics
- **Boundary entropy** in critical systems (XXZ chain)
- **Physical convergence** against exact diagonalization (ED)

All simulations use the [`quimb`](https://quimb.readthedocs.io/) library for tensor network operations.

---

## Directory Structure

```
Physics/
├── runners/
│   ├── p1_spectral_dimension/      # Return probability → spectral dimension
│   ├── p2_capacity_plateau/        # Entropy vs χ: saturation vs log-linear
│   ├── p3_physical_convergence/    # MERA vs ED ground state energy & entropy
│   └── p4_physical_convergence/    # XXZ boundary entropy scaling
├── tools/
│   └── benchmark.py                # Runners + result aggregation
├── outputs/                         # JSON/CSV results + plots
└── README.md                        # This file
```

---

## Install

```bash
pip install quimb numpy scipy matplotlib
# optional for tensor contractions
pip install cotengra
```

---

## Quick Start

### 1. Spectral Dimension (P1)

Estimates the spectral dimension \(d_s\) from return probability decay:

\[
P(t) \sim t^{-d_s/2} \quad \Rightarrow \quad d_s = -2\,\frac{d\log P}{d\log t}
\]

```bash
python runners/p1_spectral_dimension/Spectral_Dimension_Runner.py \
  --model ising_cyclic --L 16 --steps 10000 --seed 42
```

**Output:** `outputs/p1_spectral_dimension/run_<timestamp>/measurements.csv`

### 2. Capacity Plateau (P2)

Tests whether entanglement entropy saturates (finite-system effect) or grows log-linearly:

\[
S(\chi) \stackrel{?}{=} S_{\max} - c\,\chi^{-\alpha}
\quad\text{vs}\quad
S(\chi) = S_\infty + \alpha\log\chi
\]

```bash
python runners/p2_capacity_plateau/Capacity_Plateau_Runner.py \
  --model ising_cyclic --L 16 --chi_sweep 8,16,32,64,128
```

**Output:** `outputs/p2_capacity_plateau/run_<timestamp>/results.csv`

### 3. Physical Convergence (P3)

Compares MERA vs ED ground-state energy and entanglement entropy:

```bash
python runners/p3_physical_convergence/physical_convergence_runner.py \
  --model ising_cyclic --L 12 --A_size 6 --chi_sweep 8,16,32 --restarts_per_chi 3
```

**Output:** `outputs/p3_physical_convergence/run_<timestamp>/results.json`

### 4. XXZ Boundary Entropy (P4)

Fits boundary entropy scaling in critical XXZ chains:

```bash
python runners/p4_xxz_boundary/xxz_boundary_runner.py \
  --model XXZ --L 16 --chi_sweep 8,16,32,64 --bc open
```

**Output:** `outputs/p4_xxz_boundary/run_<timestamp>/entropy_fit.csv`

---

## Output Format

All runs produce:

- **measurements.csv** / **results.csv** — step-wise or χ-sweep data
- **config.json** — hyperparameters and seeds
- **plots/** — log-log and linear visualizations
- **runs/** — MERA checkpoints (optional)

Sample CSV header:

```csv
step,return_prob,entropy,S_fit,S_resid,d_s_estimated,baseline
10,0.1234,0.6543,0.6540,0.0012,1.386,0.0015
20,0.0987,0.6821,0.6819,0.0009,1.379,0.0014
```

---

## Theory Summary

| Quantity | Definition | Scaling (critical 1D) |
|----------|------------|------------------------|
| **Spectral dimension** \(d_s\) | return probability \(P(t)\) | \(P(t)\sim t^{-d_s/2}\) |
| **Capacity** \(S\) | entanglement entropy | \(S\sim \frac{c}{6}\log L\) |
| **Boundary entropy** \(b\) | surface correction | \(S = \frac{c}{6}\log L + b\) |

For the 1D Ising universality class (c = 1/2), expect:
- \(d_s = 1\)
- \(S \sim 0.0833\log L\)

---

## References

- [quimb documentation](https://quimb.readthedocs.io/)
- MERA review: [Evenbly & Vidal, arXiv:1106.1082](https://arxiv.org/abs/1106.1082)
- Spectral dimension: [Lohmayer et al., arXiv:0906.2366](https://arxiv.org/abs/0906.2366)
- Boundary entropy: [Affleck & Ludwig, Phys. Rev. Lett. 67, 161 (1991)](https://doi.org/10.1103/PhysRevLett.67.161)

---

## License

MIT
