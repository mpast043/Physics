# Streamlined MERA Runners

This repository is a data-first runner suite for MERA-related experiments and diagnostics.

The code is organized so that reusable numerical logic lives in `shared/`, while the top-level runner scripts handle only:

- argument parsing
- run-directory creation
- looping / orchestration
- summary generation
- artifact writing

This split is intentional. It keeps the runner layer simple and makes it easier to debug or replace the MERA backend without rewriting every runner.

## What is included

Top-level runners:

- `Spectral_Dimension_Runner.py`
- `Capacity_Plateau_Runner.py`
- `physical_convergence_runner.py`
- `Isometric_Gluing_Runner.py`
- `XXZ_Boundary_Runner.py`

Shared modules:

- `shared/mera_backend.py`
- `shared/runner_utils.py`

Supporting notes:

- `REVIEW_NOTES.md`

## Project structure

```text
streamlined_mera_runners/
├── README.md
├── REVIEW_NOTES.md
├── Spectral_Dimension_Runner.py
├── Capacity_Plateau_Runner.py
├── physical_convergence_runner.py
├── Isometric_Gluing_Runner.py
├── XXZ_Boundary_Runner.py
├── __init__.py
└── shared/
    ├── __init__.py
    ├── mera_backend.py
    └── runner_utils.py
```

## Design goals

This suite is designed around a few rules:

1. data first
   - runners collect measurements and write artifacts
   - runners do not embed claim verdicts or acceptance logic

2. shared backend
   - ED, entanglement helpers, fidelity, and MERA optimization live in `shared/mera_backend.py`
   - runners do not import each other for core numerical work

3. reproducible outputs
   - every run writes into a unique run directory
   - JSON and CSV files are written atomically

4. safe refactoring path
   - the current backend keeps the existing quimb/TNOptimizer flow
   - if you later replace it with a more paper-aligned environment-sweep implementation, the runner interfaces can stay stable

## Runner overview

### 1. Spectral Dimension Runner

File: `Spectral_Dimension_Runner.py`

Purpose:

- synthetic calibration runner for spectral-dimension style fitting
- generates return-probability data from a power-law model with multiplicative noise
- estimates slope and inferred `d_s`

Important note:

- this is a synthetic pipeline-validation runner, not a physical diffusion/operator runner

Typical outputs:

- `metadata.json`
- `config.json`
- `raw.csv`
- `summary.json`
- `run.json`

### 2. Capacity Plateau Runner

File: `Capacity_Plateau_Runner.py`

Purpose:

- sweep MERA bond dimension `chi`
- collect entropy, fidelity, and energy
- compare simple growth models such as log-linear vs saturating behavior

It uses the shared backend for:

- exact diagonalization reference
- MERA optimization
- entropy / fidelity extraction

Typical outputs:

- `metadata.json`
- `config.json`
- `raw.csv`
- `fits.json`
- `summary.json`
- `run.json`

### 3. Physical Convergence Runner

File: `physical_convergence_runner.py`

Purpose:

- run MERA optimization across `chi` and restarts
- record raw restart-level outcomes
- keep best-per-chi summaries separate from raw trial data

This is the runner most directly tied to MERA optimization stability.

Typical outputs:

- `config.json`
- `ed_reference.json`
- `raw_results.csv`
- `best_per_chi.csv`
- `failures.json`
- `summary.json`
- `manifest.json`

### 4. Isometric Gluing Runner

File: `Isometric_Gluing_Runner.py`

Purpose:

- diagnostics runner for reduced density matrices and entropy consistency checks
- evaluates a true `A|B` bipartition using the actual complement of `A`

Important note:

- this is a diagnostics runner, not a full proof of MERA gluing correctness

Typical outputs:

- `metadata.json`
- `config.json`
- `summary.json`
- `run.json`

### 5. XXZ Boundary Runner

File: `XXZ_Boundary_Runner.py`

Purpose:

- data-first finite-size / boundary analysis for externally supplied XXZ entropy measurements
- reads `.csv` or `.json` measurement files
- performs grouped fits and summaries

Typical outputs:

- `metadata.json`
- `config.json`
- `group_fits.csv`
- `summary.json`
- `run.json`

## Shared modules

### `shared/mera_backend.py`

This module centralizes reusable numerical work:

- exact diagonalization
- entanglement helpers
- fidelity helpers
- reduced density matrix helpers
- MERA optimization wrapper

It also applies conservative thread caps for native scientific libraries:

- `OMP_NUM_THREADS`
- `OPENBLAS_NUM_THREADS`
- `MKL_NUM_THREADS`
- `VECLIB_MAXIMUM_THREADS`
- `NUMEXPR_NUM_THREADS`

That is mainly to reduce instability from multi-threaded BLAS / OpenMP interactions while debugging the optimization path.

### `shared/runner_utils.py`

This module centralizes common runner utilities:

- UTC timestamps
- run ID creation
- unique run-directory creation
- atomic JSON writing
- atomic CSV writing

## Dependencies

The exact environment depends on which runners you use, but the core stack is:

- Python 3.10+
- `numpy`
- `quimb`
- `scipy`
- `threadpoolctl`

Optional / backend-dependent:

- `cotengra`
- `torch`
- local `entanglement_utils`

A minimal install might look like:

```bash
pip install numpy scipy quimb threadpoolctl cotengra torch nevergrad cmaes optuna
```

If you already have a working environment for the original runners, use that first.

## Running the suite

Run from the repository root so the `shared` package resolves cleanly.

### Spectral Dimension smoke test

```bash
python Spectral_Dimension_Runner.py \
  --true-ds 1.5 \
  --noise-level 0.02 \
  --steps 10,20,40,80,160 \
  --seed 42 \
  --output ./results/spectral_dimension
```

### Capacity Plateau smoke test

```bash
python Capacity_Plateau_Runner.py \
  --L 8 \
  --A-size 4 \
  --model ising_cyclic \
  --chi-sweep 2,4 \
  --fit-steps 5 \
  --seed 42 \
  --output ./results/capacity_plateau
```

### Physical Convergence smoke test

```bash
python physical_convergence_runner.py \
  --L 8 \
  --A-size 4 \
  --model ising_open \
  --chi-sweep 2,4 \
  --restarts-per-chi 1 \
  --fit-steps 5 \
  --seed 42 \
  --output ./results/physical_convergence
```

### Isometric Gluing smoke test

```bash
python Isometric_Gluing_Runner.py \
  --L 8 \
  --A-size 4 \
  --model heisenberg_open \
  --output ./results/isometric_gluing
```

### XXZ Boundary smoke test

```bash
python XXZ_Boundary_Runner.py \
  --input ./path/to/xxz_measurements.csv \
  --output ./results/xxz_boundary
```

## Output convention

All runners create a unique run directory below the requested output root.

Example:

```text
results/capacity_plateau/run_CAPACITY_PLATEAU_20260306T190000Z_ab12cd34/
```

Within that directory, each runner writes a consistent core set of artifacts plus runner-specific raw data.

Common files include:

- `metadata.json`
- `config.json`
- `summary.json`
- `run.json`

Additional files depend on the runner.

## Current limitations

1. the backend still uses the current quimb `TNOptimizer` path
   - it is not yet a full environment-based MERA sweep implementation

2. optimization stability may still be limited by backend/runtime issues
   - the conservative thread caps reduce risk but do not guarantee full stability

3. the Spectral Dimension runner is synthetic
   - it validates the fitting pipeline, not a physical MERA-derived diffusion object

4. the Isometric Gluing runner is diagnostic
   - it checks density-matrix and entropy consistency, not full gluing theory

## Recommended workflow

1. run a tiny smoke test for each runner
2. inspect whether the run directory and artifacts are correct
3. only then increase `chi`, restarts, or fit steps
4. keep raw artifacts and summaries together per run
5. do analysis from written outputs rather than reusing mutable in-memory objects

## Notes on paper alignment

The code layout now mirrors the paper-level separation between reusable MERA machinery and experiment orchestration:

- backend primitives in `shared/mera_backend.py`
- runner orchestration in the top-level scripts

The backend is still an implementation convenience layer around the current optimization path. A future upgrade could replace that backend with a more explicit environment / sweep implementation without forcing a runner rewrite.

## Troubleshooting

### Import errors for `shared`

Run scripts from the repo root, not from a different working directory.

### `entanglement_utils` missing

The backend falls back to local reduced-density and entropy helpers when possible.

### MERA optimization crashes or hangs

Start with:

- very small `L`
- small `chi`
- `restarts-per-chi 1`
- small `fit-steps`

The backend already sets conservative thread defaults, but some stacks can still be fragile.

### Results look inconsistent

Check the raw artifact files first:

- raw CSV files
- failure logs
- summary JSON

Do not trust only terminal output.

## Next refactor target

The next major improvement would be replacing the current black-box optimizer path in `shared/mera_backend.py` with a more explicit MERA optimization loop built around reusable tensor/environment update primitives.

## March 08

Framework status

Current evidence supports controlled convergence under targeted continuation on the trusted heisenberg_open branch.

At L=8, the workflow converges essentially to the exact finite-size target. At L=16, targeted continuation produces stable near-ED solutions for both chi=8 and chi=16. In the tested regime, the difference between stabilized chi=8 and stabilized chi=16 is small, suggesting that continuation quality is currently a stronger lever than increasing bond dimension from 8 to 16.

Plateau behavior is not supported by the current frozen evidence set.

Reference comparison table
Case	Energy	Entropy	Fidelity	ΔE = E - E_ED	ΔS = |S - S_ED|	Interpretation
L=8, trusted success case, chi=16	-3.3749325980	0.4569870681	0.9999999997	0.0000000006	0.0000112950	Exact to near-exact convergence
L=16, stabilized continuation, chi=8	-6.8976045070	0.4850359906	0.9914124147	0.0141326385	0.0456576396	Stable near-ED solution
L=16, stabilized continuation, chi=16	-6.8978780524	0.4848293615	0.9918055675	0.0138590932	0.0458642687	Stable near-ED solution, only marginally different from chi=8
Note on the frozen chi=16 continuation pair 

Two chi=16 continuation seeds were frozen, and they agree closely. 

/workspaces/Physics/Keep/L16_Chi16_Best
/workspaces/Physics/Keep/L16_Chi16_Seed_168042

Seed 168042
Energy: -6.8983214874
Entropy: 0.4828124955
Fidelity: 0.9917535259
ΔE: 0.0134156582
ΔS: 0.0478811347

Seed 174042
Energy: -6.8978780524
Entropy: 0.4848293615
Fidelity: 0.9918055675
ΔE: 0.0138590932
ΔS: 0.0458642687

These two runs indicate that the L=16, chi=16 frontier has largely collapsed into the same near-ED solution class under targeted continuation.

Bounded framework claim

On the trusted heisenberg_open branch, the framework admits MERA realizations that converge cleanly to exact finite-size targets at L=8 and to stable near-ED solutions at L=16 under targeted continuation. In the current tested regime, continuation quality appears more important than increasing bond dimension from chi=8 to chi=16. Plateau behavior is not supported by the frozen evidence set.

Interpretation

The framework is no longer best described as a plateau-seeking program. The strongest supported result is controlled convergence on a trusted branch under exact comparison and targeted continuation.

The L=8 case is the clean validation anchor. The L=16 case is now better described as a stable near-ED extension of that anchor, rather than as a noisy diagnostic failure. The present limit is not branch sanity, but that higher-level plateau structure has not yet emerged as a robust additional signal beyond convergence.

/workspaces/Physics/Keep/L16_Chi8_Best_82042

Roadmap update

The next phase of work should focus on bounded convergence claims, frontier-style summaries, and only revisit plateau testing after a larger-size or alternate setup produces a stable monotone ordering across chi.

Results

The current framework workflow demonstrates controlled convergence of MERA realizations on the trusted heisenberg_open branch under exact comparison.

At L=8, the optimization converges essentially to the exact diagonalization (ED) target, with energy, entropy, and fidelity matching the reference within numerical precision. This case serves as the primary validation anchor for the workflow.

At L=16, targeted continuation produces stable near-ED solutions for both χ=8 and χ=16. The resulting states achieve fidelities above 0.99 and energies within approximately 1.4×10⁻² of the ED reference. The stabilized solutions at χ=8 and χ=16 are very similar, indicating that continuation quality currently has a larger impact on solution quality than increasing bond dimension from 8 to 16.

Plateau behavior is not observed in the frozen evidence set. The strongest supported result of the current implementation is therefore controlled convergence on a trusted branch under exact reference comparison.

If you'd like, I can also help you add a small diagram-style section (just text) showing the workflow ladder you ended up discovering:

Correctness → Convergence → Frontier stabilization → Plateau (not yet reached)

A_size  chi     seed         energy      entropy     fidelity           dE           dS  run_id
------------------------------------------------------------------------------------------------------------------------
     2    8    82042   -6.898920624  0.399437873  0.992717429  0.012816522  0.021039325  RUN_20260309T015149Z_21648_b5b990a0
     2   16   174042   -6.905611362  0.404770071  0.997135777  0.006125783  0.015707126  RUN_20260309T015420Z_23012_c3fb8501
     6    8    82042   -6.903926692  0.546067662  0.995471316  0.007810454  0.032254048  RUN_20260309T020438Z_28981_ef1e117e
     6   16   174042   -6.904906893  0.548111890  0.996334876  0.006830253  0.030209820  RUN_20260309T020604Z_29830_570fdd71