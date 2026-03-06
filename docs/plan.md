Paste this into `docs/PLAN.md`:

````md
# Research Runner Plan

## Status

This document locks the current scope and workflow for the runner repo.

The goal of this phase is **not** to prove the framework or embed scientific verdicts into code.  
The goal is to build a **stable, reproducible, data-first pipeline** for collecting, preserving, and analyzing results across the current runner set.

This plan exists to reduce drift, prevent overwrites, and keep the project legible.

---

## Scope Lock

The repo is currently limited to these five runners:

1. `spectral_dimension`
2. `capacity_plateau`
3. `isometric_gluing`
4. `physical_convergence`
5. `xxz_boundary`

No additional runners should be added until these five are standardized.

### Important note on Runner 4
This plan assumes the repo-level runner remains named `physical_convergence`.

If the actual code has shifted to a different purpose such as `capacity_comparison`, that must be treated as an explicit rename/scope decision and updated in this plan, the repo structure, and the file names at the same time.

---

## Project Goal

Build one clean evidence pipeline that does the following:

1. generates raw data
2. stores all metadata needed to reproduce a run
3. separates simulation from analysis
4. prevents accidental overwrites
5. supports cross-run comparison without special-case cleanup

---

## Non-Goals

This phase does **not** aim to do the following:

- encode acceptance criteria in runners
- encode falsifiers in runners
- embed scientific verdict language in runner output
- add new experiments before architecture is stable
- force perfect conceptual wording before the data pipeline is clean

Unknown scientific wording is tracked as an open question, not treated as a blocker.

---

## Working Principles

### 1. Data first
Each runner collects raw measurements and minimal derived summaries only.

### 2. No verdict logic in runners
Runners do not emit ACCEPT/REJECT, supported/refuted, or theory pass/fail conclusions.

### 3. Unique run directories only
Every run writes to a new run-specific output directory.

### 4. Analysis is separate
Simulation produces data. Analysis scripts fit, compare, summarize, and plot later.

### 5. Shared contract
All runners should save the same core artifact types and use a consistent summary schema.

### 6. Open questions are explicit
Anything conceptually unclear gets written into `docs/OPEN_QUESTIONS.md`.

---

## Repo Structure

```text
repo_root/
  README.md
  requirements.txt
  pyproject.toml

  runners/
    run_spectral_dimension.py
    run_capacity_plateau.py
    run_isometric_gluing.py
    run_physical_convergence.py
    run_xxz_boundary.py

  analysis/
    analyze_spectral_dimension.py
    analyze_capacity_plateau.py
    analyze_isometric_gluing.py
    analyze_physical_convergence.py
    analyze_xxz_boundary.py
    build_master_summary.py

  shared/
    io_utils.py
    fit_utils.py
    run_utils.py
    schema.py
    mera_backend.py

  docs/
    PLAN.md
    RUNNER_CONTRACT.md
    ANALYSIS_CONTRACT.md
    OPEN_QUESTIONS.md
    LITERATURE_NOTES.md

  configs/
    spectral_dimension/
    capacity_plateau/
    isometric_gluing/
    physical_convergence/
    xxz_boundary/

  results/
    spectral_dimension/
    capacity_plateau/
    isometric_gluing/
    physical_convergence/
    xxz_boundary/
````

---

## Output Contract

Each run must write to a unique directory, for example:

```text
results/xxz_boundary/RUN_20260306_140501/
```

Each run directory should contain at minimum:

```text
config.json
metadata.json
summary.json
run.log
```

And at least one raw measurement artifact such as:

```text
raw_data.csv
measurements.json
fits.json
fit_points.json
```

### Required meanings

* `config.json`
  Exact run configuration and input parameters.

* `metadata.json`
  Run ID, timestamp, environment details, version, seed, script identity.

* `summary.json`
  Minimal compact summary of the run. No verdict language.

* `run.log`
  Warnings, fallback behavior, and execution notes.

* raw measurement files
  The actual collected data needed to reproduce the summary.

---

## Shared Summary Shape

Every runner summary should follow the same top-level pattern as closely as possible:

```json
{
  "runner": "runner_name",
  "run_id": "RUN_...",
  "timestamp_utc": "...",
  "model": "...",
  "system_size": "...",
  "parameters": {},
  "status": "ok",
  "warnings": [],
  "observables": {},
  "fits": {},
  "artifacts": {}
}
```

Not every field must be populated the same way, but the structure should stay recognizable across runners.

---

## Runner Definitions

## 1. Spectral Dimension

**Purpose**
Measure spectral-dimension-like behavior from the object currently produced by the runner.

**Current object under test**
Synthetic return-probability curve generator or, if upgraded later, the derived diffusion/spectral operator produced by the run.

**Construction**
Generate return-probability data across selected steps under the configured spectral-dimension parameter and noise model.

**Raw outputs**

* diffusion steps
* return probabilities
* log-transformed fit coordinates where applicable

**Summary outputs**

* fitted slope
* estimated spectral dimension
* fit quality
* residual diagnostics

**Current note**
If this runner remains synthetic, it should be treated as a calibration/pipeline runner rather than direct physical evidence.

---

## 2. Capacity Plateau

**Purpose**
Measure whether the selected capacity observable exhibits a stable plateau-like region as auxiliary bond dimension changes.

**Object under test**
Capacity curve induced by the optimized network family across auxiliary bond dimension.

**Construction**
Generate and optimize a sequence of networks across increasing auxiliary bond dimension.

**Raw outputs**

* bond dimension values
* capacity values
* fidelity values if relevant
* energy values if relevant
* model-selection inputs

**Summary outputs**

* candidate plateau region
* plateau mean or saturation estimate
* spread/stability indicators
* model comparison metrics such as AIC/BIC

**Current note**
This runner is about plateau behavior, not framework proof.

---

## 3. Isometric Gluing

**Purpose**
Measure gluing quality and associated consistency diagnostics for the implemented isometric/gluing procedure.

**Object under test**
Optimized glued tensor-network state, map, or reduced-state structure produced by the gluing workflow.

**Construction**
Construct the selected state/network object and compute density-matrix or related diagnostics across the chosen cut.

**Raw outputs**

* reduced density matrices or derived diagnostics
* entropy values
* validity checks
* gluing/isometry-related error or consistency metrics
* optional energy and cut entropy support values

**Summary outputs**

* primary gluing/consistency metric
* density-matrix validity summary
* supporting entropy/energy values
* warnings

**Current note**
Energy and cut entropy may be useful support observables, but they should not replace the actual gluing/consistency identity of the runner.

---

## 4. Physical Convergence

**Purpose**
Measure whether selected physical observables stabilize as a numerical control parameter improves.

**Object under test**
Observable trajectory as a function of numerical control parameter such as bond dimension, restart quality, truncation setting, or fit depth.

**Construction**
Repeat the same physical setup across increasing numerical-control settings and collect the resulting observable values.

**Raw outputs**

* control parameter values
* per-run observable values
* per-restart outputs where applicable
* fidelity, entropy, energy, and elapsed time if relevant

**Summary outputs**

* best value per control setting
* final deltas between successive settings
* convergence trend
* warning flags for non-convergence or failed restarts

**Current note**
If this runner has actually become a different concept such as `capacity_comparison`, that rename must be made explicit and propagated everywhere.

---

## 5. XXZ Boundary

**Purpose**
Use XXZ entanglement data as a boundary/calibration runner for scaling analysis under different boundary settings.

**Object under test**
XXZ entanglement dataset across system size, bipartition, boundary condition, and control settings.

**Construction**
Ingest or generate entropy measurements for XXZ and fit the selected boundary-aware scaling form.

**Raw outputs**

* delta
* boundary condition
* system size
* bipartition index
* entropy
* fit coordinate
* fitted values and residuals where applicable

**Summary outputs**

* fit coefficients
* fit success/failure counts
* fit quality statistics
* optional effective central-charge estimate
* warnings about endpoint exclusion or parity filtering

**Current note**
This runner is data-first and calibration-oriented. Literature interpretation belongs in analysis/docs, not in the runner.

---

## Work Phases

## Phase 1: Freeze architecture

Complete the repo layout and stop changing high-level structure.

Deliverables:

* folder structure created
* runner file names fixed
* docs folder initialized
* results folder layout fixed

---

## Phase 2: Build shared utilities

Create the common infrastructure before refactoring every runner.

Deliverables:

* `shared/run_utils.py`
* `shared/io_utils.py`
* `shared/fit_utils.py`
* `shared/schema.py`

### Shared utility responsibilities

**run_utils.py**

* generate run IDs
* create unique run directories
* capture timestamps
* initialize logging
* store environment metadata

**io_utils.py**

* safe JSON writes
* safe CSV writes
* overwrite protection
* atomic writes where practical

**fit_utils.py**

* linear fits
* R²
* RMSE
* residual calculations
* log-coordinate helpers

**schema.py**

* required keys for summaries
* metadata validation
* per-runner schema helpers if needed

---

## Phase 3: Refactor runners in order

Refactor in this order:

1. `xxz_boundary`
2. `physical_convergence`
3. `spectral_dimension`
4. `capacity_plateau`
5. `isometric_gluing`

### Why this order

* `xxz_boundary` is already close to a clean data runner and helps standardize fit reporting.
* `physical_convergence` provides a good pattern for repeated-run output structure.
* `spectral_dimension` is important but still has wording questions, so it comes after the pattern is established.
* `capacity_plateau` benefits from shared fit/model-selection utilities.
* `isometric_gluing` is the most specialized and is easier to clean after the rest of the contract is stable.

---

## Phase 4: Add analysis layer

After at least the first two runners are standardized, build analysis scripts.

Deliverables:

* `analyze_xxz_boundary.py`
* `analyze_physical_convergence.py`
* `build_master_summary.py`

Then complete the remaining runner-specific analysis scripts.

### Master summary

`build_master_summary.py` should scan all available run summaries and build:

```text
results/master_summary.csv
results/master_summary.json
```

This becomes the global index of completed runs.

---

## Phase 5: Smoke tests only

Before any serious compute, every runner must pass a tiny smoke test.

Each smoke test should verify:

1. a unique run directory is created
2. required files exist
3. summary schema validates
4. raw outputs are readable
5. warnings are logged clearly
6. no overwrite occurs
7. the summary can be rebuilt from saved raw data

These are pipeline tests, not scientific tests.

---

## Phase 6: Pilot runs

Only after smoke tests are clean.

Pilot runs should be deliberately small and chosen to test the usefulness of the saved outputs, not maximize scientific coverage.

### Pilot run goals

* confirm summary fields are sufficient
* confirm raw outputs support later reanalysis
* confirm file naming and grouping are stable
* confirm warnings are informative

---

## Phase 7: Real sweeps

Only after smoke tests and pilot runs are clean.

At this point the pipeline should already be stable enough that larger compute does not create new organizational confusion.

---

## Immediate Task Order

1. finalize repo structure
2. create docs files
3. implement `shared/run_utils.py`
4. implement `shared/io_utils.py`
5. implement `shared/schema.py`
6. implement `shared/fit_utils.py`
7. refactor `run_xxz_boundary.py`
8. refactor `run_physical_convergence.py`
9. build `analyze_xxz_boundary.py`
10. build `build_master_summary.py`
11. refactor `run_spectral_dimension.py`
12. refactor `run_capacity_plateau.py`
13. refactor `run_isometric_gluing.py`
14. run smoke tests for all five
15. run pilot computations
16. only then start larger sweeps

---

## Definition of Done for This Phase

This phase is complete when all of the following are true:

* all five runners exist in stable locations
* all five write unique run directories
* all five produce consistent core artifacts
* all five avoid overwrite collisions
* no runner contains verdict or falsifier logic
* all five can be analyzed without custom cleanup hacks
* a master summary can be built across runs
* open conceptual questions are documented explicitly instead of floating in chat

---

## Stop Rules

Pause and fix structure if any of the following occurs:

* two runs attempt to write to the same output path
* summary fields drift across runners without a documented reason
* raw data is insufficient to reproduce the summary
* warnings are hidden or dropped
* one runner requires heavy special-case handling in analysis
* runner purpose and file name no longer match

---

## Open Questions Policy

Unresolved conceptual issues should be logged in `docs/OPEN_QUESTIONS.md` using this format:

```text
Open question
Current temporary wording
Why uncertain
What will resolve it
Owner
Status
```

Example:

```text
Open question:
What exactly is the object under test for spectral_dimension?

Current temporary wording:
Synthetic return-probability generator or derived diffusion/spectral operator.

Why uncertain:
Current implementation is synthetic, but future versions may derive the operator from a constructed object.

What will resolve it:
Audit the final build-and-measure path in the standardized runner.

Owner:
Megan

Status:
Open
```

---

## Final Guidance

This phase should be judged by clarity and reproducibility, not by how many experiments are added.

The project moves forward by making the current five runners trustworthy, comparable, and easy to analyze later.

That is the priority.

```

Next I’d draft `RUNNER_CONTRACT.md` so the five scripts all target the same schema.
```
