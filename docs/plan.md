# Research Runner Plan

## Status

This document locks the current scope, workflow, and priorities for the runner repository.

The goal of the current phase is not to prove the framework or embed scientific verdicts into code.  
The goal is to maintain a stable, reproducible, data-first pipeline for collecting, preserving, and analyzing results across the current runner set.

This plan exists to reduce drift, prevent overwrites, keep the project legible, and align repository structure with the evidence workflow now in use.

---

## Current Phase Goal

The current phase is focused on three things:

1. preserving clean, reproducible runner outputs
2. separating runner data production from cross-run interpretation
3. supporting bounded convergence claims from frozen evidence sets without embedding those claims inside runner code

This is no longer a plateau-first phase.

The strongest supported workflow priority is:

**correctness → convergence → frontier stabilization → plateau only if later supported**

---

## Scope Lock

The repository is currently limited to these five runner families:

1. `spectral_dimension`
2. `capacity_plateau`
3. `isometric_gluing`
4. `physical_convergence`
5. `xxz_boundary`

No additional runner families should be added until these five are stable, documented, and analyzable without custom cleanup.

---

## Current Research Priority

The current repo-level research priority is to support trustworthy convergence analysis on the trusted branch, especially through the `physical_convergence` workflow.

This means:

- correctness checks come before higher-level interpretation
- convergence against exact references comes before plateau analysis
- targeted continuation and frontier stabilization are valid workflow extensions
- plateau behavior should only be revisited after convergence is stable and monotone in a trusted regime

At present, plateau is a downstream question, not the primary organizing principle.

---

## Project Goal

Build and maintain one clean evidence pipeline that does the following:

1. generates raw numerical data
2. stores all metadata needed to reproduce a run
3. separates simulation from analysis and documentation
4. prevents accidental overwrites
5. supports cross-run comparison without special-case cleanup
6. supports frozen evidence summaries based on reproducible run artifacts

---

## Non-Goals

This phase does **not** aim to do the following:

- encode framework acceptance criteria in runners
- encode falsifiers in runners
- embed scientific verdict language in runner output
- treat plateau-like behavior as mandatory evidence
- add new experiment families before the architecture is stable
- force final conceptual wording before the pipeline is clean

Unknown scientific wording should be tracked explicitly as an open question, not treated as a blocker.

---

## Working Principles

### 1. Data first

Each runner collects raw measurements and minimal derived summaries only.

### 2. No framework verdict logic in runners

Runners do not emit framework-level conclusions such as ACCEPT/REJECT, supported/refuted, or theory pass/fail.

### 3. Bounded local validation is allowed

Runners may record bounded local validation metadata such as:

- energy sanity checks
- branch policy tags
- explicit warnings
- restart failure counts
- local eligibility flags for later review

This is allowed because it is local numerical metadata, not framework adjudication.

### 4. Unique run directories only

Every run writes to a new run-specific output directory.

### 5. Analysis is separate

Simulation produces data. Analysis scripts, summaries, and docs compare, interpret, and contextualize results later.

### 6. Frozen evidence is a documentation-layer decision

Runners produce reproducible artifacts.  
Docs and evidence summaries decide which runs are trusted anchors or frozen evidence.

### 7. Open questions are explicit

Anything conceptually unclear should be written into `docs/OPEN_QUESTIONS.md`.

---

## Repo Structure

The exact repository may continue to evolve, but the working structure should remain close to:

```text
repo_root/
  README.md
  docs/
    PLAN.md
    RUNNER_CONTRACT.md
    OPEN_QUESTIONS.md

  runners/
    Spectral_Dimension_Runner.py
    Capacity_Plateau_Runner.py
    Isometric_Gluing_Runner.py
    physical_convergence_runner.py
    XXZ_Boundary_Runner.py

  shared/
    mera_backend.py
    io_utils.py
    run_utils.py
    schema.py
    fit_utils.py

  results/
    spectral_dimension/
    capacity_plateau/
    isometric_gluing/
    physical_convergence/
    xxz_boundary/