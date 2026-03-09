# Runner Contract

## Purpose

This document defines the required structure and behavior for all runner scripts in this repository.

The contract exists to ensure that:

- every runner behaves in a predictable way
- outputs are reproducible and traceable
- runs do not overwrite one another
- downstream analysis can consume outputs without manual cleanup
- runner logic remains data-first
- bounded validation metadata is recorded without drifting into framework-level verdict logic

This is a contract for runner behavior, input/output structure, artifact layout, and reproducibility safeguards.

---

## Applies To

This contract applies to the following runner scripts:

- `Spectral_Dimension_Runner.py`
- `Capacity_Plateau_Runner.py`
- `Isometric_Gluing_Runner.py`
- `physical_convergence_runner.py`
- `XXZ_Boundary_Runner.py`

If additional runners are added, they should follow this contract unless a runner-specific contract supersedes it.

---

## Core Rules

### 1. Data first

A runner collects raw measurements and minimal derived summaries only.

Allowed:
- raw numerical outputs
- per-run summaries
- per-parameter best-of-run summaries
- reproducibility metadata
- explicit warnings and failure records
- bounded validation metadata tied to numerical sanity checks

Not allowed:
- cross-run scientific interpretation
- literature comparison
- framework claim language
- roadmap conclusions
- paper-style discussion

---

### 2. No framework verdict logic

A runner must not emit framework-level verdicts such as:

- ACCEPT / REJECT
- supported / unsupported theory claim
- falsifier passed / failed
- proof language
- global framework success language

A runner may emit bounded local validation metadata, such as:

- whether a numerical sanity check passed
- whether a branch is tagged `baseline` or `investigation`
- whether an energy sanity rule was violated
- whether a run is eligible for later review under repo policy

This distinction is important:
- **local numerical validation is allowed**
- **framework interpretation is not**

---

### 3. Unique run directory required

Every run must create and write to a unique run directory.

A runner must never write directly into a shared results root without creating a fresh run subdirectory.

Example:

```text
results/physical_convergence/RUN_20260308T174010Z_107907_48b41865/