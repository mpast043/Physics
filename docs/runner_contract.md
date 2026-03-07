# Runner Contract

## Purpose

This document defines the required structure and behavior for all runner scripts in this repo.

The contract exists to ensure that:

- every runner behaves in a predictable way
- outputs are reproducible
- runs do not overwrite one another
- analysis scripts can consume outputs without special-case cleanup
- runner logic stays data-first and does not drift into verdict logic

This is a contract for runner behavior, input/output structure, and saved artifacts.

---

## Applies To

This contract applies to the following runner scripts:

- `Spectral_Dimension_Runner.py`
- `Capacity_Plateau_Runner.py`
- `Isometric_Gluing_Runner.py`
- `physical_convergence_runner.py`
- `xxz_boundary_fits.py`

---

## Core Rules

### 1. Data first
A runner collects raw measurements and minimal derived summaries only.

### 2. No verdict logic
A runner must not emit:
- ACCEPT / REJECT
- supported / unsupported
- passed / failed theory claim
- falsifier outcomes
- framework verdict language

### 3. Unique run directory required
Every run must create and write to a unique run directory.

### 4. No silent overwrite
A runner must never silently overwrite prior results.

### 5. Analysis is separate
A runner may compute minimal local summaries, but full comparison, cross-run interpretation, plotting, and literature benchmarking belong in analysis scripts or docs.

### 6. Warnings must be explicit
Any fallback, truncation, fit exclusion, unstable computation, or partial failure must be logged.

---

## Standard Runner Lifecycle

Every runner should follow this order:

1. load config
2. validate inputs
3. generate run ID
4. create unique output directory
5. initialize log
6. save `config.json`
7. save `metadata.json` start block
8. collect raw data
9. compute minimal summary values
10. save raw artifacts
11. save `summary.json`
12. finalize `metadata.json`
13. close with status and warnings

---

## Required Input Shape

Every runner may have additional runner-specific parameters, but the following common fields should exist in either the config file, CLI arguments, or internal normalized config.

## Common required inputs

```text
runner_name
model_name
run_label
seed
output_root
save_raw_data
save_plots
