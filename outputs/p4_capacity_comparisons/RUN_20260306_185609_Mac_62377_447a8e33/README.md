# MERA Capacity Allocator Run

Run ID: `RUN_20260306_185609_Mac_62377_447a8e33`
Created (UTC): `2026-03-06T18:56:09+00:00`
Seed: `42`

## Inputs
- n_sites_values: [8]
- chi_values: [8]
- trials_per_condition: 5

## Files
- `manifest.json` — reproducibility metadata
- `raw_trials.jsonl` — raw per-trial JSON records
- `raw_trials.csv` — raw per-trial CSV records
- `summary.csv` — grouped summary by `(n_sites, chi)`
- `summary.json` — grouped summary in JSON
- `overall_summary.json` — top-level summary
- `capacity_ratio_chi_*.png` — optional plots when `--plots` is enabled
- `mera_win_fraction.png` — optional plot when `--plots` is enabled

## Overall
- conditions: 1
- mean capacity ratio (random / mera): 0.30506329113924047
- median capacity ratio (random / mera): 0.30506329113924047
- mean MERA win fraction: 0.0
- all conditions mostly favor MERA: False
