# GREENFIELD/evals/AGENTS.md

Evaluation code for the Neural-PBIR / nvdiffrecmc comparisons on Stanford-ORB.

- `subset_sorb_eval.ipynb` — our own eval notebook; it consumes the raw eval output JSON now under `ASSETS/evals/` (full outputs) and `ASSETS/evals/subset/` (subset outputs).
- This directory was split out of a former root-level `evals/` per the `ASSETS/`/`GREENFIELD/` data-separation convention in the root `AGENTS.md` — the data half of that split lives in `ASSETS/evals/`.
