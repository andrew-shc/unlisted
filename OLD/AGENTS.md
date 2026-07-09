# OLD/AGENTS.md

Reference cabinet for our own code that's no longer in active use but might be useful to look back on later. Move things here instead of deleting them or leaving them cluttering the main tree.

The lowercase `old/` directory (`it0_intermediate_results_and_code`, `it1_defer_mitsuba_testings`, `it2_reverse_analysis`, `it3_trellis_prior`) that used to serve this purpose informally has been consolidated in here (2026-07-08).

Data/code separation still applies inside `OLD/`: all embedded data (`it2_reverse_analysis/results___*/` run outputs, loose `.ply`/`.png` files, `pose_enc.npy`/`.npz`) moved to the mirrored `ASSETS/OLD/...` layout (2026-07-08, ~16.5GB total) — `OLD/` itself keeps only the notebooks/scripts.

`it4_greenfield_restir_v1/` (2026-07-09): the entire prior contents of `GREENFIELD/` (all scripts/notebooks and the `evals/` subdir) were moved here wholesale to clear `GREENFIELD/` for a restart. `GREENFIELD/AGENTS.md` (the research-idea doc) was deliberately left in place, not moved — it's being revisited/replanned separately. Data under `ASSETS/` that these scripts produced (`restir_NPBIR_v1/`, `standard_NPBIR/`, `standard_NPBIR_reeval/`, `wandb/`, `RESTIR_SUMMARY.md`, etc.) was **not** relocated to mirror this move — it's still at the `ASSETS/` root rather than `ASSETS/OLD/it4_greenfield_restir_v1/...`; do that mirroring if/when this iteration is fully retired.
