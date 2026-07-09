# GREENFIELD/AGENTS.md

This file is special: it holds the research idea and domain intent for the project, not just directory bookkeeping. See root `AGENTS.md` for general repo conventions. Engineering detail for code we've added inside vendored folders (e.g. `DigitalTwinCatalog`'s `run_restir.py`/`opt.py`, its gin configs) lives in root `BROWNFIELD.md`, not here — those files physically live in brownfield territory even though we wrote them.

## Research idea

Replace the standard PathTracer backward pass in the **Neural-PBIR** physics-based inverse rendering (PBIR) stage with **PathReSTIR** (spatial sample resampling, via `psdr-jit`) to reduce gradient variance in inverse rendering — particularly for specular materials — without increasing per-pixel sample count.

## Domain-specific intent

- **Neural-PBIR** is a multi-stage inverse-rendering pipeline (Neural Surface Recon → Neural Distillation → PBIR → Evaluation) that recovers geometry, materials, and lighting from posed multi-view images, evaluated on **Stanford-ORB**.
- Standard PBIR uses a plain PathTracer for both forward render (loss) and backward render (gradient). PathTracer gradients are noisy for glossy/specular BRDFs at low sample counts, limiting how far spp can be reduced.
- **PathReSTIR** substitutes a ReSTIR-style spatial reservoir resampling pass into the *backward* (gradient) render only, reusing candidate paths across neighboring pixels so fewer backward samples suffice.
- The forward render stays a standard PathTracer — only the backward/gradient pass uses ReSTIR. Both are unbiased MC estimators of the same integral, so this is variance reduction for differentiable rendering, not a new forward algorithm: visual outputs should match standard Neural-PBIR, gradient quality/convergence is what should improve.

## Current status

`GREENFIELD/` is otherwise empty right now — the prior iteration's scripts/notebooks (`nbir_setup_refined.*`, `nbir_restir_pipeline.py`, `test_shape_opt_cornell.py`, `evals/`) were archived wholesale to `OLD/it4_greenfield_restir_v1/` on 2026-07-09 to restart clean (see `OLD/AGENTS.md`). This doc was deliberately left in place and is being replanned; treat Results/Next Steps below as the record to pick back up from, not as pointing at live code.

## Pipeline map

```
Stanford-ORB → Neural Surface Recon → Neural Distillation → PBIR (ReSTIR lives here) → Evaluation
```
All four stages run inside vendored `DigitalTwinCatalog/neural_pbir/` (+ `Stanford-ORB/scripts/test.py` for eval) — see `BROWNFIELD.md` for the concrete scripts, integrator wiring, SPP settings, and gin config layout for the ReSTIR-specific additions (`run_restir.py`, `opt.py` WandB hooks).

## Known issues — psdr-jit itself

(`psdr-jit/` is directly-tracked and actively developed by us, not vendored, so these stay here rather than in `BROWNFIELD.md`.)

- **`PathReSTIR::renderC` is direct-illumination-only** (a stub needed for silhouette/boundary edges) — using it for the forward loss would compare a direct-only render against a full-path target and bias the gradient. This is *why* forward/backward are decoupled (see Domain-specific intent above).
- **Sampler size bug:** `renderD` seeds `sampler[0]` with `num_pixels` states but `__render<true>` needs `num_pixels × spp`. Fixed with re-seeding before the backward render call in `psdr-jit/src/integrator/path_restir.cpp`; rebuilt `.so` copied to `$CONDA_PREFIX/lib/python3.11/site-packages/psdr_jit/psdr_jit.so`.

## Results

**Cornell box shape optimization (50 iters):**

| Method | n_candidates | bwd spp | lr | Final L1 | Final Y (GT=82.5) | Speed |
|---|---|---|---|---|---|---|
| Standard PathTracer | — | 16 | 5.0 | 0.00660 | 82.9 | 0.215 it/s |
| PathReSTIR-cand4 | 4 | 1 | 3.0 | 0.00665 | 87.1 | ~0.1 it/s |
| PathReSTIR-cand8 | 8 | 1 | 3.0 | 0.00663 (iter 30) | 82.2 | ~0.1 it/s |

PathReSTIR-cand8 was near GT at iter 30 but the run was killed by the grad-vis hang (now fixed, see `BROWNFIELD.md`).

**Stanford-ORB (teapot_scene001) — ReSTIR pipeline:** hasn't run yet. First full run is still Next Step #1.

## Run Protocol

This is the canonical iteration-count policy for this project — if any other note (memory, chat, older doc) conflicts, this section wins.

- **50 iters** for Cornell box benchmarks.
- **200 iters** for Stanford-ORB ReSTIR PBIR (initial eval, vs 500 for standard); `checkpoint_iter=25` (vs 50 standard) for more frequent saves.
- Always run with `python -u` or `PYTHONUNBUFFERED=1` (`conda run` buffers stdout otherwise — see root `AGENTS.md`).
- No gradient visualization in production runs (was causing JIT hangs, see Known Issues in `BROWNFIELD.md`).
- Update this file after each run with config, results, and observations.

## Next Steps

1. Rebuild/restore a ReSTIR pipeline entry point in `GREENFIELD/` (from `OLD/it4_greenfield_restir_v1/` or fresh) and run it on teapot_scene001; record eval metrics here.
2. Compare ReSTIR vs standard on the same scene.
3. Check whether ReSTIR gradient variance is actually lower for specular objects (WandB loss curves).
4. If JIT compilation hangs again: check `~/.cache/drjit` — pre-warmed kernels persist across runs.
5. Extend to the remaining 6 Stanford-ORB scenes once single-scene convergence is confirmed.
6. Tune `n_candidates`, `n_neighbors`, `spatial_radius` for Stanford-ORB's image resolution.
