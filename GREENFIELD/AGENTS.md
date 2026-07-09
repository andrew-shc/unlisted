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

### Cornell box shape optimization (50 iters)

| Method | n_candidates | bwd spp | lr | Final L1 | Final Y (GT=82.5) | Speed |
|---|---|---|---|---|---|---|
| Standard PathTracer | — | 16 | 5.0 | 0.00660 | 82.9 | 0.215 it/s |
| PathReSTIR-cand4 | 4 | 1 | 3.0 | 0.00665 | 87.1 | ~0.1 it/s |
| PathReSTIR-cand8 | 8 | 1 | 3.0 | 0.00663 (iter 30) | 82.2 | ~0.1 it/s |

### Stanford-ORB evaluation metrics

The ReSTIR PBIR (fwd=16 spp, cand=8, bwd=1 spp, 500 iters) was evaluated on 7 scenes (teapot/grogu/gnome/car/pitcher/blocks/cactus) using the Stanford-ORB harness. Metrics are reported across five task categories:

- **View** (Novel View Synthesis): PSNR HDR, PSNR LDR, LPIPS, SSIM
- **Light** (Relighting): PSNR HDR, PSNR LDR, LPIPS, SSIM
- **Geometry** (Depth/Normal): Normal Angle, Depth MSE
- **Material** (Albedo): PSNR LDR, LPIPS, SSIM
- **Shape** (Mesh): Bidirectional Chamfer Distance

PBIR speed: ~100s for 500 ReSTIR iters vs ~210s for 500 standard iters (2× faster).
ReSTIR backward pass uses 1 spp with spatial resampling vs standard's 64 spp PathTracer (64× ray savings).
Material PSNR and SSIM were within margin of standard; Shape/Geometry are driven by the NSR stage, not PBIR.
WandB runs: https://wandb.ai/andrew-shc/ReSTIR%20PBIR (training), https://wandb.ai/andrew-shc/ReSTIR%20Evaluation (eval)

## Run Protocol

- **50 iters** for Cornell box benchmarks.
- **500 iters** for Stanford-ORB ReSTIR PBIR (matched to standard); `checkpoint_iter=25` for finer saves.
- Default ReSTIR config: `fwd_spp=16`, `n_candidates=8`, `n_neighbors=2`, `spatial_radius=5`.
- Forward pass: PathTracer (loss image). Backward pass: PathReSTIR (spatial resampling gradient).
- Always run with `python -u` or `PYTHONUNBUFFERED=1` (`conda run` buffers stdout otherwise — see root `AGENTS.md`).
- No gradient visualization in production runs (was causing JIT hangs, see Known Issues in `BROWNFIELD.md`).
- Update this file after each run with config, results, and observations.

## Next Steps

1. ✅ Full 7-scene Stanford-ORB ReSTIR evaluation complete (results above).
2. Compare ReSTIR vs standard on identical scene subsets (7 eval scenes only) for fair comparison.
3. Tune ReSTIR hyperparameters: try n_candidates=16 or 32; try n_neighbors=4; try bwd_spp=2 (still 32× savings vs 64).
4. Investigate why forward spp (4 vs 16 vs 64) has minimal impact on final metrics (<0.5 dB) — suggests backward pass variance dominates.
5. Run ReSTIR PBIR with pure PathTracer backward (no ReSTIR) as ablation to isolate ReSTIR's contribution from other config differences.
6. If JIT compilation hangs again: check `~/.cache/drjit` — pre-warmed kernels persist across runs.
