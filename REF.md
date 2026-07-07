# Neural-PBIR + PathReSTIR — Architecture & Code Reference

## Goal

Replace the standard PathTracer backward pass in the **Neural-PBIR** PBIR stage with **PathReSTIR** (spatial sample resampling from `psdr-jit`) to reduce gradient variance — particularly for specular materials — without increasing per-pixel sample count.

---

## Pipeline Architecture

```
Stanford-ORB
   │
   ├─ Neural Surface Recon (NSR)
   │    scripts: neural_pbir/neural_surface_recon/run_template.py
   │    output:  {CHKPT}/{SCENE}/neural_surface_recon/mesh.obj
   │
   ├─ Neural Distillation
   │    scripts: neural_pbir/neural_distillation/run.py
   │    output:  {CHKPT}/{SCENE}/neural_distillation/{albedo,roughness,envmap}.exr
   │
   ├─ PBIR  ← ReSTIR lives here
   │    Standard:  neural_pbir/pbir/run.py
   │    ReSTIR:    neural_pbir/pbir/run_restir.py   ← NEW
   │
   │    Stage 1  microfacet_naive-envmap_sg   (SG envmap coarse init, PathTracer)
   │    Stage 2  microfacet_basis-envmap_ls   (LS envmap fine, PathReSTIR bwd)
   │
   │    output:  {CHKPT}/{SCENE}/pbir/{mesh,diffuse,roughness,envmap}.*
   │
   └─ Evaluation
        scripts: Stanford-ORB/scripts/test.py
        output:  {CHKPT}/eval_outputs_pbir.json
```

---

## Entry Points

| Script | Purpose |
|---|---|
| `nbir_setup_refined.py` / `.ipynb` | Standard Neural-PBIR pipeline, all 7 scenes |
| `nbir_restir_pipeline.py` | ReSTIR pipeline, teapot_scene001 only (initial eval) |
| `test_shape_opt_cornell.py` | Cornell box benchmark: PathTracer vs PathReSTIR |

---

## ReSTIR Integration (`run_restir.py`)

### Integrator layout in psdr-jit scene

```python
scene['integrator']        = Integrator('path',        ...)   # fwd renderC  → L1 loss
scene['integrator_restir'] = Integrator('path_restir', ...)   # bwd renderD  → gradient
scene['integrator_vis']    = Integrator('path',        ...)   # checkpoint renders
```

### PathReSTIRRenderer

Custom `torch.autograd.Function`:
- **forward()** → `connector.renderC(scene, fwd_options, integrator_id=0)` — full PathTracer
- **backward()** → `connector.renderD(grads, scene, bwd_options, bwd_integrator_id='integrator_restir')` — PathReSTIR

Stage 1 keeps a plain `Renderer` (PathTracer fwd+bwd) — ReSTIR overhead unwarranted for coarse SG init.

### SPP settings

| Stage | Renderer | fwd spp | bwd spp |
|---|---|---|---|
| Stage 1 (naive-envmap_sg) | PathTracer | 64 | 64 |
| Stage 2 (basis-envmap_ls) | PathReSTIRRenderer | 16 | 4 |

ReSTIR compensates for lower bwd spp via spatial resampling (n_candidates=4, n_neighbors=5).

### ReSTIR parameters

```python
RESTIR_CONFIG = {
    'max_depth':      3,
    'n_candidates':   4,   # reservoir candidates per pixel
    'n_neighbors':    5,   # spatial neighbours
    'spatial_radius': 10,  # pixel radius for neighbor search
    'hide_emitters':  False,
}
```

### Gin config structure

```
pbir/configs/restir_template/
  pipeline_restir.gin          # dataset_class = @NeuralPBIRDataset
  microfacet_naive-envmap_sg.gin   # model + num_epochs (render_opt overridden in Python)
  microfacet_basis-envmap_ls.gin   # model + num_epochs (render_opt overridden in Python)
```

### psdr-jit monkey-patches (applied at import in `run_restir.py`)

1. **Mesh handler** — re-applies `use_face_normal` after every `load_raw()` call (connector resets it)
2. **Integrator handler** — registers `'path_restir'` type; the stock connector only knows `'path'`, `'direct'`, `'field'`, `'collocated'`

These patches are identical to those in `test_shape_opt_cornell.py`.

---

## WandB Integration

### Initialization

`nbir_restir_pipeline.py` and `nbir_setup_refined.ipynb` both call `wandb.init()`.  
Credentials are loaded from `.env` (gitignored):

```
WANDB_API_KEY=...
WANDB_ENTITY=andrew-shc
WANDB_PROJECT=ReSTIR PBIR
```

Claude's read permission for `.env` is denied in `.claude/settings.json`.

### What's logged

| Source | Key | Content |
|---|---|---|
| `opt.py` (per iter) | `pbir/loss`, `pbir/image_loss`, `pbir/render_ms`, `pbir/opt_ms` | Training scalars |
| `opt.py` (checkpoint) | `pbir/vis` | Target‖rendered image strip |
| `opt.py` (final) | `pbir/final_diffuse`, `pbir/final_roughness`, `pbir/final_envmap` | Material maps |
| `opt.py` (final) | `pbir/animation` | GIF of vis renders across training |
| outer pipeline | `nsr_min`, `distill_min`, `pbir_restir_min`, `total_min` | Stage timings |
| outer pipeline | `eval` | Stanford-ORB JSON metrics |
| outer pipeline | `pbir/stage2_animation` | GIF path from PBIR stage |

---

## Known Issues / Design Decisions

### PathReSTIR `renderC` is direct-only

`PathReSTIR::renderC()` in psdr-jit is a direct-illumination-only stub (needed for silhouette/boundary edges). Using it for the forward loss would compare a direct-only render against a full-path target → biased gradient.

**Fix**: decouple forward and backward:
- Forward → standard `PathTracer::renderC` (full-path images for loss)
- Backward → `PathReSTIR::renderD` (full-path adjoint with spatial resampling)

Both are unbiased MC estimators of the same integral.

### Sampler size bug in psdr-jit PathReSTIR

`renderD` seeds `sampler[0]` with `num_pixels` states but `__render<true>` needs `num_pixels × spp` states.

**Fix**: added re-seeding before the backward render call in `psdr-jit/src/integrator/path_restir.cpp`. Rebuilt `.so` copied to `$CONDA_PREFIX/lib/python3.11/site-packages/psdr_jit/psdr_jit.so`.

### Gradient visualization hangs

`compute_dI_dY` builds a fresh `Scene()` + calls `configure()` every 5 iters.  
At iter 30 of ReSTIR-cand8, this hung for 1+ hour (DrJIT CUDA JIT).

**Fix**: removed `render_grad` from production runs. Loss curves + vis renders are sufficient.

### conda run stdout buffering

`conda run` buffers all stdout until process exit.  
**Fix**: use `conda run --no-capture-output` or `PYTHONUNBUFFERED=1`.

---

## Results

### Cornell box shape optimization (50 iters)

| Method | n_candidates | bwd spp | lr | Final L1 | Final Y (GT=82.5) | Speed |
|---|---|---|---|---|---|---|
| Standard PathTracer | — | 16 | 5.0 | 0.00660 | 82.9 | 0.215 it/s |
| PathReSTIR-cand4 | 4 | 1 | 3.0 | 0.00665 | 87.1 | ~0.1 it/s |
| PathReSTIR-cand8 | 8 | 1 | 3.0 | 0.00663 (iter 30) | 82.2 | ~0.1 it/s |

PathReSTIR-cand8 near GT at iter 30 but run killed at grad-vis hang.

### Stanford-ORB (teapot_scene001) — ReSTIR pipeline

Pending first full run of `nbir_restir_pipeline.py`. Results will update here.

---

## Run Protocol

- **50 iters** for Cornell box benchmarks.
- **200 iters** for Stanford-ORB ReSTIR PBIR (initial eval, vs 500 standard).
- **checkpoint_iter=25** for ReSTIR PBIR (vs 50 standard) — more frequent saves.
- Always run with `python -u` (unbuffered) or `PYTHONUNBUFFERED=1`.
- No gradient visualization in production runs (causes JIT hangs).
- Update this file after each run with config, results, and observations.

---

## File Index

```
metrology_ir/
  nbir_setup_refined.ipynb/.py     standard Neural-PBIR pipeline (7 scenes)
  nbir_restir_pipeline.py          ReSTIR pipeline (1 scene initial eval)
  test_shape_opt_cornell.py        Cornell box PathTracer vs PathReSTIR benchmark
  REF.md                           this file
  .env                             WandB keys (gitignored)

  DigitalTwinCatalog/neural_pbir/pbir/
    run.py                         standard PBIR entry point
    run_restir.py                  ReSTIR PBIR entry point  ← NEW
    opt.py                         training loop (WandB hooks added)
    configs/
      template/                    standard gin configs
      restir_template/             ReSTIR gin configs  ← NEW
        pipeline_restir.gin
        microfacet_naive-envmap_sg.gin
        microfacet_basis-envmap_ls.gin

  psdr-jit/src/integrator/
    path_restir.cpp                sampler size fix + PathReSTIR implementation
```

---

## Next Steps

1. Run `nbir_restir_pipeline.py` on teapot_scene001 and record eval metrics here.
2. Compare ReSTIR vs standard on same scene (run `nbir_setup_refined.py` for teapot).
3. Investigate whether ReSTIR gradient variance is actually lower for specular objects (check WandB loss curves).
4. If JIT compilation hangs again: check `~/.cache/drjit` — pre-warmed kernels persist across runs.
5. Extend to remaining 6 scenes once single-scene convergence is confirmed.
6. Tune `n_candidates`, `n_neighbors`, `spatial_radius` for Stanford-ORB image resolution.
