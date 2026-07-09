# GREENFIELD/restir/AGENTS.md

## Purpose

Canonical GREENFIELD entry points for the ReSTIR differentiable rendering pipeline.
These scripts orchestrate the multi-stage Neural-PBIR pipeline (Neural Surface
Reconstruction → Neural Distillation → PBIR with PathReSTIR → Evaluation),
as well as standalone benchmarks for the PathReSTIR integrator.

The C++ ReSTIR implementation lives in `psdr-jit/` (`PathReSTIR`, `DirectReSTIR`).
The DigitalTwinCatalog runtime provides the PBIR infrastructure (`opt.py`,
`datasets.py`, `configs/`). These scripts are the user-facing entry points that
import from, call into, and orchestrate that infrastructure, adding WandB logging
throughout.

## Directory Structure

```
GREENFIELD/restir/
    __init__.py              # Package marker (empty)
    AGENTS.md                # This file
    run_restir.py            # PRIMARY: single-stage PBIR with PathReSTIR backward gradient
    cornell_benchmark.py     # Standalone Cornell box shape optimization benchmark
    pipeline_restir.py       # Full end-to-end Stanford-ORB pipeline orchestrator
    pipeline_setup.py        # Batch NSR + Neural Distillation setup for all scenes
    pipeline_evaluate.py     # Stanford-ORB evaluation on PBIR results
    generate_animation.py    # Cornell box shape optimization animation generator
```

## File Purposes

### `run_restir.py`

The canonical PBIR stage entry point. Users run this script for a single PBIR
optimization that uses PathTracer for the forward pass (full-path images for L1
loss) and PathReSTIR for the backward pass (low-variance gradient via spatial
resampling). Key architectural note: `PathReSTIR::renderC()` is a
direct-illumination-only stub — the forward pass *must* use a standard PathTracer
to get correct full-path images for the loss.

### `cornell_benchmark.py`

Standalone benchmark that compares three methods on a Cornell box shape
optimization (sphere Y position):
- Standard PathTracer (spp=16)
- PathReSTIR spp=1, n_candidates=4
- PathReSTIR spp=1, n_candidates=8

Builds the scene programmatically from OBJ files (no XML needed). Runs 50
iterations of Adam for each method. Logs everything to WandB and saves a GIF
animation + JSON numerical results.

### `pipeline_restir.py`

Full end-to-end orchestrator that runs all four stages of the Stanford-ORB
evaluation pipeline:
1. Neural Surface Reconstruction (NSR) → geometry
2. Neural Distillation → albedo/roughness/envmap initialization
3. PBIR with PathReSTIR → refined material + envmap
4. Evaluation → Stanford-ORB metrics

Each stage runs as a subprocess with proper error handling. Stage timings are
logged to WandB.

### `pipeline_setup.py`

Batch setup script that runs NSR + Neural Distillation for all 7 Stanford-ORB
light scenes (or a user-specified subset). Skips scenes that already have results.
Logged to WandB per-scene.

### `pipeline_evaluate.py`

Runs the Stanford-ORB evaluation pipeline on existing PBIR results. Handles
envmap postprocessing, relighting, novel view synthesis, and metric computation
via the `Stanford-ORB/scripts/test.py` evaluation harness.

### `generate_animation.py`

Standalone Cornell box shape optimization animation generator. Runs a short
optimization (60 iters) and produces a side-by-side GIF showing target vs.
optimizing render. Optional WandB logging.

## Known Gotchas

### conda run buffers stdout
`conda run` buffers stdout until the process exits. Use either:
- `conda run --no-capture-output python script.py`
- `PYTHONUNBUFFERED=1 python script.py`

### DrJIT CUDA JIT can hang for 1+ hour
At higher candidate counts (n_candidates ≥ 8) with ReSTIR, the DrJIT JIT
compilation can hang for 1+ hour on the first backward pass. This is not a crash
— just slow compilation. Pre-warmed kernels persist in `~/.cache/drjit` across
runs.

### PSDRJITConnector patches MUST be applied before any scene construction
The `Mesh` face-normal patch and `Integrator` path-restir patch must be applied
at module level, before any scene is built or `get_connector()` is called.
These patches are applied at import time in `run_restir.py` and
`cornell_benchmark.py`.

### Mesh `use_face_normal` patch required for PathReSTIR
PathReSTIR requires `use_face_normal=True` on meshes. The PSDRJITConnector's
`load_raw()` clears this flag. The `_mesh_handler_reapply_face_normal` patch
re-applies it after every `load_raw()` call.

### PathReSTIR::renderC() is direct-only stub
`PathReSTIR::renderC()` is a direct-illumination-only stub (needed for
silhouette/boundary edge gradients). It must NOT be used for forward-pass loss
images — use a standard PathTracer integrator for that. The pipeline always uses:
- Forward → PathTracer `renderC` (full-path images for L1 loss)
- Backward → PathReSTIR `renderD` (spatial-resampling gradient)

### Cornell box data path
The CBOX path for Cornell box OBJ data:
```
REPO_ROOT / 'psdr-jit/tutorials/data/cbox'
```

### Data output destinations
- Script outputs go to `ASSETS/restir/` subdirectories
- WandB logs go to `ASSETS/wandb/`

### DigitalTwinCatalog in sys.path
The `DigitalTwinCatalog/neural_pbir/pbir/` directory must be in `sys.path` at
runtime for imports like `opt`, `datasets`, `irtk.*`, and `models.*` to resolve.
All scripts handle this by inserting the path at the top of `__main__`.
