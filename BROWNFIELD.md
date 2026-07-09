# BROWNFIELD.md

Tracks migration of vendored/cloned subdirectories to this repo's conventions (ASSETS/ for data, CONFIGS/ for configs, .env append-only policy). Vendored folders do **not** get their own AGENTS.md and are not re-commented/refactored — this file is where changes to them are documented instead.

Status legend: not started | partial | migrated

| Folder | Status | Notes |
|---|---|---|
| TensoSDF | not started | vendored; writes outputs into its own folders today |
| nvdiffrecmc | not started | vendored |
| TRELLIS.2 | not started | vendored |
| vggt | not started | vendored |
| WNNC | not started | vendored |
| DigitalTwinCatalog | partial | vendored; wandb output from our own `run_restir.py`/`run_restir_fwd.py` (added by us into `neural_pbir/pbir/`) redirected to `ASSETS/wandb/` — rest of the vendored tree untouched. See "DigitalTwinCatalog engineering notes" below for the ReSTIR integration details. |
| Stanford-ORB | not started | vendored, eval data |
| MIRReS-ReSTIR_Nerf_mesh | not started | vendored |
| DTUeval-python | not started | vendored |
| psdr-jit | not started | **not a vendored gitlink like the others** — its nested `.git` was removed on 2026-07-08 so it's a plain, directly-tracked/actively-managed child of this repo instead of a pinned external clone. See change log for history/provenance. |

## DigitalTwinCatalog engineering notes (`run_restir.py` / `opt.py`)

Files we added inside the vendored `neural_pbir/pbir/` tree: `run_restir.py`, `run_restir_fwd.py`, `pbir/configs/restir_template/*.gin`. Since these files physically live in a vendored/brownfield folder, their engineering detail is tracked here rather than in `GREENFIELD/AGENTS.md`.

**Integrator layout** (psdr-jit scene, set up by `run_restir.py`):
```python
scene['integrator']        = Integrator('path',        ...)   # fwd renderC  → L1 loss
scene['integrator_restir'] = Integrator('path_restir', ...)   # bwd renderD  → gradient
scene['integrator_vis']    = Integrator('path',        ...)   # checkpoint renders
```
`PathReSTIRRenderer` is a custom `torch.autograd.Function`: forward → `connector.renderC(scene, fwd_options, integrator_id=0)` (full PathTracer); backward → `connector.renderD(grads, scene, bwd_options, bwd_integrator_id='integrator_restir')` (PathReSTIR). Stage 1 keeps a plain PathTracer `Renderer` (fwd+bwd) — ReSTIR overhead unwarranted for coarse SG init.

**SPP settings:**

| Stage | Renderer | fwd spp | bwd spp |
|---|---|---|---|
| Stage 1 (naive-envmap_sg) | PathTracer | 64 | 64 |
| Stage 2 (basis-envmap_ls) | PathReSTIRRenderer | 16 | 4 |

ReSTIR compensates for the lower bwd spp via spatial resampling:
```python
RESTIR_CONFIG = {'max_depth': 3, 'n_candidates': 4, 'n_neighbors': 5, 'spatial_radius': 10, 'hide_emitters': False}
```

**Gin configs** (`pbir/configs/restir_template/`): `pipeline_restir.gin` (`dataset_class = @NeuralPBIRDataset`); `microfacet_naive-envmap_sg.gin` / `microfacet_basis-envmap_ls.gin` (model + `num_epochs`; `render_opt` overridden in Python).

**psdr-jit monkey-patches** applied at import in `run_restir.py` (same ones used by the archived `test_shape_opt_cornell.py`, see `OLD/it4_greenfield_restir_v1/`):
1. Mesh handler — re-applies `use_face_normal` after every `load_raw()` (connector resets it).
2. Integrator handler — registers the `'path_restir'` type (stock connector only knows `'path'`, `'direct'`, `'field'`, `'collocated'`).

**WandB hooks** (added to `opt.py`): reads `WANDB_API_KEY`/`WANDB_ENTITY`/`WANDB_PROJECT=ReSTIR PBIR` from root `.env`. Logs per-iter scalars (`pbir/loss`, `pbir/image_loss`, `pbir/render_ms`, `pbir/opt_ms`), checkpoint vis strips (`pbir/vis`), final material maps + training GIF, and (from the outer pipeline) stage timings + the Stanford-ORB eval JSON.

**Known issue — gradient-vis hangs:** `compute_dI_dY` in `opt.py` builds a fresh `Scene()` + calls `configure()` every 5 iters; this hung 1+ hour at iter 30 of a ReSTIR-cand8 run (DrJIT CUDA JIT recompiling). Fix: dropped `render_grad` from production runs — loss curves + vis renders are sufficient.

## Change log
(Add an entry each time a brownfield folder's outputs/config are migrated — what moved, from where, to where, and why.)
- 2026-07-08: Moved `psdr-jit` from `/home/ahc/Documents/psdr-jit` (sibling of this repo, own remote `github.com:andrew-shc/psdr-jit`) into `./psdr-jit` at the repo root. Initially kept as a vendored clone with its own `.git`; then, since psdr-jit will be actively developed as part of this project (not just built-and-left-alone like other vendored folders), its `.git` was removed entirely so it's tracked directly by `metrology_ir`'s own git history rather than pinned as a separate repo/gitlink. Its full prior git history (all branches/commits, including uncommitted-at-move-time work on ReSTIR integrators) was backed up to `/home/ahc/Documents/psdr-jit-full-history.bundle` (outside this repo) before deletion, in case that history is ever needed — restore with `git clone /home/ahc/Documents/psdr-jit-full-history.bundle`. `build/` and `*.so` artifacts remain ignored via root `.gitignore`. AGENTS.md's install/rebuild instructions use `cd psdr-jit`. Not yet `git add`ed — left for explicit review/commit.
