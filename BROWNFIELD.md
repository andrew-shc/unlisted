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
| DigitalTwinCatalog | partial | vendored; wandb output from our own `run_restir.py`/`run_restir_fwd.py` (added by us into `neural_pbir/pbir/`) redirected to `ASSETS/wandb/` — rest of the vendored tree untouched |
| Stanford-ORB | not started | vendored, eval data |
| MIRReS-ReSTIR_Nerf_mesh | not started | vendored |
| DTUeval-python | not started | vendored |

## Change log
(Add an entry each time a brownfield folder's outputs/config are migrated — what moved, from where, to where, and why.)
