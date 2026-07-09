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
| psdr-jit | not started | **not a vendored gitlink like the others** — its nested `.git` was removed on 2026-07-08 so it's a plain, directly-tracked/actively-managed child of this repo instead of a pinned external clone. See change log for history/provenance. |

## Change log
(Add an entry each time a brownfield folder's outputs/config are migrated — what moved, from where, to where, and why.)
- 2026-07-08: Moved `psdr-jit` from `/home/ahc/Documents/psdr-jit` (sibling of this repo, own remote `github.com:andrew-shc/psdr-jit`) into `./psdr-jit` at the repo root. Initially kept as a vendored clone with its own `.git`; then, since psdr-jit will be actively developed as part of this project (not just built-and-left-alone like other vendored folders), its `.git` was removed entirely so it's tracked directly by `metrology_ir`'s own git history rather than pinned as a separate repo/gitlink. Its full prior git history (all branches/commits, including uncommitted-at-move-time work on ReSTIR integrators) was backed up to `/home/ahc/Documents/psdr-jit-full-history.bundle` (outside this repo) before deletion, in case that history is ever needed — restore with `git clone /home/ahc/Documents/psdr-jit-full-history.bundle`. `build/` and `*.so` artifacts remain ignored via root `.gitignore`. AGENTS.md's install/rebuild instructions use `cd psdr-jit`. Not yet `git add`ed — left for explicit review/commit.
