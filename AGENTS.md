# AGENTS.md

Instructions for coding agents working in this repository.

## Commits
- Never auto-commit. Only commit when the user explicitly asks you to, and only after they've had a chance to see the diff.

## Documentation
- Code should be self-documenting: use clear names, and comment generously to explain intent, non-obvious choices, and gotchas — don't be stingy with comments in this repo. This applies to our own (greenfield) code; see below for vendored/brownfield code.
- Keep greenfield files small and flat: each file should read as roughly one function, one class, or a handful of tightly related utility functions — not a module with several unrelated responsibilities bundled together. The goal is that skimming file names and sizes in `GREENFIELD/` should surface most of the codebase's functionality at a glance, without opening files. When a file starts accumulating a second distinct responsibility, split it out into its own file rather than growing it.
- Every subdirectory under `GREENFIELD/` (our own code) should have its own `AGENTS.md` describing that directory's high-level purpose, its subdirectories, and any known bugs/mistakes to avoid. When you create a new directory there, add an `AGENTS.md` to it. When you learn something painful about an existing directory, add it there instead of just fixing the code silently. (`GREENFIELD/AGENTS.md` itself is special — see below.)
- `.env` and `ASSETS/` are both gitignored — never force-add them.

## Secrets (.env)
- Never read, `cat`, or otherwise inspect the contents of `.env` (no `Read` tool, no `cat`/`grep`/`head` via shell).
- New keys may only be **appended** to `.env` programmatically (e.g. `echo "NEW_KEY=value" >> .env`). Never rewrite, reorder, or regenerate the file wholesale.
- Any code you run should load `.env` automatically via the environment (`python-dotenv`, `direnv`, shell `source .env`, etc.) — don't have an agent parse or relay its contents.
- If a codebase/script you're working in doesn't already auto-load `.env`, set that up yourself (e.g. add `load_dotenv()`, wire up `direnv`) rather than reading `.env` manually or asking the user to paste secrets into the conversation.

## Data (ASSETS/) — strict separation, no exceptions
- All data, in the most general sense (evaluation results/JSON, 3D data: PLY/OBJ/GLB, 2D data: EXR/PNG/JPG, checkpoints, logs, renders, etc.), belongs in `./ASSETS/` at the repo root.
- This rule has **no "it's cleaner to keep it local" exception.** Even if a script's own directory structure feels like the natural place for its output (e.g. a vendored repo's own `output/` or `results/` convention), the data still goes under `./ASSETS/`. If you need to preserve that structure for clarity, **mirror the codebase's folder layout inside `ASSETS/`** (e.g. `TensoSDF/run_training.py` → output under `ASSETS/TensoSDF/...`, not `TensoSDF/output/...`). The point is strict separation of data from execution/code paths, always.

## Configs (CONFIGS/)
- `./CONFIGS/` holds configs-as-data only: hyperparameter files, `.gin`/`.yaml`/`.json` config files, and similar.
- `CONFIGS/` is **not** a catch-all for "code that feels auxiliary." Environment setup, preprocessing, training, and evaluation code are all part of the pipeline (main or side) and stay in their normal codebase location — never in `CONFIGS/`, even though they're sometimes used in a config-like or reference way.
- Rule of thumb: if it runs (has logic, does work), it's not a config. If in real doubt, case-by-case exceptions are fine, but default to keeping executed code out of `CONFIGS/`.
- Distinction from `ASSETS/`: `ASSETS/` is pure data (never executed); `CONFIGS/` is config-as-data (declarative parameters, not logic).
- Unlike `ASSETS/`, `CONFIGS/` is **tracked in git**, not ignored.

## Build environment
- Main conda env: `metrology_ir` (python3.11) — activate with `conda activate metrology_ir` before running any pipeline script (`nbir_setup_refined.py`, `nbir_restir_pipeline.py`, etc.).
- `psdr-jit` is not a separate environment — it's a dependency built and installed *inside* `metrology_ir`; its `.so` lands in `$CONDA_PREFIX/lib/python3.11/site-packages/psdr_jit/` (i.e. inside the `metrology_ir` env).
- `./psdr-jit/` lives at the repo root and, unlike other vendored brownfield folders, is **directly tracked** by this repo's own git (no nested `.git`, no gitlink/submodule) since it's actively developed as part of this project rather than pinned as an external clone — see BROWNFIELD.md for provenance and where its prior standalone history was backed up. Its `build/` output and `.so` artifacts are already covered by the root `.gitignore` (extensionless catch-all + `build/`/`*.so` rules), so nothing extra is needed to keep them untracked.
- `conda run` buffers stdout until process exit — use `conda run --no-capture-output` or set `PYTHONUNBUFFERED=1`.
- Rebuilding psdr-jit (run with `metrology_ir` active):
  ```bash
  pip uninstall psdr-jit
  cd psdr-jit
  git submodule update --init --recursive
  pip install --no-build-isolation -ve . --config-settings=cmake.args="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
  python -c "import psdr_jit; print(psdr_jit.__file__)"  # verify
  ```
- Known gotcha: DrJIT CUDA JIT can hang for 1+ hour under ReSTIR at higher candidate counts — not necessarily a crash, just slow compilation.

## Our code (GREENFIELD/) and set-aside code (OLD/)
- `./GREENFIELD/` is the home for all newly-written code that's ours — think of it as `src/` for this repo. Code here is free to call into sibling vendored folders (e.g. drive `TensoSDF/run_training.py` from a `GREENFIELD/` script); the boundary is about ownership, not a hard sandbox.
- `GREENFIELD/AGENTS.md` is special: it describes the research idea and high-level domain-specific intent of the project, not just directory bookkeeping. Subdirectories under `GREENFIELD/` still get their own regular per-directory `AGENTS.md` (purpose + gotchas) as they're created.
- `./OLD/` is a reference cabinet for our own code that's no longer in active use but might be useful again later. When you set something aside instead of deleting it, move it here rather than leaving it cluttering the main tree or silently deleting it.

## Pushing this repo
- `git push git@github.com:andrew-shc/unlisted.git main:metrology_ir`

## Installing psdr-jit (after installing Neural-PBIR)
```bash
pip uninstall psdr-jit
cd psdr-jit
git submodule update --init --recursive
pip install --no-build-isolation -ve . --config-settings=cmake.args="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"

# verify
python -c "import psdr_jit; print(psdr_jit.__file__)"
```
(Same steps as the "Rebuilding psdr-jit" recipe under Build environment above — that one's for reinstalling into an *existing* env after a source change, this one's for a fresh install.)

## Notebooks
- Prefer regular `.py` files over `.ipynb` notebooks for anything new — notebooks make diffs, code review, and reuse (importing a function from another script) harder, and this repo has accumulated enough of them that it's worth reversing the trend.
- Don't force a rewrite of every existing notebook proactively, but when you're touching one anyway (extending it, debugging it, reusing its logic elsewhere), rewrite it into a `.py` file as part of that work rather than adding more to the notebook.
- Exploratory/throwaway one-off analysis is the one case where a notebook is still fine — anything meant to be re-run or built on should be a script.

## .gitignore
- Keep `.gitignore` organized into the labeled sections already there (extensionless-binary catch-all, secrets, data, Python artifacts, vendored examples, data-file-extension safety net, brownfield `results_*/` carve-out) — add new entries to the matching section with a comment, don't append one-off rules at the bottom.
- Most new data should never need a `.gitignore` entry at all: it belongs under `ASSETS/`, which is already wholly ignored. Only add a new pattern here for something that can't live in `ASSETS/` yet (e.g. an untouched brownfield folder's own output directory).

## Greenfield vs. brownfield work
This repo frequently works by `git clone`-ing someone else's research repo into a subdirectory and re-running their own build/pipeline inside it (see e.g. `TensoSDF/`, `nvdiffrecmc/`, `TRELLIS.2/`, `vggt/`, `WNNC/`, `DigitalTwinCatalog/`, `Stanford-ORB/`, `MIRReS-ReSTIR_Nerf_mesh/`, `DTUeval-python/`). This creates a mix of "brownfield" (pre-existing, vendored) and "greenfield" (new, ours) code.

- **Greenfield** (code under `GREENFIELD/`, new scripts, new pipelines you create): follow every convention above from the start — heavy comments, per-directory `AGENTS.md`, `.env` handling, `ASSETS/` for data, `CONFIGS/` for configs. No exceptions, no "migrate later."
- **Brownfield** (existing vendored folders): don't do a disruptive big-bang migration or refactor of someone else's repo just to satisfy convention. Concretely:
  - No per-directory `AGENTS.md` inside vendored folders, and no retroactive re-commenting of their existing code — leave the vendored code as-is.
  - When you touch a brownfield folder (run its build, add a wrapper script, change its output paths), migrate *that touched part* to the new conventions as you go (e.g. redirect its output into `ASSETS/<folder>/...` instead of leaving it in-place) — but don't go further than what you actually touched.
  - Record every such change in `BROWNFIELD.md` at the repo root: what changed, why, and mark that folder's migration status (not started / partially migrated / fully migrated). This is where brownfield documentation lives instead of scattered `AGENTS.md`s.
  - Do not claim a folder is "fully migrated" unless all of its outputs/configs actually follow the `ASSETS/`/`CONFIGS/` split.
