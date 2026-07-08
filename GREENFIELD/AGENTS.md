# GREENFIELD/AGENTS.md

This file is special: it describes the research idea and high-level domain-specific intent of the project, not just directory bookkeeping. See root `AGENTS.md` for the general repo conventions (comments, `.env`, `ASSETS/`, `CONFIGS/`, `OLD/`, brownfield handling). Subdirectories created under `GREENFIELD/` should get their own regular per-directory `AGENTS.md` (purpose + gotchas).

## Research idea

Goal: replace the standard PathTracer backward pass in the **Neural-PBIR** physics-based inverse rendering (PBIR) stage with **PathReSTIR** (spatial sample resampling, via `psdr-jit`) to reduce gradient variance in inverse rendering — particularly for specular materials — without increasing per-pixel sample count.

## Domain-specific intent

- **Neural-PBIR** is a multi-stage inverse-rendering pipeline (Neural Surface Recon → Neural Distillation → PBIR → Evaluation) that recovers geometry, materials, and lighting from posed multi-view images, evaluated on the **Stanford-ORB** benchmark.
- Standard PBIR uses a plain PathTracer for both the forward render (loss) and backward render (gradient). PathTracer gradients are noisy for glossy/specular BRDFs at low sample counts, which limits how aggressively spp can be reduced.
- **PathReSTIR** substitutes a ReSTIR-style spatial reservoir resampling pass into the *backward* (gradient) render only, reusing candidate paths across neighboring pixels so fewer backward samples are needed for a comparable gradient signal.
- The forward render (used for the reported L1 loss / image quality) stays a standard PathTracer — only the backward/gradient pass is where ReSTIR is substituted in.
- Practical framing: this is variance reduction for differentiable rendering, not a new forward-rendering algorithm — the visual outputs should match standard Neural-PBIR; what should improve is gradient quality/convergence at a given (or reduced) sample budget.

For architecture details (integrator wiring, SPP settings per stage, ReSTIR parameters, psdr-jit monkey-patches), see `reference.md` and `REF.md` at the repo root — those stay in place as detailed engineering notes; this file is the one-paragraph "why we're doing this" anchor.
