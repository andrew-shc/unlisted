# psdr-jit: Integrator Architecture & Differentiability

Reference doc for extending `psdr-jit/` (directly tracked, actively developed in
this repo — see `BROWNFIELD.md` for the run-time wiring in `run_restir.py`) with
a new custom integrator implementing ReSTIR-style **spatial** reservoir
resampling (no temporal reuse), and for understanding how psdr-jit makes such an
integrator differentiable. Paired with `GREENFIELD/RESTIR.md`, which covers the
ReSTIR algorithm itself independent of psdr-jit's specific style.

Two spatial-ReSTIR integrators already exist in this codebase and are the best
starting templates — see "Existing spatial-ReSTIR integrators" below. Note this
corrects `GREENFIELD/AGENTS.md`'s "Known issues" wording: both are **spatial-only,
no temporal reuse** (confirmed by reading the full source; `direct_restir.h:11`
says so explicitly).

## 1. The `Integrator` base class contract

`include/psdr/integrator/integrator.h:8-30`:

```cpp
PSDR_CLASS_DECL_BEGIN(Integrator,, Object)
public:
    virtual ~Integrator() {}

    SpectrumC renderC(const Scene &scene, int sensor_id = 0, int seed=-1, IntC batch_pix=-1) const;
    SpectrumD renderD(const Scene &scene, int sensor_id = 0, int seed=-1, IntD batch_pix=-1) const;

protected:
    virtual SpectrumC Li(const Scene &scene, Sampler &sampler, const RayC &ray, MaskC active = true) const = 0;
    virtual SpectrumD Li(const Scene &scene, Sampler &sampler, const RayD &ray, MaskD active = true) const = 0;

    virtual void render_primary_edges(const Scene &scene, int sensor_id, SpectrumD &result) const;
    virtual void render_secondary_edges(const Scene &scene, int sensor_id, SpectrumD &result) const {}

    template <bool ad> Spectrum<ad> __render(const Scene &scene, int sensor_id) const;
    template <bool ad> Spectrum<ad> __render_batch(const Scene &scene, int sensor_id, Int<ad> batch_pix) const;
PSDR_CLASS_DECL_END(SamplingIntegrator)
```

**Must implement (pure virtual):**
- `Li(RayC) -> SpectrumC` — primal, non-differentiable per-ray radiance.
- `Li(RayD) -> SpectrumD` — differentiable per-ray radiance.

**May override:**
- `render_primary_edges` — default impl handles primary-visibility/silhouette
  boundary terms generically (`integrator.cpp:186-205`); most integrators
  (including both ReSTIR ones below) just inherit it.
- `render_secondary_edges` — no-op by default; `PathTracer` overrides it for
  indirect-visibility boundary terms (`path.cpp:274-294`). Both ReSTIR
  integrators leave secondary edges as deferred future work.

**Provided for free (non-virtual, `integrator.cpp`):**
- `renderC` (`:12-48`) — seeds `scene.m_samplers[0]` with `num_pixels*spp`
  states, calls `__render<false>`.
- `renderD` (`:51-100`) — seeds three samplers: `m_samplers[0]` with
  `num_pixels*spp` (interior), `[1]` with `num_pixels*sppe` (primary edges),
  `[2]` with `num_pixels*sppse` (secondary edges); calls `__render<true>`, then
  conditionally `render_primary_edges`/`render_secondary_edges`.
- `__render<ad>` (`:103-142`) — the shared sample-generation loop: builds a
  flattened `num_pixels*spp` index array, jitters a camera ray per sample,
  calls `Li`, then `scatter_reduce`s per-color-channel into a per-pixel
  accumulator and divides by `spp`. **Every integrator's forward/backward math
  ultimately funnels through this**, unless (like `PathReSTIR`) it needs a
  custom multi-pass `renderD`.

## 2. Baseline pattern: `PathTracer`

`include/psdr/integrator/path.h` + `src/integrator/path.cpp`. The canonical
example of writing one code path that serves both primal and differentiable
rendering.

A single template, `__Li<ad>` (`path.cpp:34-127`), is instantiated for both
`Li(RayC)` and `Li(RayD)` (lines 25-32) and branches internally with
`if constexpr (ad)` wherever the differentiable and primal math diverge, e.g.
the NEE (direct-lighting) term (`path.cpp:65-77`):

```cpp
if constexpr ( ad ) {
    bsdf_val2 = bsdf_array->evalD(its, wo_local, active_direct);
    bsdf_val2 *= G_val*ps.J/ps.pdf;
    pdf1 = bsdf_array->pdfD(its, wo_local, active_direct);
    pdf1 *= detach(G_val);
} else {
    bsdf_val2 = bsdf_array->evalC(its, wo_local, active_direct);
    bsdf_val2 *= G_val*ps.J/ps.pdf;
    pdf1 = bsdf_array->pdfC(its, wo_local, active_direct);
    pdf1 *= G_val;
}
```

Note `detach(G_val)` in the `ad` branch (also `pdf0 = bs.pdf*detach(G_val)` at
line 107 for the BSDF-sampling term): the geometry term used only to normalize
the Monte-Carlo estimator is treated as a constant w.r.t. AD — a standard
detached-pdf trick that recurs throughout psdr-jit (see §4).

`path.cpp` also implements `eval_secondary_edge<ad>` (`:171-270`) for indirect
boundary/silhouette reparameterization. Its `ad` branch ends with an idiom
that reappears in ReSTIR's gradient pass:

```cpp
SpectrumD result = (SpectrumD(value0)*dot(Vector3fD(n), u2)) & valid;
return { select(valid, sds.pixel_idx, -1), result - detach(result) };  // path.cpp:264-265
```

`x - detach(x)` is numerically zero in the primal but differentiates to
`x'` — a way to splice a pure-gradient contribution into an accumulator
without double-adding the value. Name this now; it's the house style.

## 3. Existing spatial-ReSTIR integrators

Both replace only the **depth-0 direct-lighting term** with reservoir
resampling; depth-1+ indirect bounces fall back to ordinary MIS path tracing.

### `DirectReSTIR` — primal-only, simplest reference for the algorithm

`include/psdr/integrator/direct_restir.h` (31 lines) + `src/integrator/direct_restir.cpp`
(270 lines). Not wired to `renderD` — `Li(RayD)` is a literal stub:

```cpp
SpectrumD DirectReSTIR::Li(const Scene &scene, Sampler &sampler,
                            const RayD &ray, MaskD active) const {
    return zeros<SpectrumD>();
}
```

Its header comment is explicit about scope: `// ReSTIR DI: candidate
generation + spatial reuse, no temporal reuse.` Read this file first if you
just want the candidate-generation + spatial-reuse math without the AD
complexity layered on top (§4 below).

### `PathReSTIR` — differentiable, the one wired into this project's PBIR backward pass

`include/psdr/integrator/path_restir.h` (58 lines) + `src/integrator/path_restir.cpp`
(529 lines, heavily commented — read it directly for the full picture).

**Reservoir representation** — not a `struct Reservoir`, but four parallel
per-pixel, non-AD arrays (`path_restir.h:50-54`):

```cpp
mutable Vector3fC m_res_p;   // selected emitter sample world position
mutable FloatC    m_res_J;   // area-form Jacobian at selected sample
mutable FloatC    m_res_W;   // unbiased RIS weight W = ws/(M*f_hat)
mutable MaskC     m_res_ok;  // valid reservoir per pixel
```

**Candidate generation** (weighted reservoir sampling, `path_restir.cpp:212-245`)
— vectorized across all pixels simultaneously via `select`/`masked`, not a
per-pixel scalar loop:

```cpp
res_ws += w;
MaskC accept = (scene.m_samplers[0].next_1d<false>() * res_ws) < w && active_di;
masked(res_contrib, accept) = contrib;
masked(res_p, accept) = ps.p;
...
```

**Spatial reuse** (`path_restir.cpp:249-309`) — samples a neighbor offset in a
`[-spatial_radius, +spatial_radius]` square, `gather`s that neighbor's reservoir
from a frozen pre-reuse snapshot (`res_p/J/ws/f/M/ok` — never mutated during
the neighbor loop, which is what makes this parallel and race-free instead of
sequential), re-evaluates the neighbor's sample under the *current* pixel's
BSDF/geometry, and feeds it into this pixel's reservoir:

```cpp
FloatC fq_s  = rgb2luminance<false>(contrib_s);          // target pdf re-evaluated HERE
FloatC w_nbr = select(ank && nbr_f > Epsilon, nbr_ws * fq_s / nbr_f, FloatC(0.f));
sp_ws += w_nbr;
MaskC accept_k = (scene.m_samplers[0].next_1d<false>() * sp_ws) < w_nbr && active_di;
```

Final unbiased weight and cache for the backward pass (`:311-322`):
`W_final = sp_ws/(sp_M*sp_f)`, stored into `m_res_p/J/W/ok`. See
`GREENFIELD/RESTIR.md` for what this combination rule means algorithmically
and — importantly — where it sits on the bias/cost tradeoff relative to the
paper's fully unbiased correction.

## 4. Making it differentiable: the adjoint/backward pass

**Mechanism**: Dr.Jit's tape-based reverse-mode AD. `FloatD`/`SpectrumD` are
`DiffArray<CUDAArray<T>>` (`include/psdr/types.h:25,39-40,135-136`). The raw
primitives (`enable_grad`, `backward`, `grad`) are exercised by a smoke test at
`src/psdr.cpp:72-83`:

```cpp
FloatD a = arange<FloatD>(10);
enable_grad(a);
FloatD b = a * 2.f;
backward(b);
std::cout << "grad: " << grad(a) << std::endl;
```

psdr-jit's own integrators **never call `backward()` themselves** — they only
build the differentiable graph inside `renderD` (via `FloatD`/`SpectrumD`
arithmetic and `evalD`/`sampleD`/`pdfD` calls); the actual `dr.backward(loss)`
happens later, in the Python training loop (`DigitalTwinCatalog/neural_pbir/
pbir/run_restir.py`, per `BROWNFIELD.md`).

**`PathReSTIR::renderD`'s 4-pass structure** (`path_restir.cpp:479-527`, its own
comments spell this out at `path_restir.h:17-23`):

1. `render_restir_path` — primal render + reservoir cache (§3), plain `FloatC`, no AD.
2. `render_restir_grad` — depth-0 ReSTIR gradient (the hand-derived core, below).
3. Inherited `__render<true>` — depth-1+ interior gradient via `Li(RayD)`, **after
   re-seeding** `m_samplers[0]` with `num_pixels*spp` states
   (`path_restir.cpp:511-517`) — passes 1-2 only seeded it with `num_pixels`
   states, but `__render<true>` needs one state per *sample*, not per pixel.
   This is the sampler-size bug/fix `GREENFIELD/AGENTS.md` already flags.
4. Inherited `render_primary_edges` — silhouette boundary term.

**`render_restir_grad`** (`path_restir.cpp:402-473`) is where the resampling
decision gets spliced into the AD graph:

```cpp
// Shadow test from the PRIMAL shading point (detached) so binary visibility
// doesn't create a discontinuity in the AD graph:
MaskC active_di_p = detach(active_di);
RayC shadow_ray(detach(its.p), detach(wod));
IntersectionC its_l = scene.ray_intersect<false>(shadow_ray, active_di_p);
...
SpectrumD bsdf_ad  = bsdf_array->evalD(its, wo_local, active_di);   // the ONLY AD-tracked op
SpectrumD Le_det   = SpectrumD(its_l.Le(active_di_p2));             // detached
SpectrumD J_det    = SpectrumD(m_res_J);                            // detached
SpectrumD contrib_ad = Le_det * bsdf_ad * G_ad * J_det;
SpectrumD grad_di = contrib_ad * SpectrumD(m_res_W);                // detached RIS weight
grad_di -= detach(grad_di);                                        // strip primal (§2 idiom)
```

**The rule, stated once**: the stochastic resampling decision — which
candidate/neighbor wins the reservoir, i.e. `m_res_p`/`m_res_J`/`m_res_W` — is
computed in a plain, non-AD pass and cached as a **detached constant**. Only
the physically-based re-evaluation of BSDF × geometry at that frozen winning
sample is re-run through the AD-tracked `evalD`, so gradients flow correctly
w.r.t. scene parameters (geometry, BSDF) but not through the discrete
accept/reject resampling logic itself (which has no useful gradient — it's a
stochastic argmax over a stream). This is the same detach-and-recompute
pattern `PathTracer::eval_secondary_edge`/`render_primary_edges` use for
visibility discontinuities (§2) — one house style, applied to two different
kinds of non-differentiable decisions (visibility, resampling).

**Practical guidance for a new integrator**: use `FloatD`/`SpectrumD` ops
(`evalD`/`sampleD`/`pdfD`) for anything that should carry a gradient; `detach()`
anything that's a sampling decision, resampling weight, or visibility test;
use the `x - detach(x)` idiom when injecting a gradient-only term into an
accumulator that already received the primal value elsewhere.

## 5. Wiring a new integrator into the build + Python API

1. Header: `include/psdr/integrator/your_integrator.h`, subclass `Integrator`
   via `PSDR_CLASS_DECL_BEGIN(YourIntegrator, final, Integrator)` — model on
   `path_restir.h`.
2. Source: `src/integrator/your_integrator.cpp`.
3. Add to `PSDR_SOURCE_FILES` in `CMakeLists.txt` (list starts `:54`;
   `path.cpp`/`path_restir.cpp` are at `:152`/`:155`).
4. `#include` the header in `src/psdr.cpp` next to the other integrator
   includes (`:50-56`).
5. Register the pybind11 class, modeled on the `PathReSTIR` block
   (`psdr.cpp:443-459`):

```cpp
py::class_<PathReSTIR, Integrator>(m, "PathReSTIR")
    .def(py::init<int, int, int, int>(),
         "max_depth"_a = 3, "n_candidates"_a = 4,
         "n_neighbors"_a = 5, "spatial_radius"_a = 30)
    .def("render_restir_path", &PathReSTIR::render_restir_path, ...)
    .def("renderD", &PathReSTIR::renderD, ...)
    .def_readwrite("n_candidates", &PathReSTIR::m_n_candidates)
    ...
```

**Subtlety worth flagging**: `Integrator::renderD` is declared non-virtual
(`integrator.h:13`). `PathReSTIR::renderD` (`path_restir.h:24-25`) is therefore
a same-named *shadowing* method, not a C++ `override` — it works only because
`psdr.cpp:451` explicitly re-binds `&PathReSTIR::renderD` in its own
`py::class_` block instead of relying on the inherited `Integrator` binding
(`:421-423`). A new integrator needing a custom multi-pass `renderD` must do
the same.

## 6. What not to touch: `psdr-jit/cuda/`

```
cuda/psdr_jit.cu        # OptiX raygen/miss/closesthit — GPU BVH ray-triangle intersection
cuda/psdr_jit.cpp       # stub, not part of the runtime path
cuda/psdr_jit.h         # OptiX SBT structs shared with psdr_jit.cu
cuda/host/{knn,NEE,util}.cuh   # declarations for adaptive-quadrature (NEE) light-sampling
cuda/kernel/{NEE,util}.cu      # only util.cu is actually compiled (CMakeLists.txt build list)
cuda/kernel/knn.cu             # not compiled, orphaned
```

None of this is ReSTIR-specific — `NEE.cuh` backs `src/core/AQ_distrb.cpp`, an
unrelated adaptive-quadrature emitter-sampling guiding structure; `psdr_jit.cu`
is the general-purpose BVH backend every integrator's `scene.ray_intersect<ad>()`
uses. **A new spatial-ReSTIR integrator does not need a hand-written `.cu`
file** — all of `path_restir.cpp`'s reservoir/resampling logic is vectorized
Dr.Jit C++ (`gather`/`scatter_reduce`/`select`/`masked`), JIT-compiled to CUDA
kernels on the fly at `dr::eval()` time.

## 7. Scene/Sensor/Sampler quick reference

| Header | What a new integrator needs from it |
|---|---|
| `include/psdr/scene/scene.h` | `Scene`: `m_meshes`/`m_bsdfs`/`m_emitters`/`m_sensors`; `m_samplers` (array of 3: interior/primary-edge/secondary-edge); `ray_intersect<ad>()` (`:39`); `sample_emitter_position<ad>()` (`:45`). |
| `include/psdr/sensor/sensor.h` | Abstract `Sensor`: `sample_primary_ray`, `sample_direct`, `sample_primary_edge` — spawns camera rays. |
| `include/psdr/core/sampler.h` | `Sampler`: PCG32-backed, `seed(UInt64C)`, `next_1d/2d/3d/nd<ad>()`. |
| `include/psdr/types.h:217-227` | `RenderOption`: `width, height, spp, sppe, sppse, log_level`, read via `scene.m_opts`. |
