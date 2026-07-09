# ReSTIR: Domain Knowledge & Spatial Resampling

Summary of the ReSTIR algorithm — Bitterli, Wyman, Pharr, Shirley, Lefohn,
Wyatt, *"Spatiotemporal Reservoir Resampling for Real-Time Ray Tracing with
Dynamic Direct Lighting"*, SIGGRAPH 2020 — focused on the **spatial**
resampling half of the algorithm (this project has no temporal reuse). Code
examples here are generic CUDA, independent of psdr-jit's Dr.Jit-vectorized
style; for how this maps onto psdr-jit's actual (differentiable) integrator,
see `GREENFIELD/PSDR_JIT.md`.

## 1. Motivating problem

Direct lighting with many emitters/candidates: importance sampling from a
cheap source distribution (e.g. uniform-over-lights) has high variance when
the true integrand (emitted radiance × BSDF × geometry × visibility) is
spiky, and evaluating every candidate is too expensive — each one needs a
shadow ray. ReSTIR's goal: spend that budget on generating many *cheap*
candidates, but shade (and pay for visibility on) only one per pixel, chosen
so the final estimator stays unbiased.

## 2. Resampled Importance Sampling (RIS) recap

Generate `M` candidates `x_1..x_M` from a cheap source pdf `p` (e.g. no
visibility check). Weight each by `w_i = p_hat(x_i) / p(x_i)`, where `p_hat`
is a cheap approximation of the true integrand (e.g. unshadowed contribution).
Pick one candidate `y` with probability proportional to `w_i`. The unbiased
contribution weight for the winner is:

```
W_y = (1 / p_hat(y)) * (1/M) * sum_i(w_i)
```

Final estimator: `f(y) * W_y`, where `f` is the *true* integrand (visibility
included) — evaluated only once, at the winning sample.

## 3. Weighted Reservoir Sampling (WRS) — the streaming mechanism

RIS needs to pick 1-of-M proportional to weight. WRS does this in a single
pass with O(1) memory (no need to store all M candidates), which is also what
makes reservoirs cheap to combine later (§4):

```cpp
struct Reservoir {
    float3 y;      // selected sample (e.g. point on a light)
    float  wsum;   // running sum of resampling weights
    float  M;      // number of candidates seen so far
    float  W;      // unbiased contribution weight, valid after finalize
};

__device__ void reservoirUpdate(Reservoir &r, float3 xi, float wi, float rnd) {
    r.wsum += wi;
    r.M    += 1.0f;
    if (rnd < wi / r.wsum) r.y = xi;   // classic streaming accept test
}

__device__ void reservoirFinalize(Reservoir &r, float pHat_y) {
    r.W = (pHat_y > 0.0f) ? r.wsum / (r.M * pHat_y) : 0.0f;
}
```

A per-pixel candidate-generation kernel calls `reservoirUpdate` once per
candidate (drawing a light sample, computing `p_hat` from the unshadowed
contribution, no shadow ray yet), then `reservoirFinalize` once at the end —
this produces one reservoir per pixel from M cheap candidates.

## 4. Spatial resampling — the core mechanism

One reservoir per pixel from only M candidates is still noisy. Neighboring
pixels (similar surface point, normal, material) likely picked good samples
too — reuse their winning sample as an *additional* candidate for this pixel,
without re-paying the full candidate-generation cost.

**Algorithm** (spatial-only, no temporal):
1. Each pixel already has its own reservoir `R_q` (optionally after a
   "visibility reuse" pass — trace one shadow ray for the pixel's own current
   winner and zero its weight if occluded, before spatial reuse begins; this
   reduces bias from combining occluded/unoccluded samples across geometric
   discontinuities).
2. For each of `K` spatial neighbors within a pixel-radius `r`: take the
   neighbor's stored sample `R_k.y`, **re-evaluate the target pdf `p_hat_q` at
   the *current* pixel** (this pixel's BSDF/normal/position, not the
   neighbor's — the crux of why spatial reuse needs re-weighting, not a plain
   merge), and feed it into the current pixel's reservoir via the same
   `reservoirUpdate`, with weight `w_k = p_hat_q(R_k.y) * R_k.W * R_k.M`.
3. Sum `M` across all combined reservoirs; finalize `W` with `p_hat` evaluated
   at the combined winner.

```cpp
__global__ void spatialReuseKernel(const Reservoir *reservoirsIn, Reservoir *reservoirsOut,
                                    const GBuffer *gbuf, int width, int height,
                                    int numNeighbors, int radius, unsigned seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;

    Reservoir combined = {};
    RngState rng = initRng(idx, seed);

    // seed with this pixel's own reservoir
    const Reservoir &self = reservoirsIn[idx];
    reservoirUpdate(combined, self.y, targetPdf(gbuf, idx, self.y) * self.W * self.M, rngNext(rng));
    float mSum = self.M;

    for (int n = 0; n < numNeighbors; ++n) {
        int2 off = sampleDiskOffset(rng, radius);
        int nIdx = clamp(y + off.y, 0, height - 1) * width + clamp(x + off.x, 0, width - 1);
        if (!similarSurface(gbuf, idx, nIdx)) continue;      // normal/depth reject

        const Reservoir &nbr = reservoirsIn[nIdx];           // read frozen input buffer
        float pHatHere = targetPdf(gbuf, idx, nbr.y);         // re-evaluated at THIS pixel
        reservoirUpdate(combined, nbr.y, pHatHere * nbr.W * nbr.M, rngNext(rng));
        mSum += nbr.M;
    }
    combined.M = mSum;
    reservoirFinalize(combined, targetPdf(gbuf, idx, combined.y));
    reservoirsOut[idx] = combined;   // write to a SEPARATE buffer — see below
}
```

`reservoirsIn`/`reservoirsOut` must be separate buffers: every neighbor read
must come from one frozen pre-reuse snapshot, or reuse becomes a race (a pixel
processed later would read another pixel's already-updated, post-reuse
reservoir instead of its original one).

## 5. Unbiasedness — why naive combination is biased, and the correction

Reweighting a neighbor's sample by `p_hat_q` (this pixel's target pdf) alone
is only a *valid* resampling step if this pixel's domain could really have
produced that sample — but the combination above accumulates `M` from every
neighbor unconditionally, which implicitly assumes every neighbor's own
domain also supports whatever sample eventually wins. Near occlusion
boundaries or normal/material discontinuities that assumption is false: a
sample unoccluded and valid at neighbor `k` may be occluded or off-BSDF-lobe
at pixel `q`, or vice versa. Treating all `K+1` domains as interchangeable
under- or over-counts the true candidate pool for the winning sample —
visible bias (energy loss/gain, halos near discontinuities), not just extra
variance.

**The paper's correction** (generalized balance heuristic): after the
combined reservoir has picked its final winner `y`, go back through every
pixel `r` that contributed (`r ∈ {q} ∪ neighbors`) and check whether `r`'s
domain actually supports `y` — i.e. re-evaluate `p_hat_r(y)`, which for direct
lighting means re-testing *visibility from `r`'s shading point to `y`* (one
extra shadow ray per contributing neighbor, evaluated against the single
final winner, not against every candidate). Define:

```
Z = sum over r of ( p_hat_r(y) > 0 ? M_r : 0 )
W = (1 / p_hat_q(y)) * (wsum / Z)          // replaces wsum / (M * p_hat_q(y))
```

`Z` counts only the candidate budget of pixels whose domain genuinely supports
the winning sample — this is what restores unbiasedness.

```cpp
__device__ float unbiasedZ(const Reservoir *reservoirs, const GBuffer *gbuf,
                            int selfIdx, const int *neighborIdx, int numNeighbors,
                            float3 y, float selfM) {
    float Z = (targetPdf(gbuf, selfIdx, y) > 0.0f) ? selfM : 0.0f;
    for (int n = 0; n < numNeighbors; ++n) {
        int r = neighborIdx[n];
        if (targetPdf(gbuf, r, y) > 0.0f) Z += reservoirs[r].M;  // extra visibility recheck
    }
    return Z;
}
// combined.W = (pHat_q(combined.y) > 0) ? combined.wsum / Z : 0.0f;
```

**Cost/accuracy tradeoff**: the correction costs up to `K+1` additional shadow
rays per pixel (one recheck per contributor, against the single final
winner). Real-time and many practical implementations skip it and use
`Z ≈ sum(M_r)` unconditionally — i.e. the cheap path in the `spatialReuseKernel`
above — accepting a small, discontinuity-localized bias in exchange for
avoiding the extra rays; the original spatiotemporal algorithm also relies on
temporal accumulation to further mask it (not applicable to a spatial-only
setting like this one).

**This repo's implementation** (`psdr-jit/src/integrator/path_restir.cpp:249-309`,
detailed in `GREENFIELD/PSDR_JIT.md` §3) takes the cheap path: it reweights
each neighbor's sample by the target pdf re-evaluated at the current pixel and
accumulates `M` unconditionally (`sp_M += select(ank, nbr_M, ...)`), with no
second visibility recheck of the final winner against each contributing
neighbor's domain. It's the practical/biased combination rule, not the fully
unbiased one above.

## 6. Practical parameters & tradeoffs

- **M** (candidates/pixel): more candidates → lower variance in the initial
  reservoir, linear cost.
- **K** (spatial neighbors) and **radius r**: more/farther neighbors → more
  reuse (lower variance) but higher chance of surface-discontinuity mismatch
  (more bias under the cheap combination rule) and more gather traffic.
- **M-capping**: clamping accumulated `M` prevents a single old/confident
  reservoir from dominating repeated reuse passes — mostly a spatiotemporal
  (multi-frame) concern, less relevant to a single-frame spatial-only pass.

## 7. Cross-reference

- `GREENFIELD/PSDR_JIT.md` — how this maps onto psdr-jit's actual integrators
  (`DirectReSTIR`, `PathReSTIR`), vectorized across all pixels via Dr.Jit
  rather than per-pixel CUDA threads, and how `PathReSTIR` is made
  differentiable.
- `ASSETS/RESTIR_SUMMARY.md` — a concrete PBIR run log using this algorithm.
- `OLD/it4_greenfield_restir_v1/restir_comparison.ipynb` — prior DI-ReSTIR
  benchmarking notes (PSNR-vs-time, WRS candidate counts).
