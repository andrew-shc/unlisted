#include <misc/Exception.h>
#include <psdr/core/ray.h>
#include <psdr/core/intersection.h>
#include <psdr/core/sampler.h>
#include <psdr/bsdf/bsdf.h>
#include <psdr/emitter/emitter.h>
#include <psdr/shape/mesh.h>
#include <psdr/scene/scene.h>
#include <psdr/sensor/perspective.h>
#include <psdr/utils.h>
#include <psdr/integrator/direct_restir.h>

NAMESPACE_BEGIN(psdr_jit)

DirectReSTIR::DirectReSTIR(int n_candidates, int n_neighbors, int spatial_radius)
    : m_n_candidates(n_candidates),
      m_n_neighbors(n_neighbors),
      m_spatial_radius(spatial_radius) {}

// ---------------------------------------------------------------------------
// Li: minimal single-sample NEE for use by render_primary_edges (base class).
// No spatial reuse.
// ---------------------------------------------------------------------------

SpectrumC DirectReSTIR::Li(const Scene &scene, Sampler &sampler,
                            const RayC &ray, MaskC active) const {
    IntersectionC its = scene.ray_intersect<false>(ray, active);
    active &= its.is_valid();

    SpectrumC result = m_hide_emitters ? zeros<SpectrumC>() : its.Le(active);

    MaskC active_di = active && !its.is_emitter(active);
    BSDFArrayC bsdf_array = its.shape->bsdf();

    PositionSampleC ps = scene.sample_emitter_position<false>(
        its.p, sampler.next_2d<false>(), active_di);
    MaskC active_d = active_di && ps.is_valid;

    Vector3fC wod = ps.p - its.p;
    FloatC dist_sq = squared_norm(wod);
    FloatC dist    = safe_sqrt(dist_sq);
    wod /= dist;

    RayC sray(its.p, wod);
    IntersectionC its1 = scene.ray_intersect<false, false>(sray, active_d);
    active_d &= its1.is_valid() && its1.is_emitter(active_d);
    active_d &= its1.t > dist - ShadowEpsilon;

    FloatC G = abs(dot(its1.n, -wod)) / dist_sq;
    SpectrumC Le  = its1.Le(active_d);
    SpectrumC bv  = bsdf_array->evalC(its, its.sh_frame.to_local(wod), active_d);
    result[active_d] += Le * bv * G * ps.J / ps.pdf;
    return result;
}

SpectrumD DirectReSTIR::Li(const Scene &scene, Sampler &sampler,
                            const RayD &ray, MaskD active) const {
    return zeros<SpectrumD>();
}

// ---------------------------------------------------------------------------
// render_restir: two-pass ReSTIR DI (candidate WRS + parallel spatial reuse).
// ---------------------------------------------------------------------------

SpectrumC DirectReSTIR::render_restir(const Scene &scene, int sensor_id,
                                       int seed) const {
    PSDR_ASSERT_MSG(scene.is_ready(), "Scene must be configured!");
    PSDR_ASSERT_MSG(sensor_id >= 0 && sensor_id < scene.m_num_sensors,
                    "Invalid sensor id!");
    PSDR_ASSERT(m_n_candidates >= 1);

    const RenderOption &opts = scene.m_opts;
    const int W = opts.width, H = opts.height;
    const int num_pixels = W * H;

    if (seed != -1)
        scene.m_samplers[0].seed(arange<UInt64C>(num_pixels) +
                                  static_cast<uint64_t>(seed));

    // ---- Step 1: primary ray intersection for all pixels ----
    auto [fdx, fdy] = meshgrid(arange<FloatC>(W), arange<FloatC>(H));
    Vector2fC samples = (Vector2fC(fdx, fdy) + scene.m_samplers[0].next_2d<false>())
                        / ScalarVector2f(W, H);
    RayC cam_ray = scene.m_sensors[sensor_id]->sample_primary_ray(samples);

    MaskC active(true);
    IntersectionC its = scene.ray_intersect<false>(cam_ray, active);
    active &= its.is_valid();

    SpectrumC result = zeros<SpectrumC>(num_pixels);
    if (!m_hide_emitters)
        result[active] += its.Le(active);

    MaskC active_di = active && !its.is_emitter(active);
    BSDFArrayC bsdf_array = its.shape->bsdf();

    // ---- Step 2: candidate generation → WRS reservoir (per pixel) ----
    //
    // Reservoir fields (all num_pixels wide):
    //   res_contrib : full RGB contribution f(y) at this pixel's shading point
    //   res_p       : 3-D position of selected emitter sample
    //   res_J       : Jacobian at selected sample
    //   res_ws      : running weight sum  Σ w_i = Σ f_hat(x_i)/p(x_i)
    //   res_f       : scalar target f_hat(y) = luminance(res_contrib)
    //   res_M       : candidate count (FloatC to avoid int→float cast later)
    //   res_ok      : whether at least one visible candidate was accepted
    //
    // Target function f_hat = luminance(Le * bsdf * G * J).
    // Unbiased RIS weight:  W = res_ws / (res_M * res_f).

    SpectrumC res_contrib = zeros<SpectrumC>(num_pixels); // FIX 1: sized zeros, not scalar
    Vector3fC res_p  = zeros<Vector3fC>(num_pixels);
    FloatC    res_J  = zeros<FloatC>(num_pixels);
    FloatC    res_ws = zeros<FloatC>(num_pixels);
    FloatC    res_f  = zeros<FloatC>(num_pixels);
    FloatC    res_M  = zeros<FloatC>(num_pixels);          // FIX 2: FloatC, not IntC
    MaskC     res_ok = active_di ^ active_di;              // all-false, num_pixels wide

    for (int c = 0; c < m_n_candidates; c++) {
        PositionSampleC ps = scene.sample_emitter_position<false>(
            its.p, scene.m_samplers[0].next_2d<false>(), active_di);
        MaskC ac = active_di && ps.is_valid;

        Vector3fC wod  = ps.p - its.p;
        FloatC dist_sq = squared_norm(wod);
        FloatC dist    = safe_sqrt(dist_sq);
        wod /= dist;

        RayC sray(its.p, wod);
        IntersectionC its_l = scene.ray_intersect<false>(sray, ac);
        ac &= its_l.is_valid() && its_l.is_emitter(ac);
        ac &= its_l.t > dist - ShadowEpsilon;

        FloatC G   = abs(dot(its_l.n, -wod)) / dist_sq;
        SpectrumC Le = its_l.Le(ac);                               // 0 where !ac
        SpectrumC bv = bsdf_array->evalC(its, its.sh_frame.to_local(wod), ac); // 0 where !ac
        SpectrumC contrib = Le * bv * G * ps.J;                    // already 0 where !ac

        FloatC f_hat = rgb2luminance<false>(contrib);
        FloatC w = select(ps.pdf > Epsilon, f_hat / ps.pdf, FloatC(0.f));

        res_ws += w;
        MaskC accept = (scene.m_samplers[0].next_1d<false>() * res_ws) < w && active_di;

        // FIX 1: use masked() instead of select() for SpectrumC updates —
        // select(MaskC, SpectrumC, SpectrumC) has an ambiguous mask type in
        // Dr.Jit (MaskC vs Array<MaskC,3>); masked() is the pattern used
        // everywhere else in psdr-jit for spectrum writes.
        masked(res_contrib, accept) = contrib;
        masked(res_p,       accept) = ps.p;
        masked(res_J,       accept) = ps.J;
        masked(res_f,       accept) = f_hat;
        res_ok = res_ok || accept;
        res_M += select(active_di, 1.f, 0.f);                      // FIX 2: float literal
    }

    // Materialise reservoir arrays so they can be used as gather sources.
    drjit::eval(res_contrib, res_p, res_J, res_ws, res_f, res_M, res_ok);

    // ---- Step 3: spatial reuse (parallel – all reads from initial buffer) ----
    //
    // For each pixel q, draw K random neighbours s within the search radius.
    // The neighbour's reservoir contributes a virtual candidate with weight:
    //
    //   w_s = f_q(y_s) / f_s(y_s) * res_ws_s
    //       = f_q(y_s) * W_s * M_s          (mathematically equivalent)
    //
    // where f_q(y_s) is the target function re-evaluated at q's shading point,
    // including a visibility test from q to y_s.
    //
    // All K neighbours read from the *pre-reuse* buffer (parallel reuse);
    // sequential reuse would need drjit::eval() between iterations.
    //
    // M is incremented only for neighbours whose sample is actually visible from
    // the current pixel (w_s > 0).  Counting invisible neighbours in M inflates
    // the denominator and causes systematic darkening — see FIX 3.

    IntC pixel_idx = arange<IntC>(num_pixels);
    IntC px = pixel_idx % W;
    IntC py = pixel_idx / W;

    SpectrumC sp_contrib = res_contrib;
    Vector3fC sp_p       = res_p;
    FloatC    sp_J       = res_J;
    FloatC    sp_ws      = res_ws;
    FloatC    sp_f       = res_f;
    FloatC    sp_M       = res_M;                                   // FIX 2: FloatC
    MaskC     sp_ok      = res_ok;

    for (int k = 0; k < m_n_neighbors; k++) {
        // Random offset in [-radius, radius]²
        FloatC u = scene.m_samplers[0].next_1d<false>();
        FloatC v = scene.m_samplers[0].next_1d<false>();
        IntC dx  = IntC((u * 2.f - 1.f) * static_cast<float>(m_spatial_radius));
        IntC dy  = IntC((v * 2.f - 1.f) * static_cast<float>(m_spatial_radius));
        IntC nx  = clamp(px + dx, 0, W - 1);
        IntC ny  = clamp(py + dy, 0, H - 1);
        IntC nbr = ny * W + nx;

        // Gather neighbour's initial (pre-reuse) reservoir.
        Vector3fC nbr_p  = gather<Vector3fC>(res_p,  nbr, active_di);
        FloatC    nbr_J  = gather<FloatC>   (res_J,  nbr, active_di);
        FloatC    nbr_ws = gather<FloatC>   (res_ws, nbr, active_di);
        FloatC    nbr_f  = gather<FloatC>   (res_f,  nbr, active_di);
        FloatC    nbr_M  = gather<FloatC>   (res_M,  nbr, active_di); // FIX 2: FloatC
        MaskC     nbr_ok = gather<MaskC>    (res_ok, nbr, active_di);

        MaskC ank = active_di && nbr_ok;

        // Re-evaluate f_q(y_s): target function at *current* pixel for neighbour sample.
        Vector3fC wod_s  = nbr_p - its.p;
        FloatC dist_sq_s = squared_norm(wod_s);
        FloatC dist_s    = safe_sqrt(dist_sq_s);
        wod_s /= dist_s;

        RayC sray_s(its.p, wod_s);
        IntersectionC its_s = scene.ray_intersect<false>(sray_s, ank);
        ank &= its_s.is_valid() && its_s.is_emitter(ank);
        ank &= its_s.t > dist_s - ShadowEpsilon;

        FloatC G_s     = abs(dot(its_s.n, -wod_s)) / dist_sq_s;
        SpectrumC Le_s  = its_s.Le(ank);                            // 0 where !ank
        SpectrumC bv_s  = bsdf_array->evalC(its, its.sh_frame.to_local(wod_s), ank); // 0 where !ank
        SpectrumC contrib_s = Le_s * bv_s * G_s * nbr_J;           // already 0 where !ank

        FloatC fq_s  = rgb2luminance<false>(contrib_s);

        // Combination weight: ws_s * f_q(y_s) / f_s(y_s) = W_s * M_s * f_q(y_s)
        FloatC w_nbr = select(ank && nbr_f > Epsilon,
                              nbr_ws * fq_s / nbr_f,
                              FloatC(0.f));

        sp_ws += w_nbr;
        MaskC accept_k = (scene.m_samplers[0].next_1d<false>() * sp_ws) < w_nbr
                         && active_di;

        // FIX 1: masked() instead of select() for SpectrumC.
        masked(sp_contrib, accept_k) = contrib_s;
        masked(sp_p,       accept_k) = nbr_p;
        masked(sp_J,       accept_k) = nbr_J;
        masked(sp_f,       accept_k) = fq_s;
        sp_ok = sp_ok || accept_k;

        // FIX 3: only count the neighbour's M when it actually contributes
        // (w_nbr > 0, i.e. the neighbour's sample is visible from this pixel).
        // Counting occluded neighbours in M inflates the denominator and causes
        // systematic darkening proportional to the fraction of invisible neighbours.
        sp_M += select(ank, nbr_M, FloatC(0.f));
    }

    // ---- Step 4: final shading ----
    //
    // W = sp_ws / (sp_M * sp_f)  (sp_M is FloatC — no cast needed)
    // contribution = sp_contrib * W
    //
    // sp_contrib already holds Le * bsdf * G * J evaluated from *this* pixel's
    // shading point, so no additional shadow ray is required here.

    // FIX 2: sp_M is already FloatC — direct division, no FloatC() cast.
    FloatC W_final = select(sp_ok && sp_f > Epsilon,
                            sp_ws / (sp_M * sp_f),
                            FloatC(0.f));

    result[active_di && sp_ok] += sp_contrib * W_final;

    drjit::eval(result);
    return result;
}

NAMESPACE_END(psdr_jit)
