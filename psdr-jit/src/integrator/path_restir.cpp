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
#include <psdr/integrator/path_restir.h>

NAMESPACE_BEGIN(psdr_jit)

PathReSTIR::PathReSTIR(int max_depth, int n_candidates,
                       int n_neighbors, int spatial_radius)
    : m_max_depth(max_depth),
      m_n_candidates(n_candidates),
      m_n_neighbors(n_neighbors),
      m_spatial_radius(spatial_radius) {}

// ---------------------------------------------------------------------------
// Li(RayC): minimal single-sample NEE — used by render_primary_edges.
// ---------------------------------------------------------------------------

SpectrumC PathReSTIR::Li(const Scene &scene, Sampler &sampler,
                          const RayC &ray, MaskC active) const {
    IntersectionC its = scene.ray_intersect<false>(ray, active);
    active &= its.is_valid();

    SpectrumC result = m_hide_emitters ? zeros<SpectrumC>() : its.Le(active);
    MaskC active_di = active && !its.is_emitter(active);
    BSDFArrayC bsdf_array = its.shape->bsdf();

    PositionSampleC ps = scene.sample_emitter_position<false>(
        its.p, sampler.next_2d<false>(), active_di);
    MaskC ac = active_di && ps.is_valid;

    Vector3fC wod = ps.p - its.p;
    FloatC dist_sq = squared_norm(wod);
    FloatC dist    = safe_sqrt(dist_sq);
    wod /= dist;

    RayC sray(its.p, wod);
    IntersectionC its1 = scene.ray_intersect<false, false>(sray, ac);
    ac &= its1.is_valid() && its1.is_emitter(ac) && its1.t > dist - ShadowEpsilon;

    FloatC G   = abs(dot(its1.n, -wod)) / dist_sq;
    SpectrumC Le = its1.Le(ac);
    SpectrumC bv = bsdf_array->evalC(its, its.sh_frame.to_local(wod), ac);
    result[ac] += Le * bv * G * ps.J / ps.pdf;
    return result;
}

// ---------------------------------------------------------------------------
// Li(RayD): depth-1+ differentiable path tracing.
// Identical to PathTracer::__Li<true> but skips the depth-0 NEE block
// (that contribution is handled by render_restir_grad).
// ---------------------------------------------------------------------------

SpectrumD PathReSTIR::Li(const Scene &scene, Sampler &sampler,
                          const RayD &ray, MaskD active) const {
    if (m_max_depth < 1) return zeros<SpectrumD>();

    RayD curr_ray(ray);
    IntersectionD its = scene.ray_intersect<true>(curr_ray, active);
    active &= its.is_valid();

    SpectrumD throughput(1.f);
    // Depth-0 emitter hit: include only if not hiding emitters.
    SpectrumD result = m_hide_emitters ? zeros<SpectrumD>() : its.Le(active);

    // Depth 0 indirect: BSDF sample to find the depth-1 intersection.
    // We don't do NEE here — that was already handled by render_restir_grad.
    BSDFArrayD bsdf_arr0 = its.shape->bsdf();
    BSDFSampleD bs0 = bsdf_arr0->sampleD(its, sampler.next_nd<3, true>(), active);
    active &= bs0.is_valid;

    curr_ray = RayD(its.p, its.sh_frame.to_world(bs0.wo));
    IntersectionD its1 = scene.ray_intersect<true, true>(curr_ray, active);
    active &= its1.is_valid();

    {
        Vector3fD wo = its1.p - its.p;
        wo /= its1.t;
        FloatD cos_val = dot(its1.n, -wo);
        FloatD G_val   = abs(cos_val) / sqr(its1.t);
        FloatD J       = select(~its1.is_valid(), FloatD(1.f), its1.J);
        G_val          = select(~its1.is_valid(), FloatD(1.f), G_val);
        FloatD pdf0    = bs0.pdf * detach(G_val);
        SpectrumD bsdf_val = select(detach(its1.t) < Epsilon, SpectrumD(0.f),
                                    bsdf_arr0->evalD(its, its.sh_frame.to_local(wo), active)
                                    * G_val * J / pdf0);
        FloatD w2 = mis_weight<true>(pdf0,
                        scene.emitter_position_pdf<true>(its.p, its1, active));
        throughput *= bsdf_val;
        result[active] += its1.Le(active) * throughput * w2;
    }

    its = its1;

    // Depth 1 onward: full MIS path tracing.
    for (int depth = 1; depth < m_max_depth; ++depth) {
        BSDFArrayD bsdf_array = its.shape->bsdf();

        // NEE
        {
            PositionSampleD ps = scene.sample_emitter_position<true>(
                its.p, sampler.next_2d<true>(), active);
            MaskD active_di = active && ps.is_valid && !its.is_emitter(active);

            Vector3fD wod  = ps.p - its.p;
            FloatD dist_sq = squared_norm(wod);
            FloatD dist    = safe_sqrt(dist_sq);
            wod /= dist;

            RayD ray1(its.p, wod);
            IntersectionD its_l = scene.ray_intersect<true, true>(ray1, active_di);
            active_di &= its_l.is_valid()
                      && (its_l.t > dist - ShadowEpsilon)
                      && its_l.is_emitter(active_di);

            FloatD cos_v   = dot(its_l.n, -wod);
            FloatD G_val   = abs(cos_v) / dist_sq;
            Vector3fD wo_l = its.sh_frame.to_local(wod);
            SpectrumD bv2  = bsdf_array->evalD(its, wo_l, active_di);
            bv2 *= G_val * ps.J / ps.pdf;
            FloatD pdf1    = bsdf_array->pdfD(its, wo_l, active_di) * detach(G_val);
            active_di &= neq(pdf1, FloatD(0.f));
            FloatD w1 = mis_weight<true>(ps.pdf, pdf1);
            result[active_di] += throughput * its_l.Le(active) * bv2 * w1;
        }

        // BSDF sample
        {
            BSDFSampleD bs = bsdf_array->sampleD(its, sampler.next_nd<3, true>(), active);
            active &= bs.is_valid;

            curr_ray = RayD(its.p, its.sh_frame.to_world(bs.wo));
            IntersectionD its_next = scene.ray_intersect<true, true>(curr_ray, active);
            active &= its_next.is_valid();

            Vector3fD wo   = its_next.p - its.p;
            wo /= its_next.t;
            FloatD cos_v   = dot(its_next.n, -wo);
            FloatD G_val   = abs(cos_v) / sqr(its_next.t);
            FloatD J       = select(~its_next.is_valid(), FloatD(1.f), its_next.J);
            G_val          = select(~its_next.is_valid(), FloatD(1.f), G_val);
            FloatD pdf0    = bs.pdf * detach(G_val);
            SpectrumD bsdf_val = select(detach(its_next.t) < Epsilon, SpectrumD(0.f),
                                        bsdf_array->evalD(its, its.sh_frame.to_local(wo), active)
                                        * G_val * J / pdf0);
            FloatD w2 = mis_weight<true>(pdf0,
                            scene.emitter_position_pdf<true>(its.p, its_next, active));
            throughput *= bsdf_val;
            result[active] += its_next.Le(active) * throughput * w2;
            its = its_next;
        }

        drjit::eval(result, throughput, active);
    }

    drjit::eval(result);
    return result;
}

// ---------------------------------------------------------------------------
// render_restir_path: two-pass ReSTIR DI at depth 0, then MIS path for 1+.
// ---------------------------------------------------------------------------

SpectrumC PathReSTIR::render_restir_path(const Scene &scene, int sensor_id,
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

    // ---- Step 1: primary ray intersection ----
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

    // ---- Step 2: WRS candidate generation ----
    SpectrumC res_contrib = zeros<SpectrumC>(num_pixels);
    Vector3fC res_p       = zeros<Vector3fC>(num_pixels);
    FloatC    res_J       = zeros<FloatC>(num_pixels);
    FloatC    res_ws      = zeros<FloatC>(num_pixels);
    FloatC    res_f       = zeros<FloatC>(num_pixels);
    FloatC    res_M       = zeros<FloatC>(num_pixels);
    MaskC     res_ok      = active_di ^ active_di;  // all-false

    for (int c = 0; c < m_n_candidates; ++c) {
        PositionSampleC ps = scene.sample_emitter_position<false>(
            its.p, scene.m_samplers[0].next_2d<false>(), active_di);
        MaskC ac = active_di && ps.is_valid;

        Vector3fC wod  = ps.p - its.p;
        FloatC dist_sq = squared_norm(wod);
        FloatC dist    = safe_sqrt(dist_sq);
        wod /= dist;

        RayC sray(its.p, wod);
        IntersectionC its_l = scene.ray_intersect<false>(sray, ac);
        ac &= its_l.is_valid() && its_l.is_emitter(ac)
           && its_l.t > dist - ShadowEpsilon;

        FloatC G      = abs(dot(its_l.n, -wod)) / dist_sq;
        SpectrumC Le  = its_l.Le(ac);
        SpectrumC bv  = bsdf_array->evalC(its, its.sh_frame.to_local(wod), ac);
        SpectrumC contrib = Le * bv * G * ps.J;

        FloatC f_hat = rgb2luminance<false>(contrib);
        FloatC w     = select(ps.pdf > Epsilon, f_hat / ps.pdf, FloatC(0.f));

        res_ws += w;
        MaskC accept = (scene.m_samplers[0].next_1d<false>() * res_ws) < w
                       && active_di;

        masked(res_contrib, accept) = contrib;
        masked(res_p,       accept) = ps.p;
        masked(res_J,       accept) = ps.J;
        masked(res_f,       accept) = f_hat;
        res_ok  = res_ok || accept;
        res_M  += select(active_di, FloatC(1.f), FloatC(0.f));
    }

    drjit::eval(res_contrib, res_p, res_J, res_ws, res_f, res_M, res_ok);

    // ---- Step 3: spatial reuse (parallel, reads pre-reuse buffer) ----
    IntC pixel_idx = arange<IntC>(num_pixels);
    IntC px = pixel_idx % W;
    IntC py = pixel_idx / W;

    SpectrumC sp_contrib = res_contrib;
    Vector3fC sp_p       = res_p;
    FloatC    sp_J       = res_J;
    FloatC    sp_ws      = res_ws;
    FloatC    sp_f       = res_f;
    FloatC    sp_M       = res_M;
    MaskC     sp_ok      = res_ok;

    for (int k = 0; k < m_n_neighbors; ++k) {
        FloatC u  = scene.m_samplers[0].next_1d<false>();
        FloatC v  = scene.m_samplers[0].next_1d<false>();
        IntC dx   = IntC((u * 2.f - 1.f) * static_cast<float>(m_spatial_radius));
        IntC dy   = IntC((v * 2.f - 1.f) * static_cast<float>(m_spatial_radius));
        IntC nx   = clamp(px + dx, 0, W - 1);
        IntC ny   = clamp(py + dy, 0, H - 1);
        IntC nbr  = ny * W + nx;

        Vector3fC nbr_p  = gather<Vector3fC>(res_p,  nbr, active_di);
        FloatC    nbr_J  = gather<FloatC>   (res_J,  nbr, active_di);
        FloatC    nbr_ws = gather<FloatC>   (res_ws, nbr, active_di);
        FloatC    nbr_f  = gather<FloatC>   (res_f,  nbr, active_di);
        FloatC    nbr_M  = gather<FloatC>   (res_M,  nbr, active_di);
        MaskC     nbr_ok = gather<MaskC>    (res_ok, nbr, active_di);

        MaskC ank = active_di && nbr_ok;

        Vector3fC wod_s  = nbr_p - its.p;
        FloatC dist_sq_s = squared_norm(wod_s);
        FloatC dist_s    = safe_sqrt(dist_sq_s);
        wod_s /= dist_s;

        RayC sray_s(its.p, wod_s);
        IntersectionC its_s = scene.ray_intersect<false>(sray_s, ank);
        ank &= its_s.is_valid() && its_s.is_emitter(ank)
            && its_s.t > dist_s - ShadowEpsilon;

        FloatC G_s       = abs(dot(its_s.n, -wod_s)) / dist_sq_s;
        SpectrumC Le_s   = its_s.Le(ank);
        SpectrumC bv_s   = bsdf_array->evalC(its, its.sh_frame.to_local(wod_s), ank);
        SpectrumC contrib_s = Le_s * bv_s * G_s * nbr_J;

        FloatC fq_s  = rgb2luminance<false>(contrib_s);
        FloatC w_nbr = select(ank && nbr_f > Epsilon,
                              nbr_ws * fq_s / nbr_f, FloatC(0.f));

        sp_ws += w_nbr;
        MaskC accept_k = (scene.m_samplers[0].next_1d<false>() * sp_ws) < w_nbr
                         && active_di;

        masked(sp_contrib, accept_k) = contrib_s;
        masked(sp_p,       accept_k) = nbr_p;
        masked(sp_J,       accept_k) = nbr_J;
        masked(sp_f,       accept_k) = fq_s;
        sp_ok  = sp_ok || accept_k;
        sp_M  += select(ank, nbr_M, FloatC(0.f));
    }

    // ---- Step 4: depth-0 shading ----
    FloatC W_final = select(sp_ok && sp_f > Epsilon,
                            sp_ws / (sp_M * sp_f), FloatC(0.f));
    result[active_di && sp_ok] += sp_contrib * W_final;

    // ---- Step 5: cache reservoir for render_restir_grad ----
    m_res_p  = sp_p;
    m_res_J  = sp_J;
    m_res_W  = W_final;
    m_res_ok = sp_ok;
    m_reservoir_valid = true;
    drjit::eval(m_res_p, m_res_J, m_res_W, m_res_ok);

    drjit::eval(result);

    // ---- Step 6: depth 1+ MIS path tracing ----
    if (m_max_depth >= 1) {
        SpectrumC throughput(1.f);
        IntersectionC curr_its = its;
        MaskC active_path = active;

        for (int depth = 0; depth < m_max_depth; ++depth) {
            BSDFArrayC bsdf_arr = curr_its.shape->bsdf();

            // NEE at current vertex
            {
                PositionSampleC ps = scene.sample_emitter_position<false>(
                    curr_its.p, scene.m_samplers[0].next_2d<false>(), active_path);
                MaskC ac_nee = active_path && ps.is_valid
                             && !curr_its.is_emitter(active_path);

                Vector3fC wod  = ps.p - curr_its.p;
                FloatC dist_sq = squared_norm(wod);
                FloatC dist    = safe_sqrt(dist_sq);
                wod /= dist;

                RayC sray(curr_its.p, wod);
                IntersectionC its_l = scene.ray_intersect<false, false>(sray, ac_nee);
                ac_nee &= its_l.is_valid() && its_l.is_emitter(ac_nee)
                       && its_l.t > dist - ShadowEpsilon;

                FloatC G     = abs(dot(its_l.n, -wod)) / dist_sq;
                Vector3fC wl = curr_its.sh_frame.to_local(wod);
                SpectrumC bv2 = bsdf_arr->evalC(curr_its, wl, ac_nee);
                bv2 *= G * ps.J / ps.pdf;
                FloatC pdf1 = bsdf_arr->pdfC(curr_its, wl, ac_nee) * G;
                FloatC w1   = mis_weight<false>(ps.pdf, pdf1);
                result[ac_nee] += throughput * its_l.Le(ac_nee) * bv2 * w1;
            }

            // BSDF sample (continuation)
            {
                BSDFSampleC bs = bsdf_arr->sampleC(curr_its,
                    scene.m_samplers[0].next_nd<3, false>(), active_path);
                active_path &= bs.is_valid;

                RayC cont_ray(curr_its.p, curr_its.sh_frame.to_world(bs.wo));
                IntersectionC its_next = scene.ray_intersect<false, false>(cont_ray,
                                                                            active_path);
                active_path &= its_next.is_valid();

                FloatC cos_v  = dot(its_next.n, -cont_ray.d);
                FloatC G_val  = abs(cos_v) / sqr(its_next.t);
                FloatC pdf0   = bs.pdf * G_val;
                SpectrumC bsdf_val = select(detach(its_next.t) < Epsilon, SpectrumC(0.f),
                                            bsdf_arr->evalC(curr_its, bs.wo, active_path)
                                            / bs.pdf);
                FloatC w2 = mis_weight<false>(pdf0,
                                scene.emitter_position_pdf<false>(curr_its.p, its_next,
                                                                   active_path));
                throughput *= bsdf_val;
                result[active_path] += its_next.Le(active_path) * throughput * w2;
                curr_its = its_next;
            }

            drjit::eval(result, throughput, active_path);
        }
    }

    drjit::eval(result);
    return result;
}

// ---------------------------------------------------------------------------
// render_restir_grad: depth-0 detached-adjoint gradient.
//
// Re-evaluates f_q(y_q) = Le * bsdf(θ) * G(θ) * J with full AD tracking,
// multiplied by the detached RIS weight W_q cached from the forward pass.
// The result is the gradient contribution of the depth-0 ReSTIR DI term.
// ---------------------------------------------------------------------------

void PathReSTIR::render_restir_grad(const Scene &scene, int sensor_id,
                                     SpectrumD &result) const {
    PSDR_ASSERT(m_reservoir_valid);

    const RenderOption &opts = scene.m_opts;
    const int W = opts.width, H = opts.height;

    // Re-shoot primary rays with AD (same pixel grid, jitter is re-sampled
    // but the *direction* only matters for the BSDF; the light position y_q
    // is taken from the cached reservoir, so the gradient is correct regardless
    // of the exact jitter used here).
    auto [fdx, fdy] = meshgrid(arange<FloatC>(W), arange<FloatC>(H));
    Vector2fC samples_base(fdx, fdy);
    Vector2fD samples = (Vector2fD(samples_base) + scene.m_samplers[0].next_2d<false>())
                        / ScalarVector2f(W, H);
    RayD cam_ray = scene.m_sensors[sensor_id]->sample_primary_ray(samples);

    MaskD active(true);
    IntersectionD its = scene.ray_intersect<true>(cam_ray, active);
    active &= its.is_valid();

    MaskD active_di = active && !its.is_emitter(active) && MaskD(m_res_ok);
    BSDFArrayD bsdf_array = its.shape->bsdf();

    // Direction toward cached reservoir sample (its.p is AD-tracked).
    // The light position m_res_p is treated as a constant (detached).
    Vector3fD wod   = Vector3fD(m_res_p) - its.p;
    FloatD dist_sq  = squared_norm(wod);
    FloatD dist     = safe_sqrt(dist_sq);
    wod /= dist;

    // Shadow test from the *primal* shading point (not AD-tracked) so
    // the binary visibility does not create a discontinuity in the AD graph.
    MaskC active_di_p = detach(active_di);
    RayC shadow_ray(detach(its.p), detach(wod));
    IntersectionC its_l = scene.ray_intersect<false>(shadow_ray, active_di_p);
    MaskD vis = active_di
              & MaskD(its_l.is_valid() && its_l.is_emitter(active_di_p)
                      && its_l.t > detach(dist) - ShadowEpsilon);

    active_di &= vis;
    MaskC active_di_p2 = detach(active_di);

    // Re-evaluate integrand with AD.
    // G uses AD-tracked its.p so geometry gradients flow through.
    FloatD cos_v  = FloatD(dot(its_l.n, detach(-wod)));
    FloatD G_ad   = abs(cos_v) / dist_sq;
    Vector3fD wo_local = its.sh_frame.to_local(wod);
    SpectrumD bsdf_ad  = bsdf_array->evalD(its, wo_local, active_di);
    SpectrumD Le_det   = SpectrumD(its_l.Le(active_di_p2));
    SpectrumD J_det    = SpectrumD(m_res_J);

    SpectrumD contrib_ad = Le_det * bsdf_ad * G_ad * J_det;

    // Multiply by detached RIS weight W_q (the sampling decision is fixed).
    SpectrumD grad_di = contrib_ad * SpectrumD(m_res_W);

    // Subtract primal to get the pure gradient (no double-counting the
    // primal that render_restir_path already emitted).
    grad_di -= detach(grad_di);

    masked(grad_di, ~active_di) = SpectrumD(0.f);
    masked(grad_di, ~drjit::isfinite<SpectrumD>(grad_di)) = SpectrumD(0.f);
    drjit::eval(grad_di);

    // scatter_reduce with idx=arange is identity indexing (no aliasing),
    // equivalent to direct +=.  Using += avoids a DrJIT crash that occurs
    // when scatter_reduce is applied to a lazy-zero SpectrumD target.
    result += grad_di;

    drjit::eval(result);
}

// ---------------------------------------------------------------------------
// renderD: orchestrates the three-pass differentiable render.
// ---------------------------------------------------------------------------

SpectrumD PathReSTIR::renderD(const Scene &scene, int sensor_id,
                               int seed, IntD /*batch_pix*/) const {
    PSDR_ASSERT_MSG(scene.is_ready(), "Scene must be configured!");
    PSDR_ASSERT_MSG(sensor_id >= 0 && sensor_id < scene.m_num_sensors,
                    "Invalid sensor id!");

    const RenderOption &opts = scene.m_opts;
    const int num_pixels = opts.width * opts.height;

    // Seed all three samplers.
    if (seed != -1) {
        scene.m_samplers[0].seed(arange<UInt64C>(num_pixels) +
                                  static_cast<uint64_t>(seed));
        if (opts.sppe > 0)
            scene.m_samplers[1].seed(arange<UInt64C>(num_pixels * opts.sppe) +
                                      static_cast<uint64_t>(seed));
        if (opts.sppse > 0)
            scene.m_samplers[2].seed(arange<UInt64C>(num_pixels * opts.sppse) +
                                      static_cast<uint64_t>(seed));
    }

    SpectrumD result = zeros<SpectrumD>(num_pixels);

    // Pass 1: primal forward + reservoir cache.
    render_restir_path(scene, sensor_id, -1 /* already seeded above */);

    // Pass 2: depth-0 ReSTIR DI gradient.
    render_restir_grad(scene, sensor_id, result);

    // Pass 3: depth 1+ interior gradient via Li(RayD) through __render<true>.
    // Re-seed sampler[0] with num_pixels*spp states — passes 1 and 2 consumed it
    // with only num_pixels states, but __render<true> needs num_pixels*spp.
    if (m_max_depth >= 1 && opts.spp > 0) {
        if (seed != -1)
            scene.m_samplers[0].seed(
                arange<UInt64C>(static_cast<int64_t>(num_pixels) * opts.spp) +
                static_cast<uint64_t>(seed));
        result += __render<true>(scene, sensor_id);
    }

    // Pass 4: primary-edge boundary integral (inherited, uses Li(RayC)).
    if (opts.sppe > 0)
        render_primary_edges(scene, sensor_id, result);

    // Secondary edges: deferred (base no-op).

    drjit::eval(result);
    return result;
}

NAMESPACE_END(psdr_jit)
