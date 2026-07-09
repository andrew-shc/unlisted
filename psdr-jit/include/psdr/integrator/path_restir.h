#pragma once
#include "integrator.h"

NAMESPACE_BEGIN(psdr_jit)

PSDR_CLASS_DECL_BEGIN(PathReSTIR, final, Integrator)
public:
    PathReSTIR(int max_depth = 3, int n_candidates = 4,
               int n_neighbors = 5, int spatial_radius = 30);
    virtual ~PathReSTIR() {}

    // Primal: ReSTIR DI at depth 0 + MIS path tracing for depth 1+.
    // Caches per-pixel reservoir state (y_q, W_q) for use by renderD.
    SpectrumC render_restir_path(const Scene &scene, int sensor_id = 0,
                                  int seed = -1) const;

    // Full differentiable render. Orchestrates:
    //   1. render_restir_path   (primal + reservoir cache)
    //   2. render_restir_grad   (depth-0 detached-adjoint gradient)
    //   3. __render<true>       (depth 1+ interior gradient via Li(RayD))
    //   4. render_primary_edges (boundary integral, inherited)
    // Shadows (does not override) Integrator::renderD — base renderD is not
    // virtual. Pybind11 binds to this concrete method directly.
    SpectrumD renderD(const Scene &scene, int sensor_id = 0,
                      int seed = -1, IntD batch_pix = -1) const;

    int  m_max_depth;
    int  m_n_candidates;
    int  m_n_neighbors;
    int  m_spatial_radius;
    bool m_hide_emitters = false;

protected:
    // Li(RayC): single-sample NEE for primary-edge boundary integrals.
    SpectrumC Li(const Scene &scene, Sampler &sampler, const RayC &ray,
                 MaskC active = true) const override;

    // Li(RayD): depth-1+ differentiable path (skips depth-0 NEE, which is
    // handled separately in render_restir_grad).
    SpectrumD Li(const Scene &scene, Sampler &sampler, const RayD &ray,
                 MaskD active = true) const override;

    // Gradient for the depth-0 ReSTIR DI term: re-evaluates f_q(y_q) with
    // AD using cached reservoir, multiplied by the detached RIS weight W_q.
    void render_restir_grad(const Scene &scene, int sensor_id,
                             SpectrumD &result) const;

    // Cached reservoir state (primal). Populated by render_restir_path,
    // consumed by render_restir_grad. mutable to allow writes in const methods.
    mutable Vector3fC m_res_p;       // selected emitter sample world position
    mutable FloatC    m_res_J;       // area-form Jacobian at selected sample
    mutable FloatC    m_res_W;       // unbiased RIS weight W = ws/(M*f_hat)
    mutable MaskC     m_res_ok;      // valid reservoir per pixel
    mutable bool      m_reservoir_valid = false;

PSDR_CLASS_DECL_END(PathReSTIR)

NAMESPACE_END(psdr_jit)
