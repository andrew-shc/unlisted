#pragma once
#include "integrator.h"

NAMESPACE_BEGIN(psdr_jit)

PSDR_CLASS_DECL_BEGIN(DirectReSTIR,, Integrator)
public:
    DirectReSTIR(int n_candidates = 4, int n_neighbors = 5, int spatial_radius = 30);
    virtual ~DirectReSTIR() {}

    // ReSTIR DI: candidate generation + spatial reuse, no temporal reuse.
    // Returns a primal (non-differentiable) image. Call multiple times and
    // average for higher spp (each call uses a different seed offset).
    SpectrumC render_restir(const Scene &scene, int sensor_id = 0, int seed = -1) const;

    int  m_n_candidates;   // M: light candidates generated per pixel
    int  m_n_neighbors;    // K: spatial neighbors to reuse from
    int  m_spatial_radius; // pixel radius for neighbor search
    bool m_hide_emitters = false;

protected:
    // Minimal single-sample direct Li for primary-edge compatibility.
    SpectrumC Li(const Scene &scene, Sampler &sampler, const RayC &ray,
                 MaskC active = true) const override;
    SpectrumD Li(const Scene &scene, Sampler &sampler, const RayD &ray,
                 MaskD active = true) const override;

PSDR_CLASS_DECL_END(DirectReSTIR)

NAMESPACE_END(psdr_jit)
