# ReSTIR PBIR Run Summary

## Run 1 — teapot_scene001 PathReSTIR PBIR (2026-07-01)

**Scene**: Stanford ORB `teapot_scene001`  
**Pipeline**: `run_restir.py` — single-stage MicrofacetBasis + EnvmapLS  
**Integrators**: PathTracer (renderC, fwd) + PathReSTIR (renderD, bwd)  
**Iterations**: 200  
**Checkpoint interval**: every 25 iters  
**WandB run**: https://wandb.ai/andrew-shc/ReSTIR%20PBIR/runs/3mskxvt3

### Results

| Iter | Loss |
|------|------|
| 1    | 0.084211 |
| 25   | 0.047232 |
| 50   | 0.059924 |
| 75   | 0.091704 |
| 100  | 0.062900 |
| 125  | 0.053245 |
| 150  | 0.061062 |
| 175  | 0.046914 |
| 200  | 0.044621 |

**Final image_loss**: 0.04462  
**Total wall time**: ~2m 53s (GPU-accelerated)  
**Speed**: ~1.15 it/s at convergence

### Outputs

```
restir_NPBIR_v1/stanford_orb/teapot_scene001/pbir/
  diffuse.exr / diffuse.png
  roughness.exr / roughness.png
  specular.exr / specular.png
  envmap.exr
  mesh.obj
  microfacet_basis-envmap_ls/
    vis_animation.gif
    loss.pt
    {25,50,...,200}/   (checkpoint renders)
    final/             (final material maps)
```

WandB logged: per-iter loss, render_ms, opt_ms, checkpoint vis images, final material maps, animation GIF.

### Fix Applied (PathReSTIR segfault)

The run required a C++ fix to `psdr-jit` before it could complete:

- **File**: `/home/ahc/Documents/psdr-jit/src/integrator/path_restir.cpp`
- **Root cause**: `render_restir_grad` called `scatter_reduce(ReduceOp::Add, result[j], grad_di[j], idx, active_di)` where `result = zeros<SpectrumD>(num_pixels)` is a DrJIT lazy-zero constant. `drjit::eval(result)` crashed materializing scatter_reduce into a lazy target.
- **Fix**: Replaced scatter_reduce loop with `result += grad_di` (valid since `idx = arange(num_pixels)` is identity indexing with no aliasing).
- **Secondary fix**: Guarded `scene.m_samplers[1/2].seed(...)` in `renderD` with `if (opts.sppe > 0)` / `if (opts.sppse > 0)` to avoid seeding samplers for zero-spp passes.
- Build: `ninja -j4` → `cmake --install . --prefix site-packages` (editable install in `psdr_jit/psdr_jit.so`).
