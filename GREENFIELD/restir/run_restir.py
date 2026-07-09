#!/usr/bin/env python3
"""
Single-stage PBIR with PathReSTIR backward gradient — Stanford-ORB drop-in.

Uses MicrofacetBasis + EnvmapLS initialized from neural-distillation outputs, with:
  forward  → PathTracer renderC (full-path images for L1 loss)
  backward → PathReSTIR renderD (spatial-resampling gradient)

Usage:
  python GREENFIELD/restir/run_restir.py <configroot> <ckptroot> [--gt_envmap_path PATH]
  # conda activate metrology_ir
  # PYTHONUNBUFFERED=1 python GREENFIELD/restir/run_restir.py ...

Notes:
  - PathReSTIR::renderC() is a direct-illumination-only stub (needed for
    silhouette/boundary edges). The full multi-bounce forward path comes from
    renderD's primal+cache pass. Using PathTracer for renderC gives correct
    full-path images for the loss; PathReSTIR renderD gives the low-variance
    adjoint gradient.
  - The `use_face_normal` patch is required for PathReSTIR to work correctly.
    It is applied at module import, before any scene is configured.
"""

import os
import sys
import shutil
from argparse import ArgumentParser
from pathlib import Path

# --- Add DigitalTwinCatalog to sys.path ---
REPO_ROOT = Path(__file__).resolve().parents[2]
NPBIR_PBIR = str(REPO_ROOT / 'DigitalTwinCatalog' / 'neural_pbir' / 'pbir')
if NPBIR_PBIR not in sys.path:
    sys.path.insert(0, NPBIR_PBIR)

import gin
import psdr_jit
import torch
from mmcv import Config
from opt import optimize
from irtk.scene import Mesh, Integrator
from irtk.renderer import Renderer
from irtk.connector import get_connector
from irtk.config import configs
from irtk.connectors.psdr_jit_connector import PSDRJITConnector
from datasets import NeuralPBIRDataset  # noqa: F401 — gin needs this imported

# ---------------------------------------------------------------------------
# Patches — applied at import, before any scene is configured
# ---------------------------------------------------------------------------

_orig_mesh_handler = PSDRJITConnector.extensions[Mesh]

def _mesh_handler_reapply_face_normal(name, scene):
    """Re-apply use_face_normal after load_raw() clears it."""
    result = _orig_mesh_handler(name, scene)
    cache = scene.cached.get('psdr_jit')
    if cache and name in cache.get('name_map', {}):
        if scene[name]['use_face_normal']:
            psdr_mesh = cache['scene'].param_map[cache['name_map'][name]]
            psdr_mesh.use_face_normal = True
    return result

PSDRJITConnector.extensions[Mesh] = _mesh_handler_reapply_face_normal

_orig_integrator_handler = PSDRJITConnector.extensions[Integrator]

def _integrator_handler_with_restir(name, scene):
    """Register 'path_restir' integrator type for the connector."""
    integrator = scene[name]
    if integrator['type'] == 'path_restir':
        cache = scene.cached['psdr_jit']
        cfg = integrator['config']
        psdr_integrator = psdr_jit.PathReSTIR()
        for attr in ('max_depth', 'n_candidates', 'n_neighbors',
                     'spatial_radius', 'hide_emitters'):
            if attr in cfg:
                setattr(psdr_integrator, attr, cfg[attr])
        cache['integrators'][name] = psdr_integrator
        return []
    return _orig_integrator_handler(name, scene)

PSDRJITConnector.extensions[Integrator] = _integrator_handler_with_restir

# ---------------------------------------------------------------------------
# PathReSTIR autograd function
# forward  → PathTracer renderC (index 0) — full-path, unbiased loss
# backward → PathReSTIR renderD          — spatial resampling gradient
# ---------------------------------------------------------------------------

class PathReSTIRFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, connector, scene, fwd_options, bwd_options,
                sensor_ids, fwd_integrator_id, bwd_integrator_id, *params):
        images = connector.renderC(scene, fwd_options,
                                   sensor_ids=sensor_ids,
                                   integrator_id=fwd_integrator_id)
        images = torch.nan_to_num(torch.stack(images, dim=0))
        ctx.connector = connector
        ctx.scene = scene
        ctx.bwd_options = bwd_options
        ctx.sensor_ids = sensor_ids
        ctx.bwd_integrator_id = bwd_integrator_id
        ctx.num_no_grads = 7
        return images  # already on CUDA from DrJIT

    @staticmethod
    def backward(ctx, grad_out):
        image_grads = [g.to(configs['device']) for g in grad_out]
        param_grads = ctx.connector.renderD(
            image_grads, ctx.scene, ctx.bwd_options,
            ctx.sensor_ids, ctx.bwd_integrator_id)
        return tuple([None] * ctx.num_no_grads + param_grads)


class PathReSTIRRenderer(torch.nn.Module):
    """PathTracer renderC + PathReSTIR renderD."""
    def __init__(self, fwd_options, bwd_options,
                 fwd_integrator_id=0, bwd_integrator_id='integrator_restir'):
        super().__init__()
        self.connector = get_connector('psdr_jit')
        self.fwd_options = fwd_options
        self.bwd_options = bwd_options
        self.fwd_integrator_id = fwd_integrator_id
        self.bwd_integrator_id = bwd_integrator_id

    def forward(self, scene, sensor_ids=None, integrator_id=0):
        if sensor_ids is None:
            sensor_ids = [0]
        if torch.is_tensor(sensor_ids):
            sensor_ids = sensor_ids.flatten().tolist()
        sensor_ids = [int(s) for s in sensor_ids]
        params = [scene[n] for n in scene.requiring_grad]
        return PathReSTIRFunction.apply(
            self.connector, scene, self.fwd_options, self.bwd_options,
            sensor_ids, self.fwd_integrator_id, self.bwd_integrator_id,
            *params
        )

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESTIR_CONFIG = {
    'max_depth':      3,
    'n_candidates':   2,   # reduced for 2048×2048 reservoir memory
    'n_neighbors':    2,
    'spatial_radius': 5,
    'hide_emitters':  False,
}

FWD_OPT = {'spp': 4, 'sppe': 0, 'sppse': 0, 'npass': 1, 'log_level': 0}
BWD_OPT = {'spp': 1, 'sppe': 0, 'sppse': 0, 'npass': 1, 'log_level': 0}
VIS_OPT = {'spp': 8, 'sppe': 0, 'sppse': 0, 'npass': 1, 'log_level': 0}

MAX_ITER        = 200
CHECKPOINT_ITER = 25

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@gin.configurable
def pipeline_restir(configroot, ckptroot, dataset_class, gt_envmap_path=None):
    """Run the full PBIR optimization with PathReSTIR backward gradient."""
    cfg = Config.fromfile(ckptroot / "neural_surface_recon" / "config.py")
    dataset = dataset_class(dataroot=cfg.data.datadir, ckptroot=ckptroot)
    scene = dataset.get_scene()

    # PathReSTIR and vis integrators alongside the default PathTracer (id=0)
    scene.set('integrator_restir', Integrator('path_restir', RESTIR_CONFIG))
    scene.set('integrator_vis',    Integrator('path', {'max_depth': 3, 'hide_emitters': False}))

    gin.parse_config_file(configroot / "microfacet_basis-envmap_ls.gin")
    if gt_envmap_path is not None:
        gin.bind_parameter("EnvmapLS.gt_envmap_path", gt_envmap_path)

    render_opt = PathReSTIRRenderer(fwd_options=FWD_OPT, bwd_options=BWD_OPT)
    render_vis = Renderer('psdr_jit', render_options=VIS_OPT)

    result_root = Path(dataset.result_root)
    result_path = result_root / "microfacet_basis-envmap_ls"

    scene = optimize(
        scene=scene, dataset=dataset, result_path=result_path,
        render_opt=render_opt, render_vis=render_vis,
        max_iter=MAX_ITER, checkpoint_iter=CHECKPOINT_ITER,
    )

    shutil.copytree(result_path / "final", result_root, dirs_exist_ok=True)
    print(f"Final results → {result_root}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(description="PBIR with PathReSTIR backward gradient")
    parser.add_argument("configroot", type=str, help="Path to gin config directory")
    parser.add_argument("ckptroot",   type=str, help="Path to checkpoint root (scene folder)")
    parser.add_argument("--gt_envmap_path", type=str, default=None,
                        help="Optional ground-truth envmap for supervision")
    args = parser.parse_args()

    configroot = Path(args.configroot)
    ckptroot   = Path(args.ckptroot)

    # WandB — init here so opt.py picks up the active run for per-iter logging
    try:
        from dotenv import load_dotenv
        import wandb
        load_dotenv(REPO_ROOT / ".env")
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "ReSTIR PBIR"),
            entity=os.environ.get("WANDB_ENTITY"),
            dir=os.environ.get("WANDB_DIR", str(REPO_ROOT / "ASSETS" / "wandb")),
            name=f"restir_NPBIR_v1/{ckptroot.name}",
            config={
                "method":                 "restir_NPBIR_v1",
                "scene":                  ckptroot.name,
                "restir_n_candidates":    RESTIR_CONFIG['n_candidates'],
                "restir_n_neighbors":     RESTIR_CONFIG['n_neighbors'],
                "restir_spatial_radius":  RESTIR_CONFIG['spatial_radius'],
                "fwd_spp":                FWD_OPT['spp'],
                "bwd_spp":                BWD_OPT['spp'],
                "max_iter":               MAX_ITER,
                "checkpoint_iter":        CHECKPOINT_ITER,
            },
        )
    except Exception as e:
        print(f"WandB init skipped: {e}")

    gin.parse_config_file(configroot / "pipeline_restir.gin")
    pipeline_restir(configroot=configroot, ckptroot=ckptroot,
                    gt_envmap_path=args.gt_envmap_path)

    try:
        import wandb
        if wandb.run:
            wandb.finish()
    except Exception:
        pass
