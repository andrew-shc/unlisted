"""
Minimal stage-2-only test: does PathReSTIRRenderer renderC work on a fresh process
with teapot_scene001, without running stage 1 first?

If this works → the segfault in run_restir.py is a cross-stage state issue.
If this crashes the same way → intrinsic PathReSTIR + Stanford-ORB scene bug.

Run:
  cd DigitalTwinCatalog/neural_pbir/pbir
  python test_restir_stage2.py <configroot> <ckptroot>
  # e.g.:
  # python test_restir_stage2.py ./configs/restir_template \
  #   ../../../restir_NPBIR_v1/stanford_orb/teapot_scene001
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import gin
import psdr_jit
import torch
from mmcv import Config
from irtk.scene import Mesh, Integrator
from irtk.renderer import Renderer
from irtk.connector import get_connector
from irtk.connectors.psdr_jit_connector import PSDRJITConnector
from datasets import NeuralPBIRDataset  # noqa: F401 — gin needs it

# ── Patches (identical to run_restir.py) ────────────────────────────────────

_orig_mesh_handler = PSDRJITConnector.extensions[Mesh]

def _mesh_handler_reapply_face_normal(name, scene):
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

# ── PathReSTIRRenderer (identical to run_restir.py) ─────────────────────────

class PathReSTIRFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, connector, scene, fwd_options, bwd_options,
                sensor_ids, fwd_integrator_id, bwd_integrator_id, *params):
        print(f"  [DBG] renderC called: sensor_ids={sensor_ids} integrator_id={fwd_integrator_id}", flush=True)
        images = connector.renderC(scene, fwd_options,
                                   sensor_ids=sensor_ids,
                                   integrator_id=fwd_integrator_id)
        print(f"  [DBG] renderC returned {len(images)} images", flush=True)
        images = torch.nan_to_num(torch.stack(images, dim=0))
        ctx.connector = connector
        ctx.scene = scene
        ctx.bwd_options = bwd_options
        ctx.sensor_ids = sensor_ids
        ctx.bwd_integrator_id = bwd_integrator_id
        ctx.num_no_grads = 7
        return images.cpu()

    @staticmethod
    def backward(ctx, grad_out):
        image_grads = [g.to('cuda') for g in grad_out]
        param_grads = ctx.connector.renderD(
            image_grads, ctx.scene, ctx.bwd_options,
            ctx.sensor_ids, ctx.bwd_integrator_id)
        return tuple([None] * ctx.num_no_grads + param_grads)


class PathReSTIRRenderer(torch.nn.Module):
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
        sensor_ids = [int(s) for s in sensor_ids]  # defensive: ensure plain ints
        print(f"  [DBG] PathReSTIRRenderer.forward sensor_ids={sensor_ids}", flush=True)
        params = [scene[n] for n in scene.requiring_grad]
        print(f"  [DBG] requiring_grad: {list(scene.requiring_grad)[:4]}...", flush=True)
        return PathReSTIRFunction.apply(
            self.connector, scene, self.fwd_options, self.bwd_options,
            sensor_ids, self.fwd_integrator_id, self.bwd_integrator_id,
            *params
        ).to('cuda')


RESTIR_CONFIG = {
    'max_depth':      3,
    'n_candidates':   4,
    'n_neighbors':    5,
    'spatial_radius': 10,
    'hide_emitters':  False,
}
STAGE2_FWD = {'spp': 4, 'sppe': 0, 'sppse': 0, 'npass': 1, 'log_level': 0}
STAGE2_BWD = {'spp': 2, 'sppe': 0, 'sppse': 0, 'npass': 1, 'log_level': 0}


def main():
    parser = ArgumentParser()
    parser.add_argument("configroot", type=str)
    parser.add_argument("ckptroot",   type=str)
    args = parser.parse_args()

    configroot = Path(args.configroot)
    ckptroot   = Path(args.ckptroot)

    # Skip all gin configs — we're not calling optimize(), just doing one renderC.
    # NeuralPBIRDataset args are passed explicitly.
    print("=== Creating dataset ===", flush=True)
    cfg = Config.fromfile(ckptroot / "neural_surface_recon" / "config.py")
    dataset = NeuralPBIRDataset(
        dataroot=cfg.data.datadir,
        ckptroot=ckptroot,
        savemem=True,
        F0=0.04,
        integrator_type='path',
        integrator_config={'max_depth': 3, 'hide_emitters': False},
    )

    print("=== get_scene() ===", flush=True)
    scene = dataset.get_scene()

    print("=== Injecting path_restir integrator ===", flush=True)
    scene.set('integrator_restir', Integrator('path_restir', RESTIR_CONFIG))
    scene.set('integrator_vis',    Integrator('path', {'max_depth': 3, 'hide_emitters': False}))

    print("=== scene.configure() ===", flush=True)
    scene.configure()
    print(f"  requiring_grad: {scene.requiring_grad}", flush=True)

    print("=== Creating PathReSTIRRenderer ===", flush=True)
    render_opt = PathReSTIRRenderer(fwd_options=STAGE2_FWD, bwd_options=STAGE2_BWD)

    print("=== ONE forward pass (sensor_id=0) ===", flush=True)
    opt_image = render_opt(scene, sensor_ids=[0])
    print(f"  opt_image shape={opt_image.shape} mean={opt_image.mean():.4f}", flush=True)

    print("=== PASS — PathReSTIRRenderer renderC works in fresh process ===", flush=True)


if __name__ == "__main__":
    main()
