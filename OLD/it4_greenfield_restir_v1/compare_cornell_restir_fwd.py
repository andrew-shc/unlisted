"""
Cornell box shape optimization: PathTracer-fwd vs ReSTIR-fwd.

Both variants use PathTracer for the backward pass (renderD).
Difference is the forward renderer used to compute the loss image:
  standard    → PathTracer renderC  (integrator_id=0)
  restir_fwd  → PathReSTIR renderC  (integrator_id='integrator_restir')

50 iters each, WandB logging, saves side-by-side comparison PNG.
"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import sys
import time
import numpy as np
import torch
import psdr_jit
import drjit
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

NPBIR_PBR = '/home/ahc/Documents/metrology_ir/DigitalTwinCatalog/neural_pbir/pbir'
if NPBIR_PBR not in sys.path:
    sys.path.insert(0, NPBIR_PBR)

import gin
gin.enter_interactive_mode()

from irtk.scene import Scene, Mesh, DiffuseBRDF, HDRFilm, Integrator, PerspectiveCamera
from irtk.renderer import Renderer
from irtk.connector import get_connector
from irtk.io import write_image, to_torch_f
from irtk.loss import l1_loss
from irtk.config import configs
from irtk.connectors.psdr_jit_connector import PSDRJITConnector
from models.shape_ls import ShapeLS

CBOX = '/home/ahc/Documents/psdr-jit/tutorials/data/cbox'

# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

RESTIR_CONFIG = {
    'max_depth':      3,
    'n_candidates':   4,
    'n_neighbors':    4,
    'spatial_radius': 10,
    'hide_emitters':  False,
}

def load_obj_tri(path):
    verts, uvs, faces, faces_uv = [], [], [], []
    with open(path) as fh:
        for line in fh:
            t = line.split()
            if not t: continue
            if t[0] == 'v':    verts.append([float(x) for x in t[1:4]])
            elif t[0] == 'vt': uvs.append([float(x) for x in t[1:3]])
            elif t[0] == 'f':
                vi, ti = [], []
                for tok in t[1:]:
                    p = tok.split('/')
                    vi.append(int(p[0]) - 1)
                    ti.append(int(p[1]) - 1 if len(p) > 1 and p[1] else 0)
                for i in range(1, len(vi) - 1):
                    faces.append([vi[0], vi[i], vi[i + 1]])
                    faces_uv.append([ti[0], ti[i], ti[i + 1]])
    v   = np.array(verts,    dtype=np.float32)
    f   = np.array(faces,    dtype=np.int32)
    uv  = np.array(uvs,      dtype=np.float32) if uvs else np.zeros((len(verts), 2), np.float32)
    fuv = np.array(faces_uv, dtype=np.int32)   if uvs else f.copy()
    return v, f, uv, fuv

def mesh_from_obj(path, use_face_normal=False, **kwargs):
    v, f, uv, fuv = load_obj_tri(path)
    kwargs.setdefault('can_change_topology', True)
    return Mesh(v, f, uv, fuv, use_face_normal=use_face_normal, **kwargs)

def build_scene():
    """Builds a Cornell box with both PathTracer (idx 0) and PathReSTIR ('integrator_restir')."""
    scene = Scene()
    scene.set('film', HDRFilm(width=512, height=512))
    # idx 0 — PathTracer (used for loss forward in standard, and backward in both)
    scene.set('integrator',        Integrator('path',       {'max_depth': 3, 'hide_emitters': False}))
    # 'integrator_restir' — PathReSTIR (used for loss forward in restir_fwd)
    scene.set('integrator_restir', Integrator('path_restir', RESTIR_CONFIG))
    scene.set('sensor 0', PerspectiveCamera(
        fov=60,
        to_world=torch.tensor([[1., 0., 0., 208.],
                               [0., 1., 0., 273.],
                               [0., 0., 1., -800.],
                               [0., 0., 0., 1.]]),
        near=1e-6, far=1e7,
    ))
    scene.set('mat_cat',   DiffuseBRDF([0.5,  0.5,  0.5]))
    scene.set('mat_white', DiffuseBRDF([0.95, 0.95, 0.95]))
    scene.set('mat_green', DiffuseBRDF([0.20, 0.90, 0.20]))
    scene.set('mat_red',   DiffuseBRDF([0.90, 0.20, 0.20]))
    lum_xfm = torch.tensor([[1., 0., 0., 0.],
                             [0., 1., 0., -0.5],
                             [0., 0., 1., 0.],
                             [0., 0., 0., 1.]])
    scene.set('luminaire', mesh_from_obj(f'{CBOX}/cbox_luminaire.obj', radiance=[20., 20., 8.], to_world=lum_xfm))
    scene.set('smallbox',  mesh_from_obj(f'{CBOX}/cbox_smallbox.obj',  mat_id='mat_cat',   use_face_normal=True))
    scene.set('largebox',  mesh_from_obj(f'{CBOX}/cbox_largebox.obj',  mat_id='mat_cat',   use_face_normal=True))
    scene.set('floor',     mesh_from_obj(f'{CBOX}/cbox_floor.obj',     mat_id='mat_white', use_face_normal=False))
    scene.set('ceiling',   mesh_from_obj(f'{CBOX}/cbox_ceiling.obj',   mat_id='mat_white', use_face_normal=False))
    scene.set('back',      mesh_from_obj(f'{CBOX}/cbox_back.obj',      mat_id='mat_white', use_face_normal=False))
    scene.set('green',     mesh_from_obj(f'{CBOX}/cbox_greenwall.obj', mat_id='mat_green', use_face_normal=False))
    scene.set('red',       mesh_from_obj(f'{CBOX}/cbox_redwall.obj',   mat_id='mat_red',   use_face_normal=False))
    return scene

# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

class ReSTIRFwdFunction(torch.autograd.Function):
    """PathReSTIR renderC forward, PathTracer renderD backward."""
    @staticmethod
    def forward(ctx, connector, scene, fwd_options, bwd_options,
                sensor_ids, fwd_integrator_id, bwd_integrator_id, *params):
        images = connector.renderC(scene, fwd_options,
                                   sensor_ids=sensor_ids,
                                   integrator_id=fwd_integrator_id)
        images = torch.nan_to_num(torch.stack(images, dim=0))
        ctx.connector         = connector
        ctx.scene             = scene
        ctx.bwd_options       = bwd_options
        ctx.sensor_ids        = sensor_ids
        ctx.bwd_integrator_id = bwd_integrator_id
        ctx.num_no_grads      = 7
        return images.cpu()  # CPU roundtrip required — DrJIT CUDA refs must be released

    @staticmethod
    def backward(ctx, grad_out):
        image_grads = [g.to(configs['device']) for g in grad_out]
        param_grads = ctx.connector.renderD(
            image_grads, ctx.scene, ctx.bwd_options,
            ctx.sensor_ids, ctx.bwd_integrator_id)
        return tuple([None] * ctx.num_no_grads + param_grads)


class ReSTIRFwdRenderer(torch.nn.Module):
    def __init__(self, fwd_options, bwd_options, fwd_integrator_id, bwd_integrator_id):
        super().__init__()
        self.connector         = get_connector('psdr_jit')
        self.fwd_options       = fwd_options
        self.bwd_options       = bwd_options
        self.fwd_integrator_id = fwd_integrator_id
        self.bwd_integrator_id = bwd_integrator_id

    def forward(self, scene, sensor_ids=[0], integrator_id=0):
        if torch.is_tensor(sensor_ids):
            sensor_ids = sensor_ids.flatten().tolist()
        params = [scene[n] for n in scene.requiring_grad]
        return ReSTIRFwdFunction.apply(
            self.connector, scene, self.fwd_options, self.bwd_options,
            sensor_ids, self.fwd_integrator_id, self.bwd_integrator_id,
            *params
        ).to(configs['device'])

# ---------------------------------------------------------------------------
# SPP config
# ---------------------------------------------------------------------------

# Both variants use the same spp budget so the comparison is fair
# sppe/sppse > 0 required for shape optimization (silhouette/boundary integrals)
STD_OPT = {'spp': 16, 'sppe': 16, 'sppse': 8, 'npass': 1, 'log_level': 0}
# restir_fwd: PathReSTIR renderC (no boundary needed in primal), PathTracer renderD (boundary needed)
RFWD_FWD_OPT = {'spp':  4, 'sppe':  0, 'sppse': 0, 'npass': 1, 'log_level': 0}
RFWD_BWD_OPT = {'spp': 16, 'sppe': 16, 'sppse': 8, 'npass': 1, 'log_level': 0}
VIS_OPT      = {'spp': 64, 'sppe':  0, 'sppse': 0, 'npass': 1, 'log_level': 0}

NUM_ITERS  = 50
SAVE_EVERY = 10
PERTURB_Y  = 70.0
LR         = 5.0
CROP       = (160, 460, 90, 390)

def crop(img):
    r0, r1, c0, c1 = CROP
    return img[r0:r1, c0:c1] if img.ndim == 2 else img[r0:r1, c0:c1, :]

def tonemap(img_np):
    return np.clip(img_np ** (1 / 2.2), 0, 1)

# ---------------------------------------------------------------------------
# Optimization runner
# ---------------------------------------------------------------------------

def run_opt(label, scene, render_opt, target, V_gt, wandb_run=None):
    render_vis = Renderer('psdr_jit', render_options={**VIS_OPT, 'npass': 1, 'log_level': 0})

    scene['smallbox']['v'] = V_gt + to_torch_f([[0., PERTURB_Y, 0.]])
    model = ShapeLS(scene, mesh_id='smallbox', optimizer_kwargs={'lr': LR, 'lmbda': 1})

    gt_mean_y = V_gt[:, 1].mean().item()
    losses, vis_frames = [], []

    pbar = tqdm(total=NUM_ITERS, desc=label, leave=True)
    for it in range(NUM_ITERS):
        t0 = time.perf_counter()
        model.zero_grad()
        model.set_data()
        scene.configure()

        opt_image  = render_opt(scene, sensor_ids=[0])[0]
        image_loss = l1_loss(target, opt_image)
        reg_loss   = model.get_regularization()
        (image_loss + reg_loss).backward()
        model.step()
        torch.cuda.synchronize()
        drjit.flush_malloc_cache()

        dt = time.perf_counter() - t0
        losses.append(image_loss.item())
        cur_y = scene['smallbox.v'].detach()[:, 1].mean().item()

        pbar.update(1)
        pbar.set_postfix({'L1': f'{image_loss.item():.5f}',
                          'Y':  f'{cur_y:.1f}/{gt_mean_y:.1f}',
                          'ms': f'{dt*1000:.0f}'})

        if wandb_run is not None:
            wandb_run.log({f'{label}/loss': image_loss.item(),
                           f'{label}/Y': cur_y,
                           f'{label}/ms': dt * 1000}, step=it)

        if it % SAVE_EVERY == 0 or it == NUM_ITERS - 1:
            with torch.no_grad():
                frame = render_vis(scene, sensor_ids=[0])[0]
            vis_frames.append(crop(tonemap(frame.cpu().numpy())))

    pbar.close()
    print(f"[{label}] done — final L1={losses[-1]:.5f}  Y={cur_y:.1f}/{gt_mean_y:.1f}", flush=True)
    return losses, vis_frames

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv(Path(__file__).parent.parent / '.env')

    try:
        import wandb
        # WandB writes its local run logs under `dir` — default into ASSETS/
        # (per AGENTS.md's data-separation rule) but let WANDB_DIR override it.
        run = wandb.init(
            project=os.environ.get('WANDB_PROJECT', 'ReSTIR PBIR'),
            entity=os.environ.get('WANDB_ENTITY'),
            dir=os.environ.get('WANDB_DIR', 'ASSETS/wandb'),
            name='cornell_restir_fwd_vs_standard',
            config={
                'scene': 'cornell_box',
                'num_iters': NUM_ITERS,
                'std_spp': STD_OPT['spp'],
                'rfwd_fwd_spp': RFWD_FWD_OPT['spp'],
                'rfwd_bwd_spp': RFWD_BWD_OPT['spp'],
                'restir_n_candidates': RESTIR_CONFIG['n_candidates'],
                'restir_n_neighbors':  RESTIR_CONFIG['n_neighbors'],
                'lr': LR,
                'perturb_y': PERTURB_Y,
            },
        )
    except Exception as e:
        print(f'WandB init skipped: {e}')
        run = None

    out_dir = Path('cornell_restir_fwd_comparison')
    out_dir.mkdir(exist_ok=True)

    # Ground-truth target
    print('Rendering GT target…', flush=True)
    scene_gt  = build_scene()
    render_ref = Renderer('psdr_jit', render_options={**VIS_OPT, 'npass': 1, 'log_level': 0})
    scene_gt.configure()
    target = render_ref(scene_gt, sensor_ids=[0])[0]
    write_image(str(out_dir / 'target.exr'), target)
    V_gt = scene_gt['smallbox.v'].detach().clone()
    del scene_gt

    # Run 1 — standard (PathTracer fwd + PathTracer bwd) via irtk Renderer
    print('\n=== standard (PathTracer fwd + PathTracer bwd) ===', flush=True)
    scene_std  = build_scene()
    render_std = Renderer('psdr_jit', render_options=STD_OPT)
    losses_std, frames_std = run_opt('standard', scene_std, render_std, target, V_gt.clone(), run)
    del scene_std

    # Run 2 — restir_fwd (PathReSTIR fwd + PathTracer bwd)
    print('\n=== restir_fwd (PathReSTIR fwd + PathTracer bwd) ===', flush=True)
    scene_rfwd  = build_scene()
    render_rfwd = ReSTIRFwdRenderer(
        fwd_options=RFWD_FWD_OPT,             # PathReSTIR renderC (sppe=0, primal only)
        bwd_options=RFWD_BWD_OPT,             # PathTracer renderD (sppe=16, boundary integrals)
        fwd_integrator_id='integrator_restir', # PathReSTIR
        bwd_integrator_id=0,                   # PathTracer
    )
    losses_rfwd, frames_rfwd = run_opt('restir_fwd', scene_rfwd, render_rfwd, target, V_gt.clone(), run)
    del scene_rfwd

    # Save comparison grid
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_frames = min(len(frames_std), len(frames_rfwd))
    fig, axes = plt.subplots(2, n_frames, figsize=(4 * n_frames, 6))
    for i in range(n_frames):
        axes[0, i].imshow(frames_std[i]);  axes[0, i].axis('off')
        axes[1, i].imshow(frames_rfwd[i]); axes[1, i].axis('off')
        it_label = f'iter {i * SAVE_EVERY}' if i < n_frames - 1 else f'iter {NUM_ITERS - 1}'
        axes[0, i].set_title(it_label, fontsize=8)
    axes[0, 0].set_ylabel('standard', fontsize=9)
    axes[1, 0].set_ylabel('restir_fwd', fontsize=9)
    plt.tight_layout()
    grid_path = str(out_dir / 'comparison_grid.png')
    plt.savefig(grid_path, dpi=150)
    plt.close()
    print(f'Saved {grid_path}', flush=True)

    # Loss curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses_std,  label='standard (PT fwd)')
    ax.plot(losses_rfwd, label='restir_fwd (ReSTIR fwd)')
    ax.set_xlabel('iteration'); ax.set_ylabel('L1 loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title('Cornell box: PathTracer vs ReSTIR forward')
    curve_path = str(out_dir / 'loss_curves.png')
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f'Saved {curve_path}', flush=True)

    if run is not None:
        import wandb as _wandb
        run.log({'comparison_grid': _wandb.Image(grid_path),
                 'loss_curves':     _wandb.Image(curve_path)})
        run.finish()

    print(f'\nFinal L1  standard={losses_std[-1]:.5f}  restir_fwd={losses_rfwd[-1]:.5f}')


if __name__ == '__main__':
    main()
