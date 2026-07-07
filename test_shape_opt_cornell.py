"""
Shape optimization on the Cornell box:
  Standard PathTracer (spp=16)  vs  PathReSTIR (psdr-jit unbiased spatial resampling).

PathReSTIR spatial resampling:
  - Gathers candidate path samples from spatial neighbours
  - Re-evaluates each candidate at the CURRENT pixel's shading point (shift mapping)
  - Unbiased: no surface-boundary bias from radiance-value averaging
  - Differentiable adjoint via psdr-jit's 3-pass algorithm
  - Integrator lives in psdr-jit C++ — no custom Python kernels

GIF shows:
  Row 0 — CROPPED VIS renders (64 spp PathTracer, zoomed to small-box region)
  Row 1 — ∂I/∂Y gradient image at current box position (animated per-iteration)
           Standard : spp=2 FD estimate (noisy)
           PathReSTIR: spp=2 FD estimate (same visualisation renderer)
  Row 2 — L1 loss curve | speed bar
"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import psdr_jit
import drjit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from tqdm import tqdm
import json
import datetime

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
# Patch: re-apply use_face_normal after every load_raw().
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

# ---------------------------------------------------------------------------
# Patch: add 'path_restir' integrator type to the irtk connector.
# The stock connector only knows 'path', 'direct', 'field', 'collocated'.
# ---------------------------------------------------------------------------
_orig_integrator_handler = PSDRJITConnector.extensions[Integrator]

def _integrator_handler_with_restir(name, scene):
    integrator = scene[name]
    if integrator['type'] == 'path_restir':
        cache = scene.cached['psdr_jit']
        cfg   = integrator['config']
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
# Mesh loading
# ---------------------------------------------------------------------------

def load_obj_tri(path):
    verts, uvs, faces, faces_uv = [], [], [], []
    with open(path) as fh:
        for line in fh:
            t = line.split()
            if not t: continue
            if t[0] == 'v':  verts.append([float(x) for x in t[1:4]])
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

# ---------------------------------------------------------------------------
# Scene factory
#
# use_restir=True: sets PathReSTIR as integrator_id=0, adds a standard
#   PathTracer as 'integrator_vis' for clean VIS renders.
#   PathReSTIR::Li(RayC) is a direct-only stub — VIS renders through it
#   would appear near-black without indirect illumination.
# ---------------------------------------------------------------------------

_RESTIR_BASE = {
    'max_depth':      3,
    'n_neighbors':    5,
    'spatial_radius': 10,
    'hide_emitters':  False,
}
RESTIR_PARAMS_4 = {**_RESTIR_BASE, 'n_candidates': 4}
RESTIR_PARAMS_8 = {**_RESTIR_BASE, 'n_candidates': 8}

def build_scene(use_restir=False, restir_params=None):
    scene = Scene()
    scene.set('film', HDRFilm(width=512, height=512))
    if use_restir:
        # 'integrator'        → PathTracer for forward renderC (clean full-path images)
        # 'integrator_restir' → PathReSTIR for backward renderD (low-variance gradient)
        # 'integrator_vis'    → PathTracer for VIS renders (PathReSTIR::Li(RayC) is direct-only)
        scene.set('integrator',        Integrator('path',       {'max_depth': 3, 'hide_emitters': False}))
        scene.set('integrator_restir', Integrator('path_restir', restir_params or RESTIR_PARAMS_4))
        scene.set('integrator_vis',    Integrator('path',       {'max_depth': 3, 'hide_emitters': False}))
    else:
        scene.set('integrator', Integrator('path', {'max_depth': 3, 'hide_emitters': False}))
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
    scene.set('luminaire', mesh_from_obj(f'{CBOX}/cbox_luminaire.obj',
        radiance=[20., 20., 8.], to_world=lum_xfm))
    scene.set('smallbox', mesh_from_obj(f'{CBOX}/cbox_smallbox.obj',  mat_id='mat_cat',   use_face_normal=True))
    scene.set('largebox', mesh_from_obj(f'{CBOX}/cbox_largebox.obj',  mat_id='mat_cat',   use_face_normal=True))
    scene.set('floor',    mesh_from_obj(f'{CBOX}/cbox_floor.obj',     mat_id='mat_white', use_face_normal=False))
    scene.set('ceiling',  mesh_from_obj(f'{CBOX}/cbox_ceiling.obj',   mat_id='mat_white', use_face_normal=False))
    scene.set('back',     mesh_from_obj(f'{CBOX}/cbox_back.obj',      mat_id='mat_white', use_face_normal=False))
    scene.set('green',    mesh_from_obj(f'{CBOX}/cbox_greenwall.obj', mat_id='mat_green', use_face_normal=False))
    scene.set('red',      mesh_from_obj(f'{CBOX}/cbox_redwall.obj',   mat_id='mat_red',   use_face_normal=False))
    return scene

# ---------------------------------------------------------------------------
# PathReSTIR renderer: PathTracer forward (full-path images for loss),
# PathReSTIR backward (low-variance gradient via spatial resampling).
#
# PathReSTIR::renderC() is a direct-illumination-only stub (needed for
# silhouette/boundary edges). The full multi-bounce forward path lives
# inside renderD's primal+cache pass. Using PathTracer for renderC gives
# correct full-path images for loss; PathReSTIR renderD gives the
# low-variance adjoint gradient.
# ---------------------------------------------------------------------------

class PathReSTIRFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, connector, scene, fwd_options, bwd_options,
                sensor_ids, fwd_integrator_id, bwd_integrator_id, *params):
        images = connector.renderC(scene, fwd_options,
                                   sensor_ids=sensor_ids,
                                   integrator_id=fwd_integrator_id)
        images = torch.nan_to_num(torch.stack(images, dim=0))
        ctx.connector        = connector
        ctx.scene            = scene
        ctx.bwd_options      = bwd_options
        ctx.sensor_ids       = sensor_ids
        ctx.bwd_integrator_id = bwd_integrator_id
        ctx.num_no_grads     = 7
        return images.cpu()

    @staticmethod
    def backward(ctx, grad_out):
        image_grads = [g.to(configs['device']) for g in grad_out]
        param_grads = ctx.connector.renderD(
            image_grads, ctx.scene, ctx.bwd_options,
            ctx.sensor_ids, ctx.bwd_integrator_id)
        return tuple([None] * ctx.num_no_grads + param_grads)


class PathReSTIRRenderer(torch.nn.Module):
    """PathTracer renderC (spp=16 clean images) + PathReSTIR renderD (spp=2 gradient)."""
    def __init__(self, fwd_options, bwd_options,
                 fwd_integrator_id=0, bwd_integrator_id='integrator_restir'):
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
        return PathReSTIRFunction.apply(
            self.connector, scene, self.fwd_options, self.bwd_options,
            sensor_ids, self.fwd_integrator_id, self.bwd_integrator_id,
            *params
        ).to(configs['device'])


# ---------------------------------------------------------------------------
# Spatial box filter (for ∂I/∂Y visualization only — not part of optimization)
# ---------------------------------------------------------------------------

def spatial_box_filter(image, radius):
    h, w, c = image.shape
    x = image.permute(2, 0, 1).unsqueeze(0)
    ks = 2 * radius + 1
    weight = torch.ones(c, 1, ks, ks, device=image.device, dtype=image.dtype) / (ks * ks)
    x_pad = F.pad(x, [radius, radius, radius, radius], mode='replicate')
    return F.conv2d(x_pad, weight, groups=c).squeeze(0).permute(1, 2, 0)

# ---------------------------------------------------------------------------
# ∂I/∂Y via finite difference at the current box position.
# Uses a fresh temporary scene + plain spp=2 PathTracer (visualization only).
# ---------------------------------------------------------------------------

GRAD_VIS_SPP = {'spp': 2, 'sppe': 0, 'sppse': 0}

def compute_dI_dY(renderer, V_cur, eps=3.0):
    scene_tmp = build_scene(use_restir=False)
    scene_tmp['smallbox']['v'] = V_cur + to_torch_f([[0., eps, 0.]])
    scene_tmp.configure()
    with torch.no_grad():
        I_plus = renderer(scene_tmp, sensor_ids=[0], integrator_id=0)[0].detach().cpu()

    scene_tmp['smallbox']['v'] = V_cur - to_torch_f([[0., eps, 0.]])
    scene_tmp.configure()
    with torch.no_grad():
        I_minus = renderer(scene_tmp, sensor_ids=[0], integrator_id=0)[0].detach().cpu()

    del scene_tmp
    dI_dY = (I_plus - I_minus) / (2.0 * eps)
    mag = (0.299 * dI_dY[:, :, 0].abs()
         + 0.587 * dI_dY[:, :, 1].abs()
         + 0.114 * dI_dY[:, :, 2].abs()).numpy()
    return mag   # [H, W]

# ---------------------------------------------------------------------------
# Crop helper — zoom into small-box region of the 512×512 render
# ---------------------------------------------------------------------------

CROP = (160, 460, 90, 390)

def crop(img, box=CROP):
    r0, r1, c0, c1 = box
    return img[r0:r1, c0:c1] if img.ndim == 2 else img[r0:r1, c0:c1, :]

# ---------------------------------------------------------------------------
# SPP settings
# ---------------------------------------------------------------------------

OPT_SPP    = {'spp': 16, 'sppe': 16, 'sppse': 8}
RESTIR_SPP = {'spp':  1, 'sppe':  1, 'sppse': 1}  # was spp=2 — SPP=1 + n_candidates=4 resampling
VIS_SPP    = {'spp': 64, 'sppe':  0, 'sppse': 0}

# ---------------------------------------------------------------------------
# Optimization runner
#
# vis_integrator_id: integrator used for the 64-spp VIS frames.
#   Standard  → 0  (the only PathTracer integrator)
#   PathReSTIR → 'integrator_vis'  (the secondary PathTracer in the scene)
# ---------------------------------------------------------------------------

def _ts():
    return datetime.datetime.now().strftime('%H:%M:%S')

def run_opt(scene, render_opt, num_iters, save_every, target, V_gt,
            perturb_y=70.0, label='OPT', lr=5.0,
            vis_integrator_id=0,
            render_grad=None, grad_eps=3.0):
    """Returns (vis_frames, grad_frames, losses, iter_times)."""
    render_vis = Renderer('psdr_jit', render_options={**VIS_SPP, 'npass': 1, 'log_level': 0})

    scene['smallbox']['v'] = V_gt + to_torch_f([[0., perturb_y, 0.]])
    model = ShapeLS(scene, mesh_id='smallbox', optimizer_kwargs={'lr': lr, 'lmbda': 1})

    vis_frames, grad_frames, losses, iter_times = [], [], [], []
    gt_mean_y = V_gt[:, 1].mean().item()

    print(f"[{_ts()}] [{label}] starting {num_iters} iters  lr={lr}", flush=True)

    pbar = tqdm(total=num_iters, desc=label, leave=True)
    for it in range(num_iters):
        t0 = time.perf_counter()

        model.zero_grad()
        model.set_data()
        scene.configure()

        opt_image  = render_opt(scene, sensor_ids=[0], integrator_id=0)[0]
        image_loss = l1_loss(target, opt_image)
        reg_loss   = model.get_regularization()
        (image_loss + reg_loss).backward()
        model.step()
        torch.cuda.synchronize()
        drjit.flush_malloc_cache()

        dt = time.perf_counter() - t0
        iter_times.append(dt)
        losses.append(image_loss.item())

        cur_y = scene['smallbox.v'].detach()[:, 1].mean().item()
        pbar.update(1)
        pbar.set_postfix({'L1': f'{image_loss.item():.5f}',
                          'Y':  f'{cur_y:.1f}/{gt_mean_y:.1f}',
                          'ms': f'{dt*1000:.0f}'})

        if it % save_every == 0:
            print(f"[{_ts()}] [{label}] iter {it:3d}/{num_iters}  "
                  f"L1={image_loss.item():.5f}  Y={cur_y:.1f}/{gt_mean_y:.1f}  "
                  f"{dt*1000:.0f}ms/it — vis render…", flush=True)
            with torch.no_grad():
                frame = render_vis(scene, sensor_ids=[0],
                                   integrator_id=vis_integrator_id)[0]
            vis_frames.append(
                crop(np.clip(frame.cpu().numpy() ** (1/2.2), 0, 1)))

            if render_grad is not None:
                print(f"[{_ts()}] [{label}] iter {it:3d}  grad vis…", flush=True)
                V_cur = scene['smallbox.v'].detach().clone()
                g = compute_dI_dY(render_grad, V_cur, eps=grad_eps)
                grad_frames.append(crop(g))

    pbar.close()
    print(f"[{_ts()}] [{label}] done.  final L1={losses[-1]:.5f}  Y={cur_y:.1f}", flush=True)
    return vis_frames, grad_frames, losses, iter_times

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir   = Path('shape_opt_cornell')
    out_dir.mkdir(exist_ok=True)
    gif_path  = 'shape_opt_cornell.gif'
    json_path = str(out_dir / 'results.json')

    num_iters  = 50
    save_every = 5
    perturb_y  = 70.0
    GRAD_EPS   = 3.0

    print(f"[{_ts()}] === starting run: {num_iters} iters, 3 methods ===", flush=True)
    print(f"[{_ts()}]   Standard:    spp=16  lr=5.0", flush=True)
    print(f"[{_ts()}]   ReSTIR-cand4: spp=1  n_candidates=4  lr=3.0", flush=True)
    print(f"[{_ts()}]   ReSTIR-cand8: spp=1  n_candidates=8  lr=3.0", flush=True)

    # ---- Ground-truth target ----
    print(f"\n[{_ts()}] rendering GT target…", flush=True)
    scene_gt = build_scene()
    render_gt = Renderer('psdr_jit', render_options={**VIS_SPP, 'npass': 1, 'log_level': 0})
    target = render_gt(scene_gt, sensor_ids=[0])[0]
    write_image(str(out_dir / 'target.exr'), target)
    V_gt      = scene_gt['smallbox.v'].detach().clone()
    gt_mean_y = V_gt[:, 1].mean().item()
    print(f"  GT small box mean Y: {gt_mean_y:.1f}")
    target_np_crop = crop(np.clip(target.cpu().numpy() ** (1/2.2), 0, 1))
    del scene_gt

    rnd_grad_vis = Renderer('psdr_jit',
                            render_options={**GRAD_VIS_SPP, 'npass': 1, 'log_level': 0})

    # ---- Standard PathTracer ----
    lr_std = 5.0
    print(f"\n[{_ts()}] --- run 1/3: Standard PathTracer (spp=16, lr={lr_std}) ---", flush=True)
    scene_std  = build_scene(use_restir=False)
    render_std = Renderer('psdr_jit', render_options={**OPT_SPP, 'npass': 1, 'log_level': 0})
    vis_std, grad_std, losses_std, times_std = run_opt(
        scene_std, render_std, num_iters, save_every, target, V_gt,
        perturb_y=perturb_y, label='Standard', lr=lr_std,
        vis_integrator_id=0,
        render_grad=rnd_grad_vis, grad_eps=GRAD_EPS)
    fps_std     = 1.0 / np.mean(times_std[5:])
    final_y_std = scene_std['smallbox.v'].detach()[:, 1].mean().item()
    print(f"  Standard:    {1000*np.mean(times_std[5:]):.0f} ms/iter  "
          f"({fps_std:.3f} it/s)  final L1={losses_std[-1]:.5f}  Y={final_y_std:.1f}")

    # ---- PathReSTIR spp=1, n_candidates=4 ----
    lr_restir = 3.0
    print(f"\n[{_ts()}] --- run 2/3: PathReSTIR spp=1, n_candidates=4 (lr={lr_restir}) ---", flush=True)
    scene_r4  = build_scene(use_restir=True, restir_params=RESTIR_PARAMS_4)
    render_r4 = PathReSTIRRenderer(
        fwd_options={**OPT_SPP,    'npass': 1, 'log_level': 0},
        bwd_options={**RESTIR_SPP, 'npass': 1, 'log_level': 0},
        fwd_integrator_id=0,
        bwd_integrator_id='integrator_restir'
    )
    vis_r4, grad_r4, losses_r4, times_r4 = run_opt(
        scene_r4, render_r4, num_iters, save_every, target, V_gt,
        perturb_y=perturb_y, label='ReSTIR-cand4', lr=lr_restir,
        vis_integrator_id='integrator_vis',
        render_grad=rnd_grad_vis, grad_eps=GRAD_EPS)
    fps_r4     = 1.0 / np.mean(times_r4[5:])
    final_y_r4 = scene_r4['smallbox.v'].detach()[:, 1].mean().item()
    print(f"  ReSTIR-4:    {1000*np.mean(times_r4[5:]):.0f} ms/iter  "
          f"({fps_r4:.3f} it/s)  final L1={losses_r4[-1]:.5f}  Y={final_y_r4:.1f}")

    # ---- PathReSTIR spp=1, n_candidates=8 ----
    print(f"\n[{_ts()}] --- run 3/3: PathReSTIR spp=1, n_candidates=8 (lr={lr_restir}) ---", flush=True)
    scene_r8  = build_scene(use_restir=True, restir_params=RESTIR_PARAMS_8)
    render_r8 = PathReSTIRRenderer(
        fwd_options={**OPT_SPP,    'npass': 1, 'log_level': 0},
        bwd_options={**RESTIR_SPP, 'npass': 1, 'log_level': 0},
        fwd_integrator_id=0,
        bwd_integrator_id='integrator_restir'
    )
    vis_r8, grad_r8, losses_r8, times_r8 = run_opt(
        scene_r8, render_r8, num_iters, save_every, target, V_gt,
        perturb_y=perturb_y, label='ReSTIR-cand8', lr=lr_restir,
        vis_integrator_id='integrator_vis',
        render_grad=rnd_grad_vis, grad_eps=GRAD_EPS)
    fps_r8     = 1.0 / np.mean(times_r8[5:])
    final_y_r8 = scene_r8['smallbox.v'].detach()[:, 1].mean().item()
    print(f"  ReSTIR-8:    {1000*np.mean(times_r8[5:]):.0f} ms/iter  "
          f"({fps_r8:.3f} it/s)  final L1={losses_r8[-1]:.5f}  Y={final_y_r8:.1f}")

    # ---- Save numerical results ----
    print(f"\n[{_ts()}] all runs complete — saving results", flush=True)
    results = {
        'gt_y':      gt_mean_y,
        'num_iters': num_iters,
        'standard': {
            'spp': OPT_SPP['spp'], 'lr': lr_std,
            'final_l1': losses_std[-1], 'final_y': final_y_std, 'fps': fps_std,
            'losses': losses_std,
        },
        'restir_cand4': {
            'spp': RESTIR_SPP['spp'], 'lr': lr_restir,
            **{k: RESTIR_PARAMS_4[k] for k in ('n_candidates', 'n_neighbors', 'spatial_radius')},
            'final_l1': losses_r4[-1], 'final_y': final_y_r4, 'fps': fps_r4,
            'losses': losses_r4,
        },
        'restir_cand8': {
            'spp': RESTIR_SPP['spp'], 'lr': lr_restir,
            **{k: RESTIR_PARAMS_8[k] for k in ('n_candidates', 'n_neighbors', 'spatial_radius')},
            'final_l1': losses_r8[-1], 'final_y': final_y_r8, 'fps': fps_r8,
            'losses': losses_r8,
        },
    }
    with open(json_path, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"\nNumerical results → {json_path}")

    # ---- Normalize gradient images ----
    all_grad = grad_std + grad_r4 + grad_r8
    vmax_g   = max(f.max() for f in all_grad) * 0.95 if all_grad else 1.0

    # ---- Build GIF (4-column layout) ----
    print(f"Building GIF    → {gif_path}")
    n_frames = min(len(vis_std), len(vis_r4), len(vis_r8),
                   len(grad_std), len(grad_r4), len(grad_r8))

    fig = plt.figure(figsize=(26, 14), facecolor='black')
    gs  = fig.add_gridspec(3, 4,
                           height_ratios=[2.5, 2.5, 1.6],
                           hspace=0.40, wspace=0.08,
                           left=0.03, right=0.97, top=0.93, bottom=0.05)

    # Row 0 — VIS renders (titles are static: final Y shown once)
    ax_tgt = fig.add_subplot(gs[0, 0])
    ax_tgt.axis('off'); ax_tgt.set_facecolor('black')
    ax_tgt.imshow(target_np_crop)
    ax_tgt.set_title(f'Target  (GT, Y={gt_mean_y:.1f})', color='white', fontsize=9)

    ax_vis_std = fig.add_subplot(gs[0, 1])
    ax_vis_std.axis('off'); ax_vis_std.set_facecolor('black')
    im_vis_std = ax_vis_std.imshow(vis_std[0])
    ax_vis_std.set_title(f'Standard  spp=16  (Y_final={final_y_std:.1f})',
                         color='#88ccff', fontsize=9)

    ax_vis_r4 = fig.add_subplot(gs[0, 2])
    ax_vis_r4.axis('off'); ax_vis_r4.set_facecolor('black')
    im_vis_r4 = ax_vis_r4.imshow(vis_r4[0])
    ax_vis_r4.set_title(f'ReSTIR  spp=1 cand=4  (Y_final={final_y_r4:.1f})',
                        color='#ffaa44', fontsize=9)

    ax_vis_r8 = fig.add_subplot(gs[0, 3])
    ax_vis_r8.axis('off'); ax_vis_r8.set_facecolor('black')
    im_vis_r8 = ax_vis_r8.imshow(vis_r8[0])
    ax_vis_r8.set_title(f'ReSTIR  spp=1 cand=8  (Y_final={final_y_r8:.1f})',
                        color='#ff6688', fontsize=9)

    # Row 1 — ∂I/∂Y gradient (animated)
    ax_glabel = fig.add_subplot(gs[1, 0])
    ax_glabel.axis('off'); ax_glabel.set_facecolor('black')
    ax_glabel.text(0.5, 0.5,
        '∂I / ∂Y_global\n(finite difference\nat current position)\n\nBright = sensitive\nto box height\n\nspp=2 PathTracer',
        ha='center', va='center', color='#aaa', fontsize=9,
        transform=ax_glabel.transAxes)

    ax_g_std = fig.add_subplot(gs[1, 1])
    ax_g_std.axis('off'); ax_g_std.set_facecolor('black')
    im_g_std  = ax_g_std.imshow(grad_std[0], cmap='inferno', vmin=0, vmax=vmax_g)
    ttl_g_std = ax_g_std.set_title('Standard  ∂I/∂Y', color='#88ccff', fontsize=9)

    ax_g_r4 = fig.add_subplot(gs[1, 2])
    ax_g_r4.axis('off'); ax_g_r4.set_facecolor('black')
    im_g_r4  = ax_g_r4.imshow(grad_r4[0], cmap='inferno', vmin=0, vmax=vmax_g)
    ttl_g_r4 = ax_g_r4.set_title('ReSTIR-4  ∂I/∂Y', color='#ffaa44', fontsize=9)

    ax_g_r8 = fig.add_subplot(gs[1, 3])
    ax_g_r8.axis('off'); ax_g_r8.set_facecolor('black')
    im_g_r8  = ax_g_r8.imshow(grad_r8[0], cmap='inferno', vmin=0, vmax=vmax_g)
    ttl_g_r8 = ax_g_r8.set_title('ReSTIR-8  ∂I/∂Y', color='#ff6688', fontsize=9)

    # Row 2 — loss curve + speed bar
    ax_loss = fig.add_subplot(gs[2, :3])
    ax_fps  = fig.add_subplot(gs[2, 3])
    for ax in [ax_loss, ax_fps]:
        ax.set_facecolor('#111')
        ax.tick_params(colors='white', labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor('#555')

    ymax = max(max(losses_std), max(losses_r4), max(losses_r8)) * 1.05
    ax_loss.set_title('L1 loss vs iteration', color='white', fontsize=10)
    ax_loss.set_xlabel('iteration', color='white', fontsize=9)
    ax_loss.set_xlim(0, num_iters); ax_loss.set_ylim(0, ymax)
    ax_loss.yaxis.grid(True, color='#333', zorder=0)
    line_std, = ax_loss.plot([], [], color='#88ccff', lw=2,
                              label=f'Standard  spp=16  lr={lr_std}')
    line_r4,  = ax_loss.plot([], [], color='#ffaa44', lw=2,
                              label=f'ReSTIR  spp=1  cand=4  lr={lr_restir}')
    line_r8,  = ax_loss.plot([], [], color='#ff6688', lw=2,
                              label=f'ReSTIR  spp=1  cand=8  lr={lr_restir}')
    ax_loss.legend(facecolor='#222', edgecolor='#555', labelcolor='white', fontsize=8)

    fps_vals = [fps_std, fps_r4, fps_r8]
    bars = ax_fps.bar(
        ['Standard\nspp=16', 'ReSTIR\ncand=4', 'ReSTIR\ncand=8'],
        fps_vals,
        color=['#88ccff', '#ffaa44', '#ff6688'], width=0.5)
    ax_fps.set_title('Speed (it/s)', color='white', fontsize=10)
    ax_fps.yaxis.grid(True, color='#333', zorder=0)
    for bar, v in zip(bars, fps_vals):
        ax_fps.text(bar.get_x() + bar.get_width() / 2,
                    v + max(fps_vals) * 0.02,
                    f'{v:.3f}', ha='center', va='bottom', color='white', fontsize=8)

    suptitle = fig.suptitle('iter 0', color='white', fontsize=12)

    def update(i):
        it = i * save_every
        im_vis_std.set_data(vis_std[i])
        im_vis_r4.set_data(vis_r4[i])
        im_vis_r8.set_data(vis_r8[i])
        im_g_std.set_data(grad_std[i])
        im_g_r4.set_data(grad_r4[i])
        im_g_r8.set_data(grad_r8[i])

        ttl_g_std.set_text(f'Standard  ∂I/∂Y  (iter {it})')
        ttl_g_r4.set_text(f'ReSTIR-4  ∂I/∂Y  (iter {it})')
        ttl_g_r8.set_text(f'ReSTIR-8  ∂I/∂Y  (iter {it})')

        xs  = list(range(min(it + 1, len(losses_std))))
        line_std.set_data(xs, losses_std[:len(xs)])
        xr4 = list(range(min(it + 1, len(losses_r4))))
        line_r4.set_data(xr4, losses_r4[:len(xr4)])
        xr8 = list(range(min(it + 1, len(losses_r8))))
        line_r8.set_data(xr8, losses_r8[:len(xr8)])

        l_std = losses_std[min(it, len(losses_std) - 1)]
        l_r4  = losses_r4[min(it, len(losses_r4) - 1)]
        l_r8  = losses_r8[min(it, len(losses_r8) - 1)]
        suptitle.set_text(
            f'Iter {it:3d}/{num_iters}   GT Y={gt_mean_y:.1f}   '
            f'Std={l_std:.5f}   R4={l_r4:.5f}   R8={l_r8:.5f}')
        return (im_vis_std, im_vis_r4, im_vis_r8,
                im_g_std, im_g_r4, im_g_r8,
                ttl_g_std, ttl_g_r4, ttl_g_r8,
                line_std, line_r4, line_r8, suptitle)

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=300, blit=True)
    anim.save(gif_path, writer='pillow', fps=4, dpi=120)
    plt.close()
    print(f"Saved → {gif_path}")

    print("\n=== Summary ===")
    print(f"  Standard      spp=16  lr={lr_std}:    "
          f"L1={losses_std[-1]:.5f}  Y={final_y_std:.1f}  {fps_std:.3f} it/s")
    print(f"  ReSTIR cand=4 spp=1   lr={lr_restir}: "
          f"L1={losses_r4[-1]:.5f}  Y={final_y_r4:.1f}  {fps_r4:.3f} it/s")
    print(f"  ReSTIR cand=8 spp=1   lr={lr_restir}: "
          f"L1={losses_r8[-1]:.5f}  Y={final_y_r8:.1f}  {fps_r8:.3f} it/s")
    print(f"  GT Y={gt_mean_y:.1f}")
    print(f"\n  Numerical results → {json_path}")


if __name__ == '__main__':
    main()
