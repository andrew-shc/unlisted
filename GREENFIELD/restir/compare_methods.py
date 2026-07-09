#!/usr/bin/env python3
"""
Side-by-side comparison of Standard PathTracer vs PathReSTIR on the Cornell
box shape optimization.

Layout:
  ┌──────────────────┬──────────────────┐
  │  Standard spp=16  │  ReSTIR spp=1     │
  │  (rendered image)  │  (rendered image)  │
  ├──────────────────┴──────────────────┤
  │      L1 Loss vs Iteration            │
  │  (overlaid lines, labeled)          │
  └─────────────────────────────────────┘

Usage:
  conda activate metrology_ir
  PYTHONUNBUFFERED=1 python GREENFIELD/restir/compare_methods.py
  PYTHONUNBUFFERED=1 python GREENFIELD/restir/compare_methods.py --iters 10 --cand 8

Output:
  ASSETS/restir/comparison/
    comparison.gif         — animated GIF
    comparison_final.png   — static final frame
    results.json           — numerical results
"""

import os
import sys
import time
import json
import datetime
import argparse
from pathlib import Path

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import numpy as np
import torch
import torch.nn.functional as F
import psdr_jit
import drjit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

# --- Paths ---
REPO_ROOT = Path(__file__).resolve().parents[2]
NPBIR_PBIR = str(REPO_ROOT / 'DigitalTwinCatalog' / 'neural_pbir' / 'pbir')
if NPBIR_PBIR not in sys.path:
    sys.path.insert(0, NPBIR_PBIR)

CBOX = str(REPO_ROOT / 'psdr-jit' / 'tutorials' / 'data' / 'cbox')

# WandB
try:
    from dotenv import load_dotenv
    import wandb
    load_dotenv(REPO_ROOT / ".env")
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

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
# ---------------------------------------------------------------------------
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
# Mesh loading helpers
# ---------------------------------------------------------------------------

def load_obj_tri(path):
    verts, uvs, faces, faces_uv = [], [], [], []
    with open(path) as fh:
        for line in fh:
            t = line.split()
            if not t:
                continue
            if t[0] == 'v':
                verts.append([float(x) for x in t[1:4]])
            elif t[0] == 'vt':
                uvs.append([float(x) for x in t[1:3]])
            elif t[0] == 'f':
                vi, ti = [], []
                for tok in t[1:]:
                    p = tok.split('/')
                    vi.append(int(p[0]) - 1)
                    ti.append(int(p[1]) - 1 if len(p) > 1 and p[1] else 0)
                for i in range(1, len(vi) - 1):
                    faces.append([vi[0], vi[i], vi[i + 1]])
                    faces_uv.append([ti[0], ti[i], ti[i + 1]])
    v = np.array(verts, dtype=np.float32)
    f = np.array(faces, dtype=np.int32)
    uv = np.array(uvs, dtype=np.float32) if uvs else np.zeros((len(verts), 2), np.float32)
    fuv = np.array(faces_uv, dtype=np.int32) if uvs else f.copy()
    return v, f, uv, fuv

def mesh_from_obj(path, use_face_normal=False, **kwargs):
    v, f, uv, fuv = load_obj_tri(path)
    kwargs.setdefault('can_change_topology', True)
    return Mesh(v, f, uv, fuv, use_face_normal=use_face_normal, **kwargs)

# ---------------------------------------------------------------------------
# Scene factory
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
        # 'integrator_vis'    → PathTracer for VIS renders (PathReSTIR::Li is direct-only)
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
# ---------------------------------------------------------------------------

class PathReSTIRFunction(torch.autograd.Function):
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
        return images.cpu()

    @staticmethod
    def backward(ctx, grad_out):
        image_grads = [g.to(configs['device']) for g in grad_out]
        param_grads = ctx.connector.renderD(
            image_grads, ctx.scene, ctx.bwd_options,
            ctx.sensor_ids, ctx.bwd_integrator_id)
        return tuple([None] * ctx.num_no_grads + param_grads)


class PathReSTIRRenderer(torch.nn.Module):
    """PathTracer renderC (clean full-path images) + PathReSTIR renderD (gradient)."""
    def __init__(self, fwd_options, bwd_options,
                 fwd_integrator_id=0, bwd_integrator_id='integrator_restir'):
        super().__init__()
        self.connector          = get_connector('psdr_jit')
        self.fwd_options        = fwd_options
        self.bwd_options        = bwd_options
        self.fwd_integrator_id  = fwd_integrator_id
        self.bwd_integrator_id  = bwd_integrator_id

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
        ).to(configs['device'])

# ---------------------------------------------------------------------------
# SPP settings
# ---------------------------------------------------------------------------

VIS_SPP = {'spp': 64, 'sppe': 0, 'sppse': 0}

# ---------------------------------------------------------------------------
# Crop helper — zoom into small-box region of the 512×512 render
# ---------------------------------------------------------------------------

CROP = (160, 460, 90, 390)

def crop(img, box=CROP):
    r0, r1, c0, c1 = box
    return img[r0:r1, c0:c1] if img.ndim == 2 else img[r0:r1, c0:c1, :]

# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------

def _ts():
    return datetime.datetime.now().strftime('%H:%M:%S')

# ---------------------------------------------------------------------------
# Optimization runner
#
# vis_integrator_id: integrator used for the 64-spp VIS frames.
#   Standard  → 0  (the only PathTracer integrator)
#   PathReSTIR → 'integrator_vis'  (the secondary PathTracer in the scene)
# ---------------------------------------------------------------------------

def run_opt(scene, render_opt, num_iters, save_every, target, V_gt,
            perturb_y=70.0, label='OPT', lr=5.0,
            vis_integrator_id=0):
    """Returns (vis_frames, losses, iter_times)."""
    render_vis = Renderer('psdr_jit', render_options={**VIS_SPP, 'npass': 1, 'log_level': 0})

    scene['smallbox']['v'] = V_gt + to_torch_f([[0., perturb_y, 0.]])
    model = ShapeLS(scene, mesh_id='smallbox', optimizer_kwargs={'lr': lr, 'lmbda': 1})

    vis_frames, losses, iter_times = [], [], []
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

        # WandB per-iteration logging
        if _HAS_WANDB and wandb.run:
            wandb.log({
                f"{label}/loss":        image_loss.item(),
                f"{label}/y_position":  cur_y,
                f"{label}/iter_time_ms": dt * 1000,
            })

        if it % save_every == 0:
            print(f"[{_ts()}] [{label}] iter {it:3d}/{num_iters}  "
                  f"L1={image_loss.item():.5f}  Y={cur_y:.1f}/{gt_mean_y:.1f}  "
                  f"{dt*1000:.0f}ms/it — vis render…", flush=True)
            with torch.no_grad():
                frame = render_vis(scene, sensor_ids=[0],
                                   integrator_id=vis_integrator_id)[0]
            vis_frames.append(
                crop(np.clip(frame.cpu().numpy() ** (1 / 2.2), 0, 1)))

    pbar.close()
    print(f"[{_ts()}] [{label}] done.  final L1={losses[-1]:.5f}  Y={cur_y:.1f}", flush=True)
    return vis_frames, losses, iter_times

# ---------------------------------------------------------------------------
# GIF figure setup
# ---------------------------------------------------------------------------

def setup_figure(num_iters, gt_mean_y, final_y_std, final_y_restir,
                 lr, restir_cand, restir_spp, spp):
    """Build the 2×2 matplotlib figure and return (fig, im_std, im_restir,
    line_std, line_restir, ax_loss, suptitle)."""
    fig = plt.figure(figsize=(14, 10), facecolor='black')
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[2.5, 1.6],
        hspace=0.35, wspace=0.05,
        left=0.05, right=0.95, top=0.92, bottom=0.07,
    )

    # --- Top row: side-by-side rendered images ---
    ax_std = fig.add_subplot(gs[0, 0])
    ax_std.axis('off')
    ax_std.set_facecolor('black')
    im_std = ax_std.imshow(np.zeros((300, 300, 3)))
    ax_std.set_title(f'Standard  spp={spp}  (Y_final={final_y_std:.1f})',
                     color='#88ccff', fontsize=11)

    ax_restir = fig.add_subplot(gs[0, 1])
    ax_restir.axis('off')
    ax_restir.set_facecolor('black')
    im_restir = ax_restir.imshow(np.zeros((300, 300, 3)))
    ax_restir.set_title(f'ReSTIR  spp={restir_spp}  cand={restir_cand}  '
                        f'(Y_final={final_y_restir:.1f})',
                        color='#ffaa44', fontsize=11)

    # --- Bottom row: L1 loss overlaid curves ---
    ax_loss = fig.add_subplot(gs[1, :])
    ax_loss.set_facecolor('#111')
    ax_loss.tick_params(colors='white', labelsize=9)
    for sp in ax_loss.spines.values():
        sp.set_edgecolor('#555')

    ax_loss.set_title('L1 Loss vs Iteration', color='white', fontsize=12)
    ax_loss.set_xlabel('Iteration', color='white', fontsize=10)
    ax_loss.set_ylabel('L1 Loss', color='white', fontsize=10)
    ax_loss.set_xlim(0, num_iters)
    ax_loss.set_ylim(0, 1)  # will be rescaled after collecting losses
    ax_loss.yaxis.grid(True, color='#333', zorder=0)

    line_std, = ax_loss.plot([], [], color='#88ccff', lw=2,
                              label=f'Standard  spp={spp}  lr={lr}')
    line_restir, = ax_loss.plot([], [], color='#ffaa44', lw=2,
                                 label=f'ReSTIR  spp={restir_spp}  cand={restir_cand}  lr={lr}')
    ax_loss.legend(facecolor='#222', edgecolor='#555', labelcolor='white', fontsize=9)

    suptitle = fig.suptitle('Iter 0', color='white', fontsize=13)

    return fig, im_std, im_restir, line_std, line_restir, ax_loss, suptitle

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(num_iters=50, restir_cand=4, spp=16, restir_spp=1, run_name=None):
    out_dir = REPO_ROOT / 'ASSETS' / 'restir' / 'comparison'
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path  = out_dir / 'comparison.gif'
    png_path  = out_dir / 'comparison_final.png'
    json_path = out_dir / 'results.json'

    save_every = 5
    perturb_y  = 70.0
    lr         = 5.0  # same lr for both methods

    restir_params = RESTIR_PARAMS_8 if restir_cand == 8 else RESTIR_PARAMS_4

    # WandB init
    if _HAS_WANDB:
        wandb_name = run_name or f"compare_methods/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "ReSTIR PBIR"),
            entity=os.environ.get("WANDB_ENTITY"),
            dir=os.environ.get("WANDB_DIR", str(REPO_ROOT / "ASSETS" / "wandb")),
            name=wandb_name,
            config={
                "method":          "compare_methods",
                "num_iters":       num_iters,
                "save_every":      save_every,
                "standard_spp":    spp,
                "restir_spp":      restir_spp,
                "restir_cand":     restir_cand,
                "lr":              lr,
            },
        )

    print(f"[{_ts()}] === starting run: {num_iters} iters, 2 methods ===", flush=True)
    print(f"[{_ts()}]   Standard:    spp={spp}  lr={lr}", flush=True)
    print(f"[{_ts()}]   ReSTIR:      spp={restir_spp}  n_candidates={restir_cand}  lr={lr}", flush=True)

    # ---- Ground-truth target ----
    print(f"\n[{_ts()}] rendering GT target…", flush=True)
    scene_gt = build_scene()
    render_gt = Renderer('psdr_jit', render_options={**VIS_SPP, 'npass': 1, 'log_level': 0})
    target = render_gt(scene_gt, sensor_ids=[0])[0]
    write_image(str(out_dir / 'target.exr'), target)
    V_gt      = scene_gt['smallbox.v'].detach().clone()
    gt_mean_y = V_gt[:, 1].mean().item()
    print(f"  GT small box mean Y: {gt_mean_y:.1f}")
    del scene_gt

    # ---- Standard PathTracer ----
    print(f"\n[{_ts()}] --- run 1/2: Standard PathTracer (spp={spp}, lr={lr}) ---", flush=True)
    scene_std  = build_scene(use_restir=False)
    render_std = Renderer('psdr_jit', render_options={'spp': spp, 'sppe': spp, 'sppse': 8,
                                                       'npass': 1, 'log_level': 0})
    vis_std, losses_std, times_std = run_opt(
        scene_std, render_std, num_iters, save_every, target, V_gt,
        perturb_y=perturb_y, label='Standard', lr=lr,
        vis_integrator_id=0)
    fps_std     = 1.0 / np.mean(times_std[5:])
    final_y_std = scene_std['smallbox.v'].detach()[:, 1].mean().item()
    print(f"  Standard:    {1000 * np.mean(times_std[5:]):.0f} ms/iter  "
          f"({fps_std:.3f} it/s)  final L1={losses_std[-1]:.5f}  Y={final_y_std:.1f}")

    # ---- PathReSTIR ----
    print(f"\n[{_ts()}] --- run 2/2: PathReSTIR (spp={restir_spp}, cand={restir_cand}, lr={lr}) ---", flush=True)
    scene_restir  = build_scene(use_restir=True, restir_params=restir_params)
    render_restir = PathReSTIRRenderer(
        fwd_options={'spp': spp, 'sppe': spp, 'sppse': 8, 'npass': 1, 'log_level': 0},
        bwd_options={'spp': restir_spp, 'sppe': restir_spp, 'sppse': restir_spp,
                     'npass': 1, 'log_level': 0},
        fwd_integrator_id=0,
        bwd_integrator_id='integrator_restir'
    )
    vis_restir, losses_restir, times_restir = run_opt(
        scene_restir, render_restir, num_iters, save_every, target, V_gt,
        perturb_y=perturb_y, label='ReSTIR', lr=lr,
        vis_integrator_id='integrator_vis')
    fps_restir     = 1.0 / np.mean(times_restir[5:])
    final_y_restir = scene_restir['smallbox.v'].detach()[:, 1].mean().item()
    print(f"  ReSTIR:     {1000 * np.mean(times_restir[5:]):.0f} ms/iter  "
          f"({fps_restir:.3f} it/s)  final L1={losses_restir[-1]:.5f}  Y={final_y_restir:.1f}")

    # ---- Save numerical results ----
    print(f"\n[{_ts()}] all runs complete — saving results", flush=True)
    results = {
        'gt_y':      gt_mean_y,
        'num_iters': num_iters,
        'standard': {
            'spp': spp, 'lr': lr,
            'final_l1': losses_std[-1], 'final_y': final_y_std, 'fps': fps_std,
            'losses': losses_std,
        },
        'restir': {
            'spp': restir_spp, 'lr': lr,
            **{k: restir_params[k] for k in ('n_candidates', 'n_neighbors', 'spatial_radius')},
            'final_l1': losses_restir[-1], 'final_y': final_y_restir, 'fps': fps_restir,
            'losses': losses_restir,
        },
    }
    with open(json_path, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"Numerical results → {json_path}")

    # Log summary to WandB
    if _HAS_WANDB and wandb.run:
        wandb.log({
            "summary/standard_final_l1":    losses_std[-1],
            "summary/standard_final_y":     final_y_std,
            "summary/standard_fps":         fps_std,
            "summary/restir_final_l1":      losses_restir[-1],
            "summary/restir_final_y":       final_y_restir,
            "summary/restir_fps":           fps_restir,
            "summary/gt_y":                 gt_mean_y,
        })
        wandb.save(str(json_path))

    # ---- Determine y-limits for loss plot ----
    all_losses = losses_std + losses_restir
    ymax = max(all_losses) * 1.05 if all_losses else 1.0
    ymin = 0.0

    # ---- Build GIF (2-row layout) ----
    print(f"Building GIF  → {gif_path}")
    n_frames = min(len(vis_std), len(vis_restir))

    fig, im_std, im_restir, line_std, line_restir, ax_loss, suptitle = setup_figure(
        num_iters, gt_mean_y, final_y_std, final_y_restir,
        lr, restir_cand, restir_spp, spp)
    ax_loss.set_ylim(ymin, ymax)

    def update(i):
        it = i * save_every

        im_std.set_data(vis_std[i])
        im_restir.set_data(vis_restir[i])

        xs = list(range(min(it + 1, len(losses_std))))
        line_std.set_data(xs, losses_std[:len(xs)])

        xr = list(range(min(it + 1, len(losses_restir))))
        line_restir.set_data(xr, losses_restir[:len(xr)])

        l_std = losses_std[min(it, len(losses_std) - 1)]
        l_restir = losses_restir[min(it, len(losses_restir) - 1)]
        suptitle.set_text(
            f'Iter {it:3d}/{num_iters}   '
            f'GT Y={gt_mean_y:.1f}   '
            f'Std={l_std:.5f}   '
            f'ReSTIR={l_restir:.5f}')

        return (im_std, im_restir, line_std, line_restir, suptitle)

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=300, blit=True)
    anim.save(str(gif_path), writer='pillow', fps=4, dpi=120)
    print(f"Saved GIF → {gif_path}")

    # ---- Save static final frame ----
    # Advance to the last frame and save
    update(n_frames - 1)
    fig.savefig(str(png_path), facecolor='black', dpi=150)
    print(f"Saved final frame → {png_path}")

    plt.close()

    # ---- Summary ----
    print(f"\n=== Summary ===")
    print(f"  Standard   spp={spp}   lr={lr}:  "
          f"L1={losses_std[-1]:.5f}  Y={final_y_std:.1f}  {fps_std:.3f} it/s")
    print(f"  ReSTIR     spp={restir_spp}  cand={restir_cand}  lr={lr}:  "
          f"L1={losses_restir[-1]:.5f}  Y={final_y_restir:.1f}  {fps_restir:.3f} it/s")
    print(f"  GT Y={gt_mean_y:.1f}")
    print(f"\n  Results → {out_dir}")

    if _HAS_WANDB and wandb.run:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare Standard PathTracer vs PathReSTIR on Cornell box shape optimization')
    parser.add_argument('--iters', type=int, default=50,
                        help='Number of optimization iterations')
    parser.add_argument('--cand', type=int, default=4, choices=[4, 8],
                        help='ReSTIR n_candidates')
    parser.add_argument('--spp', type=int, default=16,
                        help='Standard PathTracer spp')
    parser.add_argument('--restir_spp', type=int, default=1,
                        help='ReSTIR spp (backward pass)')
    parser.add_argument('--name', type=str, default=None,
                        help='WandB run name')
    args = parser.parse_args()
    main(num_iters=args.iters, restir_cand=args.cand,
         spp=args.spp, restir_spp=args.restir_spp,
         run_name=args.name)
