#!/usr/bin/env python3
"""
Standalone animation generator — runs the IRTK shape optimization on the Cornell box
and saves a side-by-side GIF: target | optimizing render.
Mirrors neural_pbir/pbir/run_shape_IT3.py but without the full notebook overhead.

Usage:
  conda activate metrology_ir
  PYTHONUNBUFFERED=1 python GREENFIELD/restir/generate_animation.py

Output:
  ASSETS/restir/animation/shape_opt_animation.gif
  ASSETS/restir/animation/shape_target_anim.exr
"""

import os
import sys
from pathlib import Path

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import torch
import numpy as np
import gin

gin.enter_interactive_mode()

# --- Paths ---
REPO_ROOT = Path(__file__).resolve().parents[2]
NPBIR_PBIR = str(REPO_ROOT / 'DigitalTwinCatalog' / 'neural_pbir' / 'pbir')
if NPBIR_PBIR not in sys.path:
    sys.path.insert(0, NPBIR_PBIR)

CBOX = str(REPO_ROOT / 'psdr-jit' / 'tutorials' / 'data' / 'cbox')

# WandB (optional)
try:
    from dotenv import load_dotenv
    import wandb
    load_dotenv(REPO_ROOT / ".env")
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

from irtk.scene import Scene, Mesh, DiffuseBRDF, HDRFilm, Integrator, PerspectiveCamera
from irtk.renderer import Renderer
from irtk.io import write_image, to_torch_f
from irtk.loss import l1_loss
from models.shape_ls import ShapeLS


def load_obj_tri(path):
    """Load a Wavefront OBJ file into numpy arrays (vertices, faces, UVs, face UVs)."""
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
    v   = np.array(verts,    dtype=np.float32)
    f   = np.array(faces,    dtype=np.int32)
    uv  = np.array(uvs,      dtype=np.float32) if uvs else np.zeros((len(verts), 2), np.float32)
    fuv = np.array(faces_uv, dtype=np.int32)   if uvs else f.copy()
    return v, f, uv, fuv


def mesh_from_obj(path, **kwargs):
    """Load an OBJ file into an IRTK Mesh, with can_change_topology=True default."""
    v, f, uv, fuv = load_obj_tri(path)
    # can_change_topology=True is the only connector path that applies use_face_normal.
    # The False path (write temp .obj → add_Mesh) silently ignores it.
    kwargs.setdefault('can_change_topology', True)
    return Mesh(v, f, uv, fuv, **kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = REPO_ROOT / 'ASSETS' / 'restir' / 'animation'
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path  = out_dir / 'shape_opt_animation.gif'
    exr_path  = out_dir / 'shape_target_anim.exr'

    # WandB init (optional)
    if _HAS_WANDB:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "ReSTIR Animation"),
            entity=os.environ.get("WANDB_ENTITY"),
            dir=os.environ.get("WANDB_DIR", str(REPO_ROOT / "ASSETS" / "wandb")),
            name=f"cornell_animation/{Path(__file__).stem}",
            config={
                "method": "shape_opt_animation",
                "num_iters": 60,
                "save_every": 2,
                "lr": 5.0,
            },
        )

    # ── Build scene ──────────────────────────────────────────────────────
    print("Building IRTK scene…")
    scene = Scene()
    scene.set('film',       HDRFilm(width=512, height=512))
    scene.set('integrator', Integrator('path', {'max_depth': 3, 'hide_emitters': False}))
    scene.set('sensor 0',   PerspectiveCamera(
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
        radiance=[20., 20., 8.], to_world=lum_xfm,
        use_face_normal=True))
    scene.set('mesh',     mesh_from_obj(f'{CBOX}/cbox_smallbox.obj', mat_id='mat_cat',   use_face_normal=True))
    scene.set('largebox', mesh_from_obj(f'{CBOX}/cbox_largebox.obj', mat_id='mat_cat',   use_face_normal=True))
    scene.set('floor',    mesh_from_obj(f'{CBOX}/cbox_floor.obj',    mat_id='mat_white', use_face_normal=True))
    scene.set('ceiling',  mesh_from_obj(f'{CBOX}/cbox_ceiling.obj',  mat_id='mat_white', use_face_normal=True))
    scene.set('back',     mesh_from_obj(f'{CBOX}/cbox_back.obj',     mat_id='mat_white', use_face_normal=True))
    scene.set('green',    mesh_from_obj(f'{CBOX}/cbox_greenwall.obj', mat_id='mat_green', use_face_normal=True))
    scene.set('red',      mesh_from_obj(f'{CBOX}/cbox_redwall.obj',   mat_id='mat_red',   use_face_normal=True))

    render_opt = Renderer('psdr_jit', render_options={
        'spp': 16, 'sppe': 16, 'sppse': 8, 'npass': 1, 'log_level': 0,
    })
    render_vis = Renderer('psdr_jit', render_options={
        'spp': 64, 'sppe': 0, 'sppse': 0, 'npass': 1, 'log_level': 0,
    })

    # ── Target render ────────────────────────────────────────────────────
    print("Rendering target…")
    target = render_vis(scene, sensor_ids=[0])[0]  # (512, 512, 3) on CUDA
    write_image(str(exr_path), target)
    print(f"Target saved → {exr_path}")

    V_gt = scene['mesh.v'].detach().clone()
    print(f"Small box: {V_gt.shape[0]} verts  Y=[{V_gt[:, 1].min():.0f}, {V_gt[:, 1].max():.0f}]")

    # ── Perturb ──────────────────────────────────────────────────────────
    scene['mesh']['v'] = V_gt + to_torch_f([[0., 70., 0.]])

    # ── Optimization + frame capture ─────────────────────────────────────
    model = ShapeLS(scene, mesh_id='mesh', optimizer_kwargs={'lr': 5.0, 'lmbda': 1})

    num_iters  = 60
    save_every = 2
    frames     = []
    mean_ys    = []

    print("Running optimization…")
    for it in range(num_iters):
        model.zero_grad()
        model.set_data()
        scene.configure()

        opt_image  = render_opt(scene, sensor_ids=[0], integrator_id=0)[0]
        image_loss = l1_loss(target, opt_image)
        image_loss.backward()
        model.step()

        if it % save_every == 0:
            with torch.no_grad():
                frame = render_vis(scene, sensor_ids=[0])[0]
            frames.append(frame.cpu().numpy().clip(0, 1))
            mean_ys.append(scene['mesh.v'].detach()[:, 1].mean().item())
            print(f"  iter {it:3d}  L1={image_loss.item():.5f}  "
                  f"mean_Y={mean_ys[-1]:.1f}  (GT={V_gt[:, 1].mean():.1f})", flush=True)

            # WandB per-iteration logging
            if _HAS_WANDB and wandb.run:
                wandb.log({
                    "animation/loss":      image_loss.item(),
                    "animation/y_position": mean_ys[-1],
                    "animation/iter":      it,
                })

    print(f"Collected {len(frames)} frames.")

    # If WandB is active, log the final GIF
    if _HAS_WANDB and wandb.run:
        # Save intermediate GIF for WandB upload
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation as mpl_anim

        tgt_np = target.cpu().numpy().clip(0, 1)
        gt_y   = V_gt[:, 1].mean().item()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.patch.set_facecolor('black')
        for ax in axes:
            ax.axis('off')
            ax.set_facecolor('black')

        axes[0].imshow(tgt_np, vmin=0, vmax=1)
        axes[0].set_title('Target', color='white', fontsize=13)

        im_cur = axes[1].imshow(frames[0], vmin=0, vmax=1)
        ttl    = axes[1].set_title(f'Iter 0  mean_Y={mean_ys[0]:.1f}  GT={gt_y:.1f}',
                                    color='white', fontsize=11)

        def update(i):
            im_cur.set_data(frames[i])
            ttl.set_text(f'Iter {i * save_every:3d}  mean_Y={mean_ys[i]:.1f}  GT={gt_y:.1f}')
            return im_cur, ttl

        anim = mpl_anim.FuncAnimation(fig, update, frames=len(frames), interval=150, blit=True)
        plt.tight_layout()
        anim.save(str(gif_path), writer='pillow', fps=6, dpi=100)
        plt.close()
        print(f"Saved → {gif_path}")

        # Log to WandB
        wandb.log({"animation/gif": wandb.Video(str(gif_path), fps=6, format="gif")})
        wandb.finish()
    else:
        # ── Build + save animation ───────────────────────────────────────
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        tgt_np = target.cpu().numpy().clip(0, 1)
        gt_y   = V_gt[:, 1].mean().item()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.patch.set_facecolor('black')
        for ax in axes:
            ax.axis('off')
            ax.set_facecolor('black')

        axes[0].imshow(tgt_np, vmin=0, vmax=1)
        axes[0].set_title('Target', color='white', fontsize=13)

        im_cur = axes[1].imshow(frames[0], vmin=0, vmax=1)
        ttl    = axes[1].set_title(f'Iter 0  mean_Y={mean_ys[0]:.1f}  GT={gt_y:.1f}',
                                    color='white', fontsize=11)

        def update(i):
            im_cur.set_data(frames[i])
            ttl.set_text(f'Iter {i * save_every:3d}  mean_Y={mean_ys[i]:.1f}  GT={gt_y:.1f}')
            return im_cur, ttl

        anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=150, blit=True)
        plt.tight_layout()
        anim.save(str(gif_path), writer='pillow', fps=6, dpi=100)
        plt.close()
        print(f"Saved → {gif_path}")


if __name__ == '__main__':
    main()
