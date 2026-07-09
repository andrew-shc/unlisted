#!/usr/bin/env python3
"""
Stanford-ORB evaluation on existing PBIR results.

Runs postprocessing, relighting, novel view synthesis, and metric computation
via the Stanford-ORB evaluation harness.

Stages:
  1. Envmap postprocessing (convert envmap to Blender format)
  2. Novel view synthesis (relighting from all test views)
  3. Relighting (environment map relighting for evaluation)
  4. Evaluation preprocessing (prepare eval inputs)
  5. Stanford-ORB metric computation (PSNR, SSIM, LPIPS)

Usage:
  conda activate metrology_ir
  PYTHONUNBUFFERED=1 python GREENFIELD/restir/pipeline_evaluate.py \\
      [--scene teapot_scene001] [--method restir_NPBIR_v1]

Outputs go under:
  ASSETS/pipeline/<method>/stanford_orb/<scene>/
"""

import json
import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from dotenv import load_dotenv
import wandb

# --- Paths ---
REPO_ROOT = Path(__file__).resolve().parents[2]
NPBIR = str(REPO_ROOT / 'DigitalTwinCatalog' / 'neural_pbir')
if NPBIR not in sys.path:
    sys.path.insert(0, NPBIR)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCENES_LIGHT = [
    "teapot_scene001",
    "grogu_scene002",
    "gnome_scene003",
    "car_scene004",
    "pitcher_scene005",
    "blocks_scene006",
    "cactus_scene007",
]

SORB_DEFAULT = REPO_ROOT.parent / 'Datasets' / 'Stanford-ORB'


def resolve_sorb():
    """Resolve Stanford-ORB dataset path."""
    env_path = os.environ.get("STANFORD_ORB_PATH")
    if env_path:
        return Path(env_path)
    return SORB_DEFAULT


def run(cmd, label, cwd=None):
    """Run a shell command with timing and print a banner."""
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"[{label}] START")
    print(f"  {cmd}")
    print('=' * 60, flush=True)
    ret = subprocess.run(cmd, shell=True, cwd=cwd or str(REPO_ROOT))
    dt = time.time() - t0
    status = "OK" if ret.returncode == 0 else f"FAILED (exit={ret.returncode})"
    print(f"[{label}] done in {dt/60:.1f} min — {status}", flush=True)
    if ret.returncode != 0:
        raise RuntimeError(f"Stage '{label}' failed with exit code {ret.returncode}")
    return dt


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def evaluate(scene, method, scenes=None):
    """Run the full evaluation pipeline for one or more scenes."""
    sorb = resolve_sorb()
    sorb_hdr = sorb / 'blender_HDR'

    method_root = REPO_ROOT / 'ASSETS' / 'pipeline' / method
    chkpt_root  = method_root / 'stanford_orb'

    # If no specific scene list provided, evaluate just the given scene
    eval_scenes = scenes if scenes else [scene]

    timings = {}
    t0_total = time.time()

    # ── Stage 1: Envmap postprocessing ────────────────────────────────
    for s in eval_scenes:
        scene_ckpt = chkpt_root / s
        envmap_path = scene_ckpt / 'pbir' / 'envmap.exr'
        if envmap_path.exists():
            run(
                f"python {NPBIR}/scripts/stanford_orb/postproc_envmap_our.py {envmap_path}",
                f"postproc_envmap/{s}",
            )
        else:
            print(f"[SKIP] postproc_envmap/{s} — envmap.exr not found at {envmap_path}")

    # ── Stage 2: Novel view synthesis ─────────────────────────────────
    for s in eval_scenes:
        scene_ckpt = chkpt_root / s
        cam   = f"{sorb_hdr}/{s}/cameras.json"
        geo   = scene_ckpt / "pbir" / "mesh.obj"
        albedo = scene_ckpt / "pbir" / "diffuse.exr"
        rough  = scene_ckpt / "pbir" / "roughness.exr"
        env    = scene_ckpt / "pbir" / "envmap_for_blender.exr"

        # Skip if any input file is missing
        if not geo.exists():
            print(f"[SKIP] novel_view/{s} — mesh.obj not found")
            continue
        if not env.exists():
            print(f"[SKIP] novel_view/{s} — envmap_for_blender.exr not found")
            continue

        run(
            f"python {NPBIR}/scripts/relit/relit.py {geo} {albedo} {rough} {cam}"
            f" --lgt_paths {env} --render_exr --with_bg",
            f"novel_view/{s}",
        )

    # ── Stage 3: Relighting ───────────────────────────────────────────
    run(
        f"python {NPBIR}/scripts/stanford_orb/relit_nl.py"
        f" --ckptroot {chkpt_root} --dataroot {sorb}",
        "relighting",
    )

    # ── Stage 4: Geometry buffer rendering (depth + normal) ───────────
    for s in eval_scenes:
        scene_ckpt = chkpt_root / s
        cam_json = f"{sorb_hdr}/{s}/cameras.json"
        mesh_obj = scene_ckpt / "pbir" / "mesh.obj"
        if not mesh_obj.exists():
            print(f"[SKIP] render_geo/{s} — mesh.obj not found")
            continue
        run(
            f"python {NPBIR}/scripts/stanford_orb/render_geo.py {cam_json} {mesh_obj}",
            f"render_geo/{s}",
        )

    # ── Stage 5: Evaluation preprocessing ─────────────────────────────
    run(
        f"python {NPBIR}/scripts/stanford_orb/eval_preprocess.py"
        f" --data_dir {sorb}/ --ckpt_dir {chkpt_root}/",
        "eval_preprocess",
    )

    # ── Filter eval_inputs to only scenes with valid outputs ─────────
    eval_input  = chkpt_root / 'eval_inputs_pbir.json'
    eval_filtered = chkpt_root / 'eval_inputs_filtered.json'
    if eval_input.exists():
        with open(eval_input) as f:
            eval_data = json.load(f)
        info = eval_data.get('info', {})
        valid_scenes = {}
        for sname, sdata in info.items():
            has_mesh = sdata.get('shape', {}).get('output_mesh') is not None
            view_ok = all(
                item.get('output_image') is not None
                for item in sdata.get('view', [])
            )
            geo_ok = all(
                item.get('output_depth') is not None and item.get('output_normal') is not None
                for item in sdata.get('geometry', [])
            )
            if has_mesh and view_ok and geo_ok:
                valid_scenes[sname] = sdata
        n_removed = len(info) - len(valid_scenes)
        print(f"[FILTER] Kept {len(valid_scenes)}/{len(info)} scenes with valid outputs "
              f"(removed {n_removed})")
        with open(eval_filtered, 'w') as f:
            json.dump({'info': valid_scenes}, f)
    else:
        eval_filtered = eval_input  # fallback if no filtering needed

    # ── Stage 6: Stanford-ORB metric computation ──────────────────────
    eval_output = chkpt_root / 'eval_outputs_filtered.json'
    sorb_repo = REPO_ROOT / 'Stanford-ORB'

    # Always auto — we filter the JSON ourselves to include only valid scenes
    scenes_arg = "auto"

    run(
        f"PYTHONPATH=. python scripts/test.py"
        f" --input-path {eval_filtered}"
        f" --output-path {eval_output} --scenes {scenes_arg}",
        "eval_metrics",
        cwd=str(sorb_repo),
    )

    # ── Log evaluation metrics ────────────────────────────────────────
    if eval_output.exists():
        with open(eval_output) as f:
            eval_data = json.load(f)
        wandb.log({"eval": eval_data, "scene": scene})
        print(f"\nEvaluation metrics: {json.dumps(eval_data, indent=2)}")
    else:
        print(f"\n[WARN] Evaluation output not found at {eval_output}")

    # ── Timing summary ────────────────────────────────────────────────
    timings["total_min"] = (time.time() - t0_total) / 60
    wandb.log({"timings": timings, "scene": scene})
    wandb.finish()

    print(f"\nDone. Method={method}, Scene(s)={eval_scenes}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Stanford-ORB evaluation on PBIR results")
    parser.add_argument("--scene", type=str, default="teapot_scene001",
                        choices=SCENES_LIGHT, help="Primary scene name")
    parser.add_argument("--method", type=str, default="restir_NPBIR_v1",
                        help="Method name (subdir under ASSETS/pipeline/)")
    parser.add_argument("--scenes", type=str, nargs="+", default=None,
                        help="Specific scenes to evaluate (default: just --scene)")
    args = parser.parse_args()

    # Load .env for WandB credentials
    load_dotenv(REPO_ROOT / ".env")

    # WandB init
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "ReSTIR Evaluation"),
        entity=os.environ.get("WANDB_ENTITY"),
        dir=os.environ.get("WANDB_DIR", str(REPO_ROOT / "ASSETS" / "wandb")),
        name=f"pipeline_evaluate/{args.scene}",
        config={
            "method": args.method,
            "scene":  args.scene,
        },
    )

    evaluate(scene=args.scene, method=args.method, scenes=args.scenes)
