#!/usr/bin/env python3
"""
End-to-end Stanford-ORB evaluation: Neural-PBIR with PathReSTIR.

Stages:
  1. Neural Surface Reconstruction (NSR)     → geometry
  2. Neural Distillation                     → albedo / roughness / envmap init
  3. PBIR with PathReSTIR (run_restir.py)    → refined material + envmap
  4. Render geometry + evaluation (Stanford-ORB metrics)

Usage (from repo root):
  conda activate metrology_ir
  PYTHONUNBUFFERED=1 python GREENFIELD/restir/pipeline_restir.py \\
      --scene teapot_scene001 [--skip-setup] [--skip-pbir] [--skip-eval]

Data outputs → ASSETS/pipeline/restir_NPBIR_v1/
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

# Default Stanford-ORB location — override via STANFORD_ORB_PATH env var
SORB_DEFAULT = REPO_ROOT.parent / 'Datasets' / 'Stanford-ORB'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def resolve_sorb():
    """Resolve Stanford-ORB dataset path."""
    env_path = os.environ.get("STANFORD_ORB_PATH")
    if env_path:
        return Path(env_path)
    return SORB_DEFAULT

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def pipeline(scene, skip_setup=False, skip_pbir=False, skip_eval=False):
    sorb = resolve_sorb()
    sorb_hdr = sorb / 'blender_HDR'
    sorb_gt  = sorb / 'ground_truth'

    method = "restir_NPBIR_v1"
    assets_root = REPO_ROOT / 'ASSETS' / 'pipeline'
    method_root = assets_root / method
    chkpt_root  = method_root / 'stanford_orb'
    restir_configs = REPO_ROOT / 'DigitalTwinCatalog' / 'neural_pbir' / 'pbir' / 'configs' / 'restir_template'

    # Ensure output directories exist
    method_root.mkdir(parents=True, exist_ok=True)
    chkpt_root.mkdir(parents=True, exist_ok=True)

    timings = {}
    t0_total = time.time()

    # ── Stage 1: Neural Surface Reconstruction ──────────────────────────
    if not skip_setup:
        dt = run(
            f"python3 {NPBIR}/neural_surface_recon/run_template.py"
            f" --template {NPBIR}/neural_surface_recon/configs/template_stanford_orb.py"
            f" --savemem {sorb_hdr}/{scene}/",
            "NSR",
        )
        timings["nsr_min"] = dt / 60
        wandb.log({"nsr_min": timings["nsr_min"], "scene": scene})

        # Move results to method output folder
        run(f"mv results/stanford_orb/{scene} {chkpt_root}/{scene}", "mv NSR results")

        # ── Stage 2: Neural Distillation ────────────────────────────────
        dt = run(
            f"python3 {NPBIR}/neural_distillation/run.py {chkpt_root}/{scene}/",
            "Distill",
        )
        timings["distill_min"] = dt / 60
        wandb.log({"distill_min": timings["distill_min"], "scene": scene})
    else:
        print("[SKIP] Setup stages (NSR + Distillation)")

    # ── Stage 3: PBIR with PathReSTIR ──────────────────────────────────
    if not skip_pbir:
        dt = run(
            f"python3 GREENFIELD/restir/run_restir.py "
            f"{restir_configs} {chkpt_root}/{scene}/",
            "PBIR-ReSTIR",
        )
        timings["pbir_restir_min"] = dt / 60
        timings["total_min"] = (time.time() - t0_total) / 60
        wandb.log({
            "pbir_restir_min": timings["pbir_restir_min"],
            "total_min":       timings["total_min"],
            "scene": scene,
        })

        # Log PBIR animation GIF if opt.py generated it
        gif_path = chkpt_root / scene / "pbir" / "microfacet_basis-envmap_ls" / "vis_animation.gif"
        if gif_path.exists():
            wandb.log({"pbir/stage2_animation": wandb.Video(str(gif_path), fps=4, format="gif")})
    else:
        print("[SKIP] PBIR stage")

    # ── Stage 4: Evaluation ────────────────────────────────────────────
    if not skip_eval:
        # Render geometry
        run(
            f"python3 {NPBIR}/scripts/stanford_orb/render_geo.py"
            f" {sorb_hdr}/{scene}/cameras.json {chkpt_root}/{scene}/pbir/mesh.obj",
            "render_geo",
        )

        # Evaluation preprocessing
        run(
            f"python {NPBIR}/scripts/stanford_orb/eval_preprocess.py"
            f" --data_dir {sorb}/ --ckpt_dir {chkpt_root}/",
            "eval_preprocess",
        )

        # Run Stanford-ORB test script for light scenes
        sorb_repo = REPO_ROOT / 'Stanford-ORB'
        eval_input = chkpt_root / 'eval_inputs_pbir.json'
        eval_output = chkpt_root / 'eval_outputs_pbir.json'
        run(
            f"PYTHONPATH=. python scripts/test.py"
            f" --input-path {eval_input}"
            f" --output-path {eval_output} --scenes example",
            "eval",
            cwd=str(sorb_repo),
        )

        # Log evaluation metrics
        if eval_output.exists():
            with open(eval_output) as f:
                eval_data = json.load(f)
            wandb.log({"eval": eval_data, "scene": scene})
    else:
        print("[SKIP] Evaluation stage")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n=== Timings ===")
    for k, v in timings.items():
        print(f"  {k}: {v:.1f} min")

    timings_path = method_root / f"timings_{scene}.txt"
    with open(timings_path, "w") as f:
        for k, v in timings.items():
            f.write(f"{k}: {v:.1f}\n")
    print(f"Timings saved → {timings_path}")

    wandb.log({"timings": timings, "scene": scene})
    wandb.finish()
    print(f"\nDone. Method={method}, Scene={scene}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Full Stanford-ORB PBIR pipeline with PathReSTIR")
    parser.add_argument("--scene", type=str, default="teapot_scene001",
                        choices=SCENES_LIGHT, help="Scene name")
    parser.add_argument("--skip-setup", action="store_true",
                        help="Skip NSR and Distillation stages")
    parser.add_argument("--skip-pbir", action="store_true",
                        help="Skip PBIR stage")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation stage")
    args = parser.parse_args()

    # Load .env for WandB credentials
    load_dotenv(REPO_ROOT / ".env")

    # WandB init
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "ReSTIR Pipeline"),
        entity=os.environ.get("WANDB_ENTITY"),
        dir=os.environ.get("WANDB_DIR", str(REPO_ROOT / "ASSETS" / "wandb")),
        name=f"pipeline_restir/{args.scene}",
        config={
            "method": "restir_NPBIR_v1",
            "scene":  args.scene,
            "skip_setup": args.skip_setup,
            "skip_pbir":  args.skip_pbir,
            "skip_eval":  args.skip_eval,
        },
    )

    pipeline(
        scene=args.scene,
        skip_setup=args.skip_setup,
        skip_pbir=args.skip_pbir,
        skip_eval=args.skip_eval,
    )
