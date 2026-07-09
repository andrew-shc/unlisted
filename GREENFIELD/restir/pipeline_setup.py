#!/usr/bin/env python3
"""
Batch NSR + Neural Distillation setup for Stanford-ORB scenes.

Runs Neural Surface Reconstruction then Neural Distillation for each scene.
Skips scenes that already have results.

Usage:
  conda activate metrology_ir
  PYTHONUNBUFFERED=1 python GREENFIELD/restir/pipeline_setup.py \\
      [--scenes teapot_scene001 grogu_scene002 ...] [--skip-nsr] [--skip-distill]
  # Without --scenes, runs all 7 light scenes.

Outputs go to:
  ASSETS/pipeline/restir_NPBIR_v1/stanford_orb/<scene>/
"""

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


def resolve_sorb():
    """Resolve Stanford-ORB dataset path."""
    env_path = os.environ.get("STANFORD_ORB_PATH")
    if env_path:
        return Path(env_path)
    return SORB_DEFAULT


def run(cmd, label):
    """Run a shell command and print a timing banner."""
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"[{label}] START")
    print(f"  {cmd}")
    print('=' * 60, flush=True)
    ret = subprocess.run(cmd, shell=True, cwd=REPO_ROOT)
    dt = time.time() - t0
    status = "OK" if ret.returncode == 0 else f"FAILED (exit={ret.returncode})"
    print(f"[{label}] done in {dt/60:.1f} min — {status}", flush=True)
    if ret.returncode != 0:
        raise RuntimeError(f"Stage '{label}' failed with exit code {ret.returncode}")
    return dt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(description="Batch NSR + Distillation setup")
    parser.add_argument("--scenes", type=str, nargs="+", default=None,
                        choices=SCENES_LIGHT, help="Scene(s) to process (default: all)")
    parser.add_argument("--skip-nsr", action="store_true",
                        help="Skip Neural Surface Reconstruction stage")
    parser.add_argument("--skip-distill", action="store_true",
                        help="Skip Neural Distillation stage")
    args = parser.parse_args()

    scenes = args.scenes if args.scenes is not None else SCENES_LIGHT
    sorb = resolve_sorb()
    sorb_hdr = sorb / 'blender_HDR'

    method = "restir_NPBIR_v1"
    method_root = REPO_ROOT / 'ASSETS' / 'pipeline' / method
    chkpt_root  = method_root / 'stanford_orb'
    chkpt_root.mkdir(parents=True, exist_ok=True)

    # Load .env for WandB credentials
    load_dotenv(REPO_ROOT / ".env")

    # WandB init
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "ReSTIR Pipeline Setup"),
        entity=os.environ.get("WANDB_ENTITY"),
        dir=os.environ.get("WANDB_DIR", str(REPO_ROOT / "ASSETS" / "wandb")),
        name=f"pipeline_setup/{'-'.join(scenes)}",
        config={
            "method": method,
            "scenes": scenes,
            "skip_nsr":     args.skip_nsr,
            "skip_distill": args.skip_distill,
        },
    )

    timings = {}
    t_total = time.time()

    for scene in scenes:
        scene_ckpt = chkpt_root / scene
        print(f"\n{'#'*60}")
        print(f"# Processing scene: {scene}")
        print(f"# Checkpoint:      {scene_ckpt}")
        print(f"{'#'*60}", flush=True)

        t_scene = time.time()

        # ── Stage 1: Neural Surface Reconstruction ────────────────────
        if args.skip_nsr:
            print(f"[SKIP] NSR for {scene}")
        elif scene_ckpt.exists():
            print(f"[SKIP] NSR for {scene} — checkpoint already exists at {scene_ckpt}")
        else:
            dt = run(
                f"python3 {NPBIR}/neural_surface_recon/run_template.py"
                f" --template {NPBIR}/neural_surface_recon/configs/template_stanford_orb.py"
                f" --savemem {sorb_hdr}/{scene}/",
                f"NSR/{scene}",
            )
            timings.setdefault(scene, {})["nsr_min"] = dt / 60

            # Move results to method output folder
            run(f"mv results/stanford_orb/{scene} {scene_ckpt}", f"mv/{scene}")

            wandb.log({"scene": scene, "nsr_min": dt / 60})

        # ── Stage 2: Neural Distillation ──────────────────────────────
        if args.skip_distill:
            print(f"[SKIP] Distillation for {scene}")
        else:
            # Check for distillation outputs (e.g. if diffuse.exr exists)
            distill_ok = (scene_ckpt / "neural_distillation" / "diffuse.exr").exists()
            if distill_ok:
                print(f"[SKIP] Distillation for {scene} — outputs already exist")
            else:
                dt = run(
                    f"python3 {NPBIR}/neural_distillation/run.py {scene_ckpt}/",
                    f"Distill/{scene}",
                )
                timings.setdefault(scene, {})["distill_min"] = dt / 60
                wandb.log({"scene": scene, "distill_min": dt / 60})

        t_elapsed = (time.time() - t_scene) / 60
        timings.setdefault(scene, {})["total_min"] = t_elapsed
        print(f"[DONE] {scene} in {t_elapsed:.1f} min", flush=True)

    # ── Summary ────────────────────────────────────────────────────────
    total_time = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"All scenes done in {total_time:.1f} min")
    for scene, t in timings.items():
        nsr = t.get("nsr_min", 0)
        distill = t.get("distill_min", 0)
        tot = t.get("total_min", 0)
        print(f"  {scene:25s}  NSR={nsr:.1f}  Distill={distill:.1f}  Total={tot:.1f}")

    timings["_total_min"] = total_time
    wandb.log({"timings": timings})
    wandb.finish()


if __name__ == "__main__":
    main()
