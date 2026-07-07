"""
End-to-end Stanford-ORB evaluation: Neural-PBIR with PathReSTIR.
Single-scene run (teapot_scene001) for initial validation.

Stages:
  1. Neural Surface Reconstruction (NSR)       → geometry
  2. Neural Distillation                        → albedo / roughness / envmap init
  3. PBIR with PathReSTIR (run_restir.py)       → refined material + envmap
  4. Evaluation (Stanford-ORB metrics)

Run:
  conda activate metrology_ir
  python nbir_restir_pipeline.py

Log streaming (background):
  PYTHONUNBUFFERED=1 conda run --no-capture-output -n metrology_ir python nbir_restir_pipeline.py
"""

import json
import os
import subprocess
import time
from dotenv import load_dotenv
import wandb

load_dotenv(".env")

# ── Config ──────────────────────────────────────────────────────────────────
SCENE    = "teapot_scene001"
METHOD   = "restir_NPBIR_v1"
NPBIR    = "./DigitalTwinCatalog/neural_pbir"
RESTIR_CONFIGS = f"{NPBIR}/pbir/configs/restir_template"
CHKPT    = f"./{METHOD}/stanford_orb"
SORB     = "/home/ahc/Datasets/Stanford-ORB"
SORB_HDR = f"{SORB}/blender_HDR"
SORB_GT  = f"{SORB}/ground_truth"

# ── WandB init ───────────────────────────────────────────────────────────────
_run = wandb.init(
    project=os.environ["WANDB_PROJECT"],
    entity=os.environ["WANDB_ENTITY"],
    name=f"{METHOD}/{SCENE}",
    config={
        "method":  METHOD,
        "scene":   SCENE,
        "restir":  True,
        "restir_n_candidates":   4,
        "restir_n_neighbors":    5,
        "restir_spatial_radius": 10,
        "pbir_max_iter":         200,
        "pbir_checkpoint_iter":  25,
    },
)


def run(cmd, label):
    """Run a shell command and print a timing banner."""
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"[{label}] START")
    print(f"  {cmd}")
    print("=" * 60, flush=True)
    ret = subprocess.run(cmd, shell=True)
    dt = time.time() - t0
    print(f"[{label}] done in {dt/60:.1f} min (exit={ret.returncode})", flush=True)
    return dt


timings = {}
t0_total = time.time()

# ── Stage 1: Neural Surface Reconstruction ───────────────────────────────────
dt = run(
    f"python3 {NPBIR}/neural_surface_recon/run_template.py"
    f" --template {NPBIR}/neural_surface_recon/configs/template_stanford_orb.py"
    f" --savemem {SORB_HDR}/{SCENE}/",
    "NSR",
)
timings["nsr_min"] = dt / 60
wandb.log({"nsr_min": timings["nsr_min"]})

os.makedirs(f"{METHOD}/stanford_orb", exist_ok=True)
run(f"mv results/stanford_orb/{SCENE} {METHOD}/stanford_orb/{SCENE}", "mv NSR")

# ── Stage 2: Neural Distillation ─────────────────────────────────────────────
dt = run(f"python3 {NPBIR}/neural_distillation/run.py {CHKPT}/{SCENE}/", "Distill")
timings["distill_min"] = dt / 60
wandb.log({"distill_min": timings["distill_min"]})

# ── Stage 3: PBIR with PathReSTIR ────────────────────────────────────────────
dt = run(
    f"python3 {NPBIR}/pbir/run_restir.py {RESTIR_CONFIGS} {CHKPT}/{SCENE}/",
    "PBIR-ReSTIR",
)
timings["pbir_restir_min"] = dt / 60
timings["total_min"] = (time.time() - t0_total) / 60
wandb.log({
    "pbir_restir_min": timings["pbir_restir_min"],
    "total_min":       timings["total_min"],
})

# Log PBIR animation GIF if opt.py generated it
_gif = f"{CHKPT}/{SCENE}/pbir/microfacet_basis-envmap_ls/vis_animation.gif"
if os.path.exists(_gif):
    wandb.log({"pbir/stage2_animation": wandb.Video(_gif, fps=4, format="gif")})

# ── Render geometry ──────────────────────────────────────────────────────────
run(
    f"python3 {NPBIR}/scripts/stanford_orb/render_geo.py"
    f" {SORB_HDR}/{SCENE}/cameras.json {CHKPT}/{SCENE}/pbir/mesh.obj",
    "render_geo",
)

# ── Evaluation ───────────────────────────────────────────────────────────────
run(
    f"python {NPBIR}/scripts/stanford_orb/eval_preprocess.py"
    f" --data_dir {SORB}/ --ckpt_dir {CHKPT}/",
    "eval_preprocess",
)
run(
    f"cd Stanford-ORB && PYTHONPATH=. python scripts/test.py"
    f" --input-path ../{CHKPT}/eval_inputs_pbir.json"
    f" --output-path ../{CHKPT}/eval_outputs_pbir.json --scenes example",
    "eval",
)

_eval_path = f"{CHKPT}/eval_outputs_pbir.json"
if os.path.exists(_eval_path):
    with open(_eval_path) as _f:
        wandb.log({"eval": json.load(_f)})

# ── Timings summary ──────────────────────────────────────────────────────────
print("\n=== Timings ===")
for k, v in timings.items():
    print(f"  {k}: {v:.1f} min")

with open(f"{METHOD}/timings_restir.txt", "w") as _f:
    for k, v in timings.items():
        _f.write(f"{k}: {v:.1f}\n")

wandb.finish()
print(f"\nDone. Method={METHOD}, Scene={SCENE}")
