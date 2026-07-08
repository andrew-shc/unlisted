#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# tmux new -s jupyter
# conda activate metrology_ir
####### conda install -c conda-forge papermill
# python -m nbconvert --to script nbir_setup_refined.ipynb
# ipython nbir_setup_refined.py 2>&1




# PYTHONUNBUFFERED=1 conda run -n metrology_ir ipython nbir_setup_refined.py 2>&1

# conda run -n metrology_ir python3 nbir_setup_refined.py

# conda run -n metrology_ir python -m nbconvert --to script nbir_setup_refined.ipynb --stdout
# conda run -n metrology_ir python3


# conda run -n metrology_ir papermill nbir_setup_refined.ipynb nbir_setup_refined_out.ipynb
# jupyter nbconvert --to notebook --execute nbir_setup_refined.ipynb --output nbir_setup_refined_out.ipynb --ExecutePreprocessor.kernel_name=metrology_ir --stdout



# ALL_METHODS = [
#     "results___NPBIR___EST_g__EST_l",  # OG
#     "results___NPBIR____GT_g__EST_l",
#     "results___NPBIR___EST_g___GT_l",
#     "results___NPBIR__VGGT_g__EST_l",
#     "results___NPBIR__VGGT_g___GT_l",
#     # "results_MITSUBA__VGGT_g__EST_l",
#     # "results_MITSUBA____GT_g__EST_l",
#     # "results_MITSUBA__VGGT_g___GT_l",
# ]

SCENES_LIGHT = [
    "teapot_scene001",
    "grogu_scene002",
    "gnome_scene003",
    "car_scene004",
    "pitcher_scene005",  # <<<<<
    "blocks_scene006",
    "cactus_scene007",
]

METHOD="standard_NPBIR_reeval"
NPBIR="./DigitalTwinCatalog/neural_pbir"
CHKPT=f"./{METHOD}/stanford_orb"
SORB = "/home/ahc/Datasets/Stanford-ORB"
SORB_HDR = f"{SORB}/blender_HDR"
SORB_GT = f"{SORB}/ground_truth"
# SCENE = "teapot_scene001" # "baking_scene001"
# SCENE = SCENES_LIGHT[0]

import json, os
from dotenv import load_dotenv
import wandb
load_dotenv(".env")

# WandB writes its local run logs under `dir` — default into ASSETS/ (per
# AGENTS.md's data-separation rule) but let WANDB_DIR override it.
_wandb_run = wandb.init(
    project=os.environ["WANDB_PROJECT"],
    entity=os.environ["WANDB_ENTITY"],
    name=METHOD,
    dir=os.environ.get("WANDB_DIR", "ASSETS/wandb"),
    config={"method": METHOD, "scenes": SCENES_LIGHT},
)


# In[ ]:


import time
timings = {}

for SCENE in SCENES_LIGHT:
    # run once per new scene within the dataset itself (regardless of method, etc.)
    # !python3 DigitalTwinCatalog/neural_pbir/scripts/preprocess/stanford_orb.py /home/ahc/Datasets/Stanford-ORB/blender_HDR/{SCENE}
    # run once per Stanford-ORB (i.e., once overall for each time you download Stanford-ORB separately)
    # !python3 {NPBIR}/scripts/stanford_orb/postproc_envmap_gt.py --gt_dir {SORB_GT} --im_dir {SORB_HDR}

    t0 = time.time()
    get_ipython().system('python3 {NPBIR}/neural_surface_recon/run_template.py --template {NPBIR}/neural_surface_recon/configs/template_stanford_orb.py --savemem {SORB_HDR}/{SCENE}/   # --scale_iter 0.01')
    t_nsr = time.time()
    wandb.log({"scene": SCENE, "nsr_min": (t_nsr - t0) / 60})

    get_ipython().system('mkdir -p {METHOD}/stanford_orb && mv results/stanford_orb/{SCENE} {METHOD}/stanford_orb/{SCENE}')

    get_ipython().system('python3 {NPBIR}/neural_distillation/run.py {CHKPT}/{SCENE}/')
    t_distill = time.time()
    wandb.log({"scene": SCENE, "distill_min": (t_distill - t_nsr) / 60})

    get_ipython().system('python3 {NPBIR}/pbir/run.py {NPBIR}/pbir/configs/template {CHKPT}/{SCENE}/')
    t_pbir = time.time()
    wandb.log({"scene": SCENE, "pbir_min": (t_pbir - t_distill) / 60, "total_min": (t_pbir - t0) / 60})

    get_ipython().system('python3 {NPBIR}/scripts/stanford_orb/render_geo.py {SORB_HDR}/{SCENE}/cameras.json {CHKPT}/{SCENE}/pbir/mesh.obj')

    # !python3 {NPBIR}/pbir/run_shape.py {NPBIR}/pbir/configs/template {CHKPT}/{SCENE}/

    # !python3 {NPBIR}/pbir/run_better.py {NPBIR}/pbir/configs/template {CHKPT}/{SCENE}/


    timings[SCENE] = {
        "nsr_min":     (t_nsr - t0) / 60,
        "distill_min": (t_distill - t_nsr) / 60,
        "pbir_min":    (t_pbir - t_distill) / 60,
        "total_min":   (t_pbir - t0) / 60,
    }


# In[4]:


for SCENE in SCENES_LIGHT:
    get_ipython().system('python {NPBIR}/scripts/stanford_orb/postproc_envmap_our.py {CHKPT}/{SCENE}/pbir/envmap.exr')

    # NOVEL VIEW SYNTHESIS
    CAM=f"{SORB_HDR}/{SCENE}/cameras.json"
    CKPT_GEO=f"{CHKPT}/{SCENE}/pbir/mesh.obj"
    CKPT_ALBEDO=f"{CHKPT}/{SCENE}/pbir/diffuse.exr"
    CKPT_ROUGH=f"{CHKPT}/{SCENE}/pbir/roughness.exr"
    CKPT_ENV=f"{CHKPT}/{SCENE}/pbir/envmap_for_blender.exr"
    get_ipython().system('python {NPBIR}/scripts/relit/relit.py {CKPT_GEO} {CKPT_ALBEDO} {CKPT_ROUGH} {CAM} --lgt_paths {CKPT_ENV} --render_exr --with_bg')

# RELIGHTING
get_ipython().system('python {NPBIR}/scripts/stanford_orb/relit_nl.py --ckptroot {CHKPT} --dataroot {SORB}')


# In[3]:


# !python {NPBIR}/scripts/stanford_orb/eval_preprocess.py --data_dir {SORB}/  --ckpt_dir {CHKPT}/
get_ipython().system('cd Stanford-ORB && PYTHONPATH=. python scripts/test.py --input-path ../{CHKPT}/eval_inputs_pbir.json --output-path ../{CHKPT}/eval_outputs_pbir.json --scenes light')


# In[ ]:


with open(f"{METHOD}/timings.txt", "w") as f:
    for scene, t in timings.items():
        f.write(f"{scene}\n")
        for k, v in t.items():
            f.write(f"  {k}: {v:.1f}\n")
        f.write("\n")
print(open(f"{METHOD}/timings.txt").read())

wandb.log({"timings": timings})

_eval_path = f"{CHKPT}/eval_outputs_pbir.json"
if os.path.exists(_eval_path):
    with open(_eval_path) as _f:
        wandb.log({"eval": json.load(_f)})

wandb.finish()


# In[3]:


import re
with open(f"{METHOD}/timings.txt") as f:
    totals = [float(m) for m in re.findall(r"total_min:\s*([\d.]+)", f.read())]
print(f"Average total_min: {sum(totals)/len(totals):.1f} min")

