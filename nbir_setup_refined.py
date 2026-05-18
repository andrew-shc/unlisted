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
    "pitcher_scene005",
    "blocks_scene006",
    "cactus_scene007",
]

METHOD="standard_NPBIR"
NPBIR="./DigitalTwinCatalog/neural_pbir"
CHKPT=f"./{METHOD}/stanford_orb"
SORB = "/home/ahc/Datasets/Stanford-ORB"
SORB_HDR = f"{SORB}/blender_HDR"
SORB_GT = f"{SORB}/ground_truth"
# SCENE = "teapot_scene001" # "baking_scene001"
# SCENE = SCENES_LIGHT[0]


# In[ ]:


for SCENE in SCENES_LIGHT:
    # # run once per new scene within the dataset itself (regardless of method, etc.)
    # !python3 DigitalTwinCatalog/neural_pbir/scripts/preprocess/stanford_orb.py /home/ahc/Datasets/Stanford-ORB/blender_HDR/{SCENE}
    # # run once per Stanford-ORB (i.e., once overall for each time you download Stanford-ORB separately)
    # # !python3 {NPBIR}/scripts/stanford_orb/postproc_envmap_gt.py --gt_dir {SORB_GT} --im_dir {SORB_HDR}

    # !python3 {NPBIR}/neural_surface_recon/run_template.py --template {NPBIR}/neural_surface_recon/configs/template_stanford_orb.py --savemem {SORB_HDR}/{SCENE}/   # --scale_iter 0.01

    # !mkdir -p {METHOD}/stanford_orb && mv results/stanford_orb/{SCENE} {METHOD}/stanford_orb/{SCENE}

    # !python3 {NPBIR}/neural_distillation/run.py {CHKPT}/{SCENE}/
    # # !python3 {NPBIR}/pbir/run.py {NPBIR}/pbir/configs/template {CHKPT}/{SCENE}/

    # !python3 {NPBIR}/scripts/stanford_orb/render_geo.py {SORB_HDR}/{SCENE}/cameras.json {CHKPT}/{SCENE}/pbir/mesh.obj

    # !python3 {NPBIR}/pbir/run_shape.py {NPBIR}/pbir/configs/template {CHKPT}/{SCENE}/

    get_ipython().system('python3 {NPBIR}/pbir/run_better.py {NPBIR}/pbir/configs/template {CHKPT}/{SCENE}/')

# !python {NPBIR}/scripts/stanford_orb/postproc_envmap_our.py "{CHKPT}/{SCENE}/pbir/envmap.exr"

# # NOVEL VIEW SYNTHESIS
# CAM=f"{SORB_HDR}/{SCENE}/cameras.json"
# CKPT_GEO=f"{CHKPT}/{SCENE}/pbir/mesh.obj"
# CKPT_ALBEDO=f"{CHKPT}/{SCENE}/pbir/diffuse.exr"
# CKPT_ROUGH=f"{CHKPT}/{SCENE}/pbir/roughness.exr"
# CKPT_ENV=f"{CHKPT}/{SCENE}/pbir/envmap_for_blender.exr"
# !python {NPBIR}/scripts/relit/relit.py {CKPT_GEO} {CKPT_ALBEDO} {CKPT_ROUGH} {CAM} --lgt_paths {CKPT_ENV} --render_exr --with_bg

# # RELIGHTING
# !python {NPBIR}/scripts/stanford_orb/relit_nl.py --ckptroot {CHKPT} --dataroot {SORB}

# !python {NPBIR}/scripts/stanford_orb/eval_preprocess.py --data_dir {SORB}/  --ckpt_dir {CHKPT}/
# !cd Stanford-ORB && PYTHONPATH=. python scripts/test.py --input-path ../{CHKPT}/eval_inputs_pbir.json --output-path ../{CHKPT}/eval_outputs_pbir.json --scenes example


# In[ ]:




