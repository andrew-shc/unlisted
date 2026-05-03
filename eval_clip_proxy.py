DO_SAM3D = False  # any environment works
# conda activate relax-flow
DO_TRELLIS = True  # TRELLIS environment
# conda activate trellis2
DO_TRELLIS2 = False  # TRELLIS.2 environment

from pathlib import Path
import subprocess
import sys
if DO_TRELLIS:
    sys.path.append(str(Path("~/Documents/vggt2/digitizing_reality/TRELLIS").expanduser()))
if DO_TRELLIS2:
    sys.path.append(str(Path("~/Documents/vggt2/digitizing_reality/TRELLIS.2").expanduser()))

import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory

import imageio
from PIL import Image
import cv2

if DO_TRELLIS:
    # from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.pipelines.relaxflow_pipeline import TrellisImageTo3DPipelineRelaxFlow
    from trellis.utils import render_utils, postprocessing_utils

if DO_TRELLIS2:
    # from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.pipelines.relaxflow_pipeline import Trellis2ImageTo3DPipelineRelaxFlow
    from trellis2.utils import render_utils as render_utils_2
    from trellis2.renderers import EnvMap
    import o_voxel

from matplotlib import pyplot as plt

import gc
import torch

import eval_utils
from PIL import Image
import torchvision.transforms.functional as TF


EVAL_DATA = Path("./eval_data")
OUTPUT_PATH = Path("./eval_results")


def main():
    prior1A =  EVAL_DATA / "DUCK/pred0/A yellow rubber duck with a pirate eye patch on its left, white background, centered, 3D render.png"
    prior1B =  EVAL_DATA / "DUCK/pred1/A yellow rubber duck with left eye blue right eye white, white background, centered, 3D render.png"
    prior1C =  EVAL_DATA / "DUCK/pred2/A yellow rubber duck with the left eye replaced by a googlie eye, white background, centered, 3D render.png"
    prior2A =  EVAL_DATA / "FLIP_CALENDAR_RED/pred0/Red flip paper calendar with many binder rings showing month of March, white background, centered, 3D render.png"
    prior2B =  EVAL_DATA / "FLIP_CALENDAR_RED/pred1/Red flip paper calendar with many binder rings showing month of April (on a yellow paper on top of the red flip calendar), white background, centered, 3D render.png"
    prior2C =  EVAL_DATA / "FLIP_CALENDAR_RED/pred2/Red flip paper calendar with many binder rings showing with two large flip placards representing the month and the day, white background, centered, 3D render.png"
    prior3A =  EVAL_DATA / "SIGN_BIKE_ROUTE/pred0/Back of the roadside sign is green denoting bear in area, white background, centered, 3D render.png"
    prior3B =  EVAL_DATA / "SIGN_BIKE_ROUTE/pred1/Back of the roadside sign is orange denoting construction ahead, white background, centered, 3D render.png"
    prior3C =  EVAL_DATA / "SIGN_BIKE_ROUTE/pred2/Back of the roadside sign is nothing and all grey, white background, centered, 3D render.png"


    masked_image1 = EVAL_DATA / "DUCK/gt_obs_segmented/frame_000000.png"
    masked_image2 = EVAL_DATA / "FLIP_CALENDAR_RED/gt_obs_segmented/frame_000001.png"
    masked_image3 = EVAL_DATA / "SIGN_BIKE_ROUTE/gt_obs_segmented/frame_000001.png"

    if DO_TRELLIS:
        trellis1_pipeline = TrellisImageTo3DPipelineRelaxFlow.from_pretrained("microsoft/TRELLIS-image-large")
        trellis1_pipeline.cuda()

    if DO_TRELLIS2:
        trellis2_envmap = EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('./TRELLIS.2/assets/hdri/studio.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        ))
        trellis2_pipeline = Trellis2ImageTo3DPipelineRelaxFlow.from_pretrained("microsoft/TRELLIS.2-4B")
        trellis2_pipeline.cuda()


    # TRELLIS2_MODIFIERS_IND = 1
    TRELLIS2_MODIFIERS = [
        # default one (the first one) previously (one must be present for cases that dont use TRELLIS2)
        (1.0,1.0,50,"1.0-1.0-50"),
        # (1.0,1.5,50,"1.0-1.5-50"),
        # (1.0,0.5,50,"1.0-0.5-50"),
        # (1.0,1.0,100,"1.0-1.0-100"),
        # (1.0,1.5,100,"1.0-1.5-100"),
        # (1.0,0.5,100,"1.0-0.5-100"),
        # (0.5,1.0,50,"0.5-1.0-50"),
        # (0.5,1.5,50,"0.5-1.5-50"),
    ]
    # for masked_img, prior, path in [
    #     (masked_image1, prior1A, "DUCK/pred0"),
    #     (masked_image1, prior1B, "DUCK/pred1"),
    #     (masked_image1, prior1C, "DUCK/pred2"),
    #     # (masked_image2, prior2A, "FLIP_CALENDAR_RED/pred0"),
    #     # (masked_image2, prior2B, "FLIP_CALENDAR_RED/pred1"),
    #     # (masked_image2, prior2C, "FLIP_CALENDAR_RED/pred2"),
    #     # (masked_image3, prior3A, "SIGN_BIKE_ROUTE/pred0"),
    #     # (masked_image3, prior3B, "SIGN_BIKE_ROUTE/pred1"),
    #     # (masked_image3, prior3C, "SIGN_BIKE_ROUTE/pred2"),
    # ]:
    #     for TRELLIS2_MODIFIERS_IND in range(len(TRELLIS2_MODIFIERS)):
    
    for TRELLIS2_MODIFIERS_IND in range(len(TRELLIS2_MODIFIERS)):
        for masked_img, prior, path in [
            (masked_image1, prior1A, "DUCK/pred0"),
            (masked_image1, prior1B, "DUCK/pred1"),
            (masked_image1, prior1C, "DUCK/pred2"),
            (masked_image2, prior2A, "FLIP_CALENDAR_RED/pred0"),
            (masked_image2, prior2B, "FLIP_CALENDAR_RED/pred1"),
            (masked_image2, prior2C, "FLIP_CALENDAR_RED/pred2"),
            (masked_image3, prior3A, "SIGN_BIKE_ROUTE/pred0"),
            (masked_image3, prior3B, "SIGN_BIKE_ROUTE/pred1"),
            (masked_image3, prior3C, "SIGN_BIKE_ROUTE/pred2"),
        ]:
            try:
                OUT_PATH_SAM3D = OUTPUT_PATH / f"{path}/SAM3D"
                OUT_PATH_SAM3D.mkdir(parents=True, exist_ok=True)
                OUT_PATH_TRELLIS = OUTPUT_PATH / f"{path}/TRELLIS"
                OUT_PATH_TRELLIS.mkdir(parents=True, exist_ok=True)

                OUT_PATH_TRELLIS2 = OUTPUT_PATH / f"{path}/TRELLIS.2{TRELLIS2_MODIFIERS[TRELLIS2_MODIFIERS_IND][3]}"
                OUT_PATH_TRELLIS2.mkdir(parents=True, exist_ok=True)

                #### SAM3D ####
                if DO_SAM3D:
                    relaxflow_dir = Path("/home/ahc/Documents/vggt2/RelaxFlow")

                    cmd = [
                        "python", "demo_relaxflow.py",
                        "--image", masked_img.resolve(),
                        # "--mask", str(folder / "mask.png"),
                        "--prior-images", prior.resolve(),
                        "--output-name", OUT_PATH_SAM3D.resolve(),
                        "--render-backend", "gsplat",
                    ]

                    subprocess.run(cmd, cwd=relaxflow_dir, check=True)


                #### TRELLIS 1 ####
                img = Image.open(masked_img)
                prior_img = Image.open(prior)

                if DO_TRELLIS:
                    output = trellis1_pipeline.run_relaxflow(
                        img,
                        prior_images=[prior_img],
                        prior_blur_sigma=1.0,
                        seed=1,
                        return_branch_outputs=True,
                    )

                    frames = render_utils.render_video(output["relaxflow"]['gaussian'][0], num_frames=10)['color']

                    for i, frame in enumerate(frames):
                        imageio.imwrite(OUT_PATH_TRELLIS / f"view_{i:02d}.png", frame)

                    print(f"Saved {len(frames)} views to {OUT_PATH_TRELLIS}")

                    video = render_utils.render_video(output["relaxflow"]['gaussian'][0])['color']
                    imageio.mimsave(OUT_PATH_TRELLIS / "TRELLIS_RelaxFlow_Combined_gs.gif", video, fps=30, loop=0)

                    video = render_utils.render_video(output["obs_only"]['gaussian'][0])['color']
                    imageio.mimsave(OUT_PATH_TRELLIS / "TRELLIS_RelaxFlow_Observation_gs.gif", video, fps=30, loop=0)

                    video = render_utils.render_video(output["prior_only"]['gaussian'][0])['color']
                    imageio.mimsave(OUT_PATH_TRELLIS / "TRELLIS_RelaxFlow_Prior_gs.gif", video, fps=30, loop=0)


                #### TRELLIS 2 ####
                if DO_TRELLIS2:
                    output = trellis2_pipeline.run_relaxflow(
                        img,
                        prior_images=[prior_img],
                        prior_weight=TRELLIS2_MODIFIERS[TRELLIS2_MODIFIERS_IND][0],
                        prior_blur_sigma=TRELLIS2_MODIFIERS[TRELLIS2_MODIFIERS_IND][1],
                        stage1_inference_steps=TRELLIS2_MODIFIERS[TRELLIS2_MODIFIERS_IND][2],
                    )

                    mesh = output["relaxflow"]["mesh"][0]
                    mesh.simplify(16777216)

                    frames = render_utils_2.render_video(mesh, envmap=trellis2_envmap, num_frames=10)

                    for i, frame in enumerate(frames["shaded"]):
                        imageio.imwrite(OUT_PATH_TRELLIS2 / f"view_{i:02d}.png", frame)

                    print(f"Saved {len(frames['shaded'])} views to {OUT_PATH_TRELLIS2}")

                    mesh_relaxflow = output["relaxflow"]["mesh"][0]
                    mesh_obs = output["obs_only"]["mesh"][0]
                    mesh_prior = output["prior_only"]["mesh"][0]
                    # nvdiffrast limit
                    mesh_relaxflow.simplify(16777216)
                    mesh_obs.simplify(16777216)
                    mesh_prior.simplify(16777216)

                    # 4. Render Video
                    video = render_utils_2.make_pbr_vis_frames(render_utils_2.render_video(mesh_relaxflow, envmap=trellis2_envmap, num_frames=60))
                    imageio.mimsave(OUT_PATH_TRELLIS2 / "TRELLIS2_RelaxFlow_Combined.gif", video, fps=15, loop=0)
                    video = render_utils_2.make_pbr_vis_frames(render_utils_2.render_video(mesh_obs, envmap=trellis2_envmap, num_frames=60))
                    imageio.mimsave(OUT_PATH_TRELLIS2 / "TRELLIS2_RelaxFlow_Observation.gif", video, fps=15, loop=0)
                    video = render_utils_2.make_pbr_vis_frames(render_utils_2.render_video(mesh_prior, envmap=trellis2_envmap, num_frames=60))
                    imageio.mimsave(OUT_PATH_TRELLIS2 / "TRELLIS2_RelaxFlow_Prior.gif", video, fps=15, loop=0)

            except torch.OutOfMemoryError:
                print(f"CUDA out of memory on {path}, skipping.")
                gc.collect()
                torch.cuda.empty_cache()



def compute_proxy_metrics():
    # CLIP PROXY METRICS

    MODELS = ["SAM3D", "TRELLIS", "TRELLIS.2",
              
              
        "TRELLIS.20.5-1.0-50",
        "TRELLIS.20.5-1.5-50",
        "TRELLIS.21.0-0.5-50",
        "TRELLIS.21.0-0.5-100",
        "TRELLIS.21.0-1.0-50",
        "TRELLIS.21.0-1.0-100",
        "TRELLIS.21.0-1.5-50",
        "TRELLIS.21.0-1.5-100",]

    scores = {m: {"clip_text": [], "clip_img": []} for m in MODELS}

    for scene_dir in sorted(EVAL_DATA.iterdir()):
        if not scene_dir.is_dir():
            continue
        gt_obs_seg_dir = scene_dir / "gt_obs_segmented"
        if not gt_obs_seg_dir.exists():
            continue
        gt_images = eval_utils.load_images_from_dir(gt_obs_seg_dir)
        if not gt_images:
            continue
        observed_render = gt_images[0]

        output_scene_dir = OUTPUT_PATH / scene_dir.name
        if not output_scene_dir.exists():
            continue
        pred_dirs = sorted([d for d in output_scene_dir.iterdir() if d.is_dir() and d.name.startswith("pred")])
        gt_dirs = sorted([d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith("pred")])


        for pred_dir, gt_dir in zip(pred_dirs, gt_dirs):
            scene_text = eval_utils.get_pred_text(gt_dir)
            if scene_text is None:
                print(f"  skip (no prompt file)  {scene_dir.name}/{gt_dir.name}")
                continue

            print(">>>>>", scene_text)

            for model in MODELS:
                pred_imgs = eval_utils.get_pred_images(pred_dir, model)
                if pred_imgs is None:
                    print(f"  skip  {scene_dir.name}/{pred_dir.name}/{model}")
                    continue

                ct = eval_utils.clip_text_similarity(pred_imgs, scene_text)
                ci = eval_utils.clip_img_similarity(gt_images, observed_render, pred_imgs)
                scores[model]["clip_text"].append(ct)
                scores[model]["clip_img"].append(ci)
                print(f"  {scene_dir.name}/{pred_dir.name}/{model}  text={ct:.4f}  img={ci:.4f}")

    print("\n=== Average scores by model ===")
    for model in MODELS:
        ct_vals = scores[model]["clip_text"]
        ci_vals = scores[model]["clip_img"]
        if ct_vals:
            print(f"{model:12s}  clip_text={sum(ct_vals)/len(ct_vals):.4f} (n={len(ct_vals)})  "
                f"clip_img={sum(ci_vals)/len(ci_vals):.4f} (n={len(ci_vals)})")
        else:
            print(f"{model:12s}  no data")

if __name__ == "__main__":
    # main()
    compute_proxy_metrics()
