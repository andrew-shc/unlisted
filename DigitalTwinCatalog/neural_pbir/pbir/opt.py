# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from time import time

import gin
import numpy as np
import torch
from irtk.io import write_image, write_mesh
from irtk.loss import l1_loss
from tqdm import tqdm

try:
    import wandb as _wb
except ImportError:
    _wb = None

try:
    from PIL import Image as _PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


@gin.configurable
def optimize(
    scene,
    dataset,
    result_path,
    model_class,
    num_epochs,
    max_iter,
    checkpoint_iter,
    render_opt,
    render_vis,
):
    # Make a folder at the result_path
    result_path = Path(result_path)
    result_path.mkdir(parents=True, exist_ok=True)

    # Handle sensors
    num_sensors = len(dataset)
    opt_sensor_ids = torch.arange(num_sensors, dtype=torch.int64)
    vis_sensor_ids = torch.randperm(num_sensors)[0:4]

    # Create the model, which solve one or multiple inverse problems
    model = model_class(scene)

    # Optimization related
    max_iter = min(max_iter, len(opt_sensor_ids) * num_epochs)
    loss_record = []
    _vis_frames = []
    iter = 0

    # End configuration and begin optimization
    print("Starting the optimization...")
    pbar = tqdm(total=max_iter)

    for _ in range(1, num_epochs + 1):
        # Randomly shuffle sensors
        sensor_perm = opt_sensor_ids[torch.randperm(num_sensors)]

        for sensor_id in sensor_perm:
            tar_image = dataset[sensor_id]

            model.zero_grad()
            model.set_data()
            scene.configure()

            t0 = time()

            opt_image = render_opt(scene, sensor_ids=[sensor_id], integrator_id=0)

            t1 = time()
            render_time = t1 - t0

            t0 = time()

            image_loss = l1_loss(tar_image, opt_image)
            reg_loss = model.get_regularization()
            loss = image_loss + reg_loss

            loss.backward()
            model.step()

            t1 = time()
            opt_time = t1 - t0

            loss_record.append(loss.item())

            if _wb is not None and _wb.run is not None:
                _wb.log({
                    "pbir/loss":       loss.item(),
                    "pbir/image_loss": image_loss.item(),
                    "pbir/render_ms":  render_time * 1000,
                    "pbir/opt_ms":     opt_time * 1000,
                }, step=iter)

            iter += 1

            pbar.update(1)

            if iter == 1 or iter % checkpoint_iter == 0:
                iter_path = result_path / str(iter)
                iter_path.mkdir(parents=True, exist_ok=True)

                model.write_results(iter_path)

                tar_images_cat = torch.cat(
                    [dataset[id] for id in vis_sensor_ids], dim=1
                )

                with torch.no_grad():
                    vis_images = render_vis(scene, sensor_ids=vis_sensor_ids)
                    vis_images_cat = torch.cat(
                        [vis_image for vis_image in vis_images], dim=1
                    )

                final_image = torch.cat([tar_images_cat, vis_images_cat], dim=0)
                write_image(iter_path / "vis.exr", final_image)

                torch.save(loss_record, result_path / "loss.pt")

                _vis_np = np.clip(
                    vis_images_cat.detach().cpu().float().numpy() ** (1 / 2.2), 0, 1
                )
                _vis_frames.append(_vis_np[::2, ::2])
                if _wb is not None and _wb.run is not None:
                    _tar_np = np.clip(
                        tar_images_cat.detach().cpu().float().numpy() ** (1 / 2.2), 0, 1
                    )
                    _wb.log(
                        {"pbir/vis": _wb.Image(np.concatenate([_tar_np, _vis_np], axis=0))},
                        step=iter,
                    )

            if iter == max_iter:
                pbar.close()

                print("Writing final outputs...")
                final_path = result_path / "final"
                final_path.mkdir(parents=True, exist_ok=True)

                d = scene["mat.d"]
                s = scene["mat.s"]
                r = scene["mat.r"]
                write_image(final_path / "diffuse.exr", d)
                write_image(final_path / "diffuse.png", d)

                write_image(final_path / "specular.exr", s)
                write_image(final_path / "specular.png", s)

                write_image(final_path / "roughness.exr", r)
                write_image(final_path / "roughness.png", r, is_srgb=False)

                envmap = scene["envmap.radiance"]
                write_image(final_path / "envmap.exr", envmap)

                v = scene["mesh.v"]
                f = scene["mesh.f"]
                uv = scene["mesh.uv"]
                fuv = scene["mesh.fuv"]
                write_mesh(final_path / "mesh.obj", v, f, uv, fuv)

                if _wb is not None and _wb.run is not None:
                    _wb.log({
                        "pbir/final_diffuse":   _wb.Image(
                            np.clip(d.detach().cpu().numpy() ** (1 / 2.2), 0, 1)),
                        "pbir/final_roughness": _wb.Image(
                            np.clip(r.detach().cpu().numpy(), 0, 1)),
                        "pbir/final_envmap":    _wb.Image(
                            np.clip(envmap.detach().cpu().numpy() /
                                    max(envmap.max().item(), 1e-6), 0, 1)),
                    })
                    if _vis_frames and _HAS_PIL:
                        _gif_path = str(result_path / "vis_animation.gif")
                        _u8 = [(_f * 255).astype(np.uint8) for _f in _vis_frames]
                        _pf = [_PILImage.fromarray(_f) for _f in _u8]
                        _pf[0].save(_gif_path, save_all=True,
                                    append_images=_pf[1:], duration=250, loop=0)
                        _wb.log({"pbir/animation": _wb.Video(_gif_path, fps=4, format="gif")})

                scene.clear_cache()

                print("Done.")

                return scene
