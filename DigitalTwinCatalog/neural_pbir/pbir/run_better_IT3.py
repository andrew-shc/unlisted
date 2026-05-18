# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import shutil
from argparse import ArgumentParser
from pathlib import Path

import gin
from mmcv import Config
from opt import optimize


@gin.configurable
def pipeline(configroot, ckptroot, dataset_class, gt_envmap_path=None, stage_overrides=None):
    cfg = Config.fromfile(ckptroot / "neural_surface_recon" / "config.py")
    dataset = dataset_class(dataroot=cfg.data.datadir, ckptroot=ckptroot)
    scene = dataset.get_scene()

    stages = [
        "microfacet_naive-envmap_sg",
        "microfacet_basis-envmap_ls",
        # "microfacet_basis-envmap_ls-shape_ls",
    ]

    result_root = Path(dataset.result_root)

    for stage in stages:
        print(f"Running stage {stage}...")
        stage_config = configroot / f"{stage}.gin"
        gin.parse_config_file(stage_config)

        if gt_envmap_path is not None:
            gin.bind_parameter("EnvmapSG.gt_envmap_path", gt_envmap_path)
            gin.bind_parameter("EnvmapLS.gt_envmap_path", gt_envmap_path)

        if stage_overrides and stage in stage_overrides:
            for param, value in stage_overrides[stage].items():
                if value is not None:
                    gin.bind_parameter(param, value)

        sub_result_path = result_root / stage
        scene = optimize(scene=scene, dataset=dataset, result_path=sub_result_path)
    shutil.copytree(result_root / stages[-1] / "final", result_root, dirs_exist_ok=True)
    print(f"Final results are written to {result_root}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("configroot", type=str, help="gin config directory")
    parser.add_argument("ckptroot", type=str, help="gin config directory")
    parser.add_argument("--gt_envmap_path", type=str, default=None,
                        help="path to a ground-truth envmap .exr; freezes envmap optimization")
    # stage 1: microfacet_naive-envmap_sg
    parser.add_argument("--s1_num_epochs", type=int, default=None, help="stage1 epochs (default: 10)")
    parser.add_argument("--s1_max_iter", type=int, default=None, help="stage1 max iters (default: 500)")
    parser.add_argument("--s1_d_lr", type=float, default=None, help="stage1 diffuse lr (default: 5e-3)")
    parser.add_argument("--s1_r_lr", type=float, default=None, help="stage1 roughness lr (default: 1e-3)")
    parser.add_argument("--s1_envmap_lr", type=float, default=None, help="stage1 envmap lr (default: 1e-3)")
    parser.add_argument("--s1_spp", type=int, default=None, help="stage1 samples per pixel (default: 64)")
    # stage 2: microfacet_basis-envmap_ls
    parser.add_argument("--s2_num_epochs", type=int, default=None, help="stage2 epochs (default: 10)")
    parser.add_argument("--s2_max_iter", type=int, default=1000, help="stage2 max iters (default: 500)")
    parser.add_argument("--s2_d_lr", type=float, default=None, help="stage2 diffuse lr (default: 1e-2)")
    parser.add_argument("--s2_r_lr", type=float, default=None, help="stage2 roughness lr (default: 5e-3)")
    parser.add_argument("--s2_envmap_lr", type=float, default=None, help="stage2 envmap lr (default: 1e-2)")
    parser.add_argument("--s2_spp", type=int, default=128, help="stage2 samples per pixel (default: 64)")
    args = parser.parse_args()

    configroot = Path(args.configroot)
    ckptroot = Path(args.ckptroot)

    gin.parse_config_file(configroot / "pipeline.gin")
    gin.bind_parameter("NeuralPBIRDataset.resultroot", str(ckptroot / "pbir_better"))

    _stage_overrides = {
        "microfacet_naive-envmap_sg": {
            "optimize.num_epochs": args.s1_num_epochs,
            "optimize.max_iter": args.s1_max_iter,
            "MicrofacetNaive.d_lr": args.s1_d_lr,
            "MicrofacetNaive.r_lr": args.s1_r_lr,
            "EnvmapSG.optimizer_kwargs": {"lr": args.s1_envmap_lr} if args.s1_envmap_lr else None,
            "opt/Renderer.render_options": {"spp": args.s1_spp, "sppe": 0, "sppse": 0, "npass": 1, "log_level": 0} if args.s1_spp else None,
        },
        "microfacet_basis-envmap_ls": {
            "optimize.num_epochs": args.s2_num_epochs,
            "optimize.max_iter": args.s2_max_iter,
            "MicrofacetBasis.d_lr": args.s2_d_lr,
            "MicrofacetBasis.r_lr": args.s2_r_lr,
            "EnvmapLS.optimizer_kwargs": {"lr": args.s2_envmap_lr, "lmbda": 1} if args.s2_envmap_lr else None,
            "opt/Renderer.render_options": {"spp": args.s2_spp, "sppe": 0, "sppse": 0, "npass": 1, "log_level": 0} if args.s2_spp else None,
        },
    }

    pipeline(configroot=configroot, ckptroot=ckptroot, gt_envmap_path=args.gt_envmap_path,
             stage_overrides=_stage_overrides)
