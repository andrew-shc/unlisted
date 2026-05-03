"""
Monkey-patch Trellis2ImageTo3DPipelineRelaxFlow with partial-t early stopping.

Adds `run_relaxflow_partial(..., shape_slat_stop_t=0.0, tex_slat_stop_t=0.0)`
which runs the full pipeline but halts each SLAT diffusion stage at the given t
value (0.0 = fully denoised, 1.0 = pure noise). Useful for visualizing what
intermediate latents decode to.

Usage:
    import trellis2_partial_patch   # applies patches on import
    outputs = pipeline.run_relaxflow_partial(
        image, prior_images,
        shape_slat_stop_t=0.5,  # stop shape SLAT at t=0.5
        tex_slat_stop_t=0.0,    # run tex SLAT to completion
        pipeline_type="1024_cascade",
        ...
    )
"""

import types
import numpy as np
import torch
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Union

from trellis2.pipelines.relaxflow_pipeline import (
    Trellis2ImageTo3DPipelineRelaxFlow,
    RELAXFLOWAttnBlurContext,
    _clone_noise,
    FLOW_BLEND_FNS,
    resolve_flow_blend,
    resolve_gating_schedule,
)
from trellis2.modules.sparse.basic import SparseTensor


# ---------------------------------------------------------------------------
# Patched core sampling loop
# ---------------------------------------------------------------------------

def _patched_sample_relaxflow_flow(
    self,
    model,
    sampler,
    noise,
    cond_obs: Dict[str, torch.Tensor],
    cond_prior: Dict[str, torch.Tensor],
    *,
    steps: int,
    rescale_t: float,
    gating_schedule: Callable[[int, int], float],
    gating_args: Dict[str, object],
    prior_weight: float,
    prior_blur_sigma: float,
    blur_attn_type: str,
    flow_combine_fn: Callable[..., torch.Tensor],
    flow_blend_args: Dict[str, object],
    infer_params: Dict[str, object],
    verbose: bool,
    run_branches: Optional[List[str]] = None,
    stop_t: float = 0.0,
    **kwargs,
):
    if run_branches is None:
        run_branches = ["blend", "obs_only", "prior_only"]
    do_blend = "blend" in run_branches
    do_obs = "obs_only" in run_branches
    do_prior = "prior_only" in run_branches

    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    t_pairs = list((float(t_seq[i]), float(t_seq[i + 1])) for i in range(steps))
    alpha_seq = self._build_alpha_schedule(steps, gating_schedule, gating_args)

    blur_context = RELAXFLOWAttnBlurContext(
        model,
        sigma=prior_blur_sigma,
        attn_type=blur_attn_type,
    )

    def run_branch(noise_branch, cond_bundle: Dict[str, torch.Tensor], use_blur: bool):
        sample = _clone_noise(noise_branch)
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            if t <= stop_t:
                break
            t_prev = max(t_prev, stop_t)
            infer_fn = lambda s, _t: self._infer_velocity(
                sampler, model, s, _t,
                cond_bundle["cond"], cond_bundle.get("neg_cond"),
                infer_params, **kwargs,
            )
            if use_blur:
                with blur_context:
                    pred_v = infer_fn(sample, t)
            else:
                pred_v = infer_fn(sample, t)
            sample = sample - (t - t_prev) * pred_v
            if t_prev <= stop_t:
                break
        return sample

    obs_only = run_branch(noise, cond_obs, use_blur=False) if do_obs else None
    prior_only = run_branch(noise, cond_prior, use_blur=True) if do_prior else None

    if not do_blend:
        return None, obs_only, prior_only

    sample = _clone_noise(noise)
    loop = tqdm(t_pairs, desc="Sampling", disable=not verbose)
    for idx, (t, t_prev) in enumerate(loop):
        if t <= stop_t:
            break
        t_prev = max(t_prev, stop_t)
        alpha_t = alpha_seq[min(idx, len(alpha_seq) - 1)]
        if prior_weight <= 0 or alpha_t <= 0:
            pred_v = self._infer_velocity(
                sampler, model, sample, t,
                cond_obs["cond"], cond_obs.get("neg_cond"),
                infer_params, **kwargs,
            )
        else:
            pred_v_obs = self._infer_velocity(
                sampler, model, sample, t,
                cond_obs["cond"], cond_obs.get("neg_cond"),
                infer_params, **kwargs,
            )
            with blur_context:
                pred_v_prior = self._infer_velocity(
                    sampler, model, sample, t,
                    cond_prior["cond"], cond_prior.get("neg_cond"),
                    infer_params, **kwargs,
                )
            pred_v = self._blend_velocity(
                flow_combine_fn, pred_v_obs, pred_v_prior,
                alpha_t, prior_weight, flow_blend_args,
            )
        sample = sample - (t - t_prev) * pred_v
        if t_prev <= stop_t:
            break

    return sample, obs_only, prior_only


# ---------------------------------------------------------------------------
# Patched shape SLAT sampler (adds stop_t)
# ---------------------------------------------------------------------------

def _patched_sample_shape_slat_relaxflow(
    self,
    cond_obs: Dict[str, torch.Tensor],
    coords: torch.Tensor,
    prior_tokens: torch.Tensor,
    flow_model: torch.nn.Module,
    *,
    gating_schedule: Callable[[int, int], float],
    prior_weight: float = 1.0,
    gating_args: Optional[dict] = None,
    inference_steps: Optional[int] = None,
    prior_blur_sigma: float = 2.5,
    blur_attn_type: str = "self",
    flow_combine_fn: Optional[Callable[..., torch.Tensor]] = None,
    flow_blend_args: Optional[dict] = None,
    sampler_params: Optional[dict] = None,
    run_branches: List[str] = [],
    stop_t: float = 0.0,
):
    gating_args = gating_args or {}
    flow_blend_args = flow_blend_args or {}
    sampler_params = sampler_params or {}

    noise = SparseTensor(
        feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
        coords=coords,
    )

    resolved_sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
    steps = inference_steps or resolved_sampler_params.get("steps", 50)
    rescale_t = resolved_sampler_params.get("rescale_t", 1.0)
    verbose = resolved_sampler_params.get("verbose", True)
    infer_params = {
        k: resolved_sampler_params[k]
        for k in ("guidance_strength", "guidance_interval")
        if k in resolved_sampler_params
    }
    cond_prior = {
        "cond": prior_tokens,
        "neg_cond": torch.zeros_like(prior_tokens),
    }
    flow_combine_fn = resolve_flow_blend(flow_combine_fn, None, FLOW_BLEND_FNS["linear"])

    if self.low_vram:
        flow_model.to(self.device)
    main_slat, obs_slat, prior_slat = self._sample_relaxflow_flow(
        flow_model,
        self.shape_slat_sampler,
        noise,
        cond_obs,
        cond_prior,
        steps=steps,
        rescale_t=rescale_t,
        gating_schedule=gating_schedule,
        gating_args=gating_args,
        prior_weight=prior_weight,
        prior_blur_sigma=prior_blur_sigma,
        blur_attn_type=blur_attn_type,
        flow_combine_fn=flow_combine_fn,
        flow_blend_args=flow_blend_args,
        infer_params=infer_params,
        verbose=verbose,
        run_branches=run_branches,
        stop_t=stop_t,
    )
    if self.low_vram:
        flow_model.cpu()

    def _denorm(slat):
        if slat is None:
            return None
        std = torch.tensor(self.shape_slat_normalization["std"])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization["mean"])[None].to(slat.device)
        return slat * std + mean

    main_slat = _denorm(main_slat)
    return {
        "blend": main_slat,
        "obs_only": _denorm(obs_slat),
        "prior_only": _denorm(prior_slat),
    }


# ---------------------------------------------------------------------------
# Patched tex SLAT sampler (adds stop_t)
# ---------------------------------------------------------------------------

def _patched_sample_tex_slat_relaxflow(
    self,
    cond_obs: Dict[str, torch.Tensor],
    shape_slat: SparseTensor,
    prior_tokens: torch.Tensor,
    flow_model: torch.nn.Module,
    *,
    gating_schedule: Callable[[int, int], float],
    prior_weight: float = 1.0,
    gating_args: Optional[dict] = None,
    inference_steps: Optional[int] = None,
    prior_blur_sigma: float = 2.5,
    blur_attn_type: str = "self",
    flow_combine_fn: Optional[Callable[..., torch.Tensor]] = None,
    flow_blend_args: Optional[dict] = None,
    sampler_params: Optional[dict] = None,
    run_branches: List[str] = [],
    stop_t: float = 0.0,
):
    gating_args = gating_args or {}
    flow_blend_args = flow_blend_args or {}
    sampler_params = sampler_params or {}

    std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
    mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
    shape_slat_norm = (shape_slat - mean) / std

    in_channels = flow_model.in_channels if isinstance(flow_model, torch.nn.Module) else flow_model[0].in_channels
    noise = shape_slat_norm.replace(
        feats=torch.randn(shape_slat_norm.coords.shape[0], in_channels - shape_slat_norm.feats.shape[1]).to(self.device)
    )

    resolved_sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
    steps = inference_steps or resolved_sampler_params.get("steps", 50)
    rescale_t = resolved_sampler_params.get("rescale_t", 1.0)
    verbose = resolved_sampler_params.get("verbose", True)
    infer_params = {
        k: resolved_sampler_params[k]
        for k in ("guidance_strength", "guidance_interval")
        if k in resolved_sampler_params
    }
    cond_prior = {
        "cond": prior_tokens,
        "neg_cond": torch.zeros_like(prior_tokens),
    }
    flow_combine_fn = resolve_flow_blend(flow_combine_fn, None, FLOW_BLEND_FNS["linear"])

    if self.low_vram:
        flow_model.to(self.device)
    main_slat, obs_slat, prior_slat = self._sample_relaxflow_flow(
        flow_model,
        self.tex_slat_sampler,
        noise,
        cond_obs,
        cond_prior,
        steps=steps,
        rescale_t=rescale_t,
        gating_schedule=gating_schedule,
        gating_args=gating_args,
        prior_weight=prior_weight,
        prior_blur_sigma=prior_blur_sigma,
        blur_attn_type=blur_attn_type,
        flow_combine_fn=flow_combine_fn,
        flow_blend_args=flow_blend_args,
        infer_params=infer_params,
        verbose=verbose,
        run_branches=run_branches,
        stop_t=stop_t,
        concat_cond=shape_slat_norm,
    )
    if self.low_vram:
        flow_model.cpu()

    _device = main_slat.device if main_slat is not None else self.device
    std = torch.tensor(self.tex_slat_normalization["std"])[None].to(_device)
    mean = torch.tensor(self.tex_slat_normalization["mean"])[None].to(_device)

    def _denorm(slat):
        if slat is None:
            return None
        return slat * std + mean

    main_slat = _denorm(main_slat)
    return {
        "blend": main_slat,
        "obs_only": _denorm(obs_slat),
        "prior_only": _denorm(prior_slat),
    }


# ---------------------------------------------------------------------------
# Patched sparse structure sampler (adds stop_t)
# ---------------------------------------------------------------------------

def _patched_sample_sparse_structure_relaxflow(
    self,
    cond_obs: Dict[str, torch.Tensor],
    prior_tokens: torch.Tensor,
    resolution: int,
    *,
    num_samples: int = 1,
    gating_schedule: Callable[[int, int], float],
    prior_weight: float = 1.0,
    gating_args: Optional[dict] = None,
    inference_steps: Optional[int] = None,
    prior_blur_sigma: float = 2.5,
    blur_attn_type: str = "self",
    flow_combine_fn: Optional[Callable[..., torch.Tensor]] = None,
    flow_blend_args: Optional[dict] = None,
    sampler_params: Optional[dict] = None,
    run_branches: List[str] = [],
    stop_t: float = 0.0,
):
    gating_args = gating_args or {}
    flow_blend_args = flow_blend_args or {}
    sampler_params = sampler_params or {}

    flow_model = self.models["sparse_structure_flow_model"]
    reso = flow_model.resolution
    noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)

    resolved_sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
    steps = inference_steps or resolved_sampler_params.get("steps", 50)
    rescale_t = resolved_sampler_params.get("rescale_t", 1.0)
    verbose = resolved_sampler_params.get("verbose", True)
    infer_params = {
        k: resolved_sampler_params[k]
        for k in ("guidance_strength", "guidance_interval")
        if k in resolved_sampler_params
    }
    cond_prior = {
        "cond": prior_tokens,
        "neg_cond": torch.zeros_like(prior_tokens),
    }
    flow_combine_fn = resolve_flow_blend(flow_combine_fn, None, FLOW_BLEND_FNS["linear"])

    if self.low_vram:
        flow_model.to(self.device)
    main_latent, obs_latent, prior_latent = self._sample_relaxflow_flow(
        flow_model,
        self.sparse_structure_sampler,
        noise,
        cond_obs,
        cond_prior,
        steps=steps,
        rescale_t=rescale_t,
        gating_schedule=gating_schedule,
        gating_args=gating_args,
        prior_weight=prior_weight,
        prior_blur_sigma=prior_blur_sigma,
        blur_attn_type=blur_attn_type,
        flow_combine_fn=flow_combine_fn,
        flow_blend_args=flow_blend_args,
        infer_params=infer_params,
        verbose=verbose,
        run_branches=run_branches,
        stop_t=stop_t,
    )
    if self.low_vram:
        flow_model.cpu()

    decoder = self.models["sparse_structure_decoder"]
    if self.low_vram:
        decoder.to(self.device)

    def _latent_to_coords(latent):
        if latent is None:
            return None
        occ = decoder(latent) > 0
        if resolution != occ.shape[2]:
            ratio = occ.shape[2] // resolution
            occ = torch.nn.functional.max_pool3d(occ.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(occ > 0)[:, [0, 2, 3, 4]].int()
        return coords if coords.shape[0] > 0 else None  # return None on empty

    coords_main  = _latent_to_coords(main_latent)
    coords_obs   = _latent_to_coords(obs_latent)
    coords_prior = _latent_to_coords(prior_latent)
    if self.low_vram:
        decoder.cpu()

    return {
        "blend":     coords_main,
        "obs_only":  coords_obs,
        "prior_only": coords_prior,
    }


# ---------------------------------------------------------------------------
# New pipeline entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_relaxflow_partial(
    self,
    image,
    prior_images: List,
    pipeline_type: Optional[str] = None,
    *,
    sparse_structure_stop_t: float = 0.0,
    shape_slat_stop_t: float = 0.0,
    tex_slat_stop_t: float = 0.0,
    run_branches: Optional[List[str]] = None,
    num_samples: int = 1,
    seed: Optional[int] = 42,
    sparse_structure_sampler_params: Optional[dict] = None,
    shape_slat_sampler_params: Optional[dict] = None,
    tex_slat_sampler_params: Optional[dict] = None,
    formats: List[str] = None,
    preprocess_image: bool = True,
    prior_weight: float = 1.0,
    prior_blur_sigma: float = 2.5,
    blur_attn_type: str = "self",
    gating_schedule: Optional[Callable[[int, int], float]] = None,
    gating_schedule_name: Optional[str] = None,
    gating_args: Optional[dict] = None,
    flow_combine_fn: Optional[Callable[..., torch.Tensor]] = None,
    flow_blend_name: Optional[str] = None,
    flow_blend_args: Optional[dict] = None,
    stage1_inference_steps: Optional[int] = None,
    prior_pooling: str = "concat",
    prior_pooling_temperature: float = 0.1,
    prior_pooling_agreement_boost: float = 0.0,
) -> dict:
    """Like run_relaxflow but stops SLAT sampling early at partial t.

    shape_slat_stop_t / tex_slat_stop_t: float in [0, 1).
        0.0 = fully denoised (normal behaviour).
        0.5 = halfway through denoising — features still noisy/blurry.

    run_branches: which branches to sample and decode.
        Values: "blend" (→ output key "relaxflow"), "obs_only", "prior_only".
        Defaults to all three. Pass e.g. ["obs_only"] to skip the others
        and avoid OOM from decoding unused meshes.
    """
    # "blend" is the internal name; output dict uses "relaxflow" for it.
    _ALL = ["blend", "obs_only", "prior_only"]
    active = set(run_branches if run_branches is not None else _ALL)
    pipeline_type = pipeline_type or self.default_pipeline_type

    if not prior_images:
        raise ValueError("Must provide at least one prior image for RelaxFlow.")

    if seed is not None and seed >= 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if preprocess_image:
        image = self.preprocess_image(image)
        prior_images = [self.preprocess_image(img) for img in prior_images]

    cond_O512 = self.get_cond([image], 512)
    cond_P512 = self.get_cond(prior_images, 512)
    P512_tokens_list = [cond_P512["cond"][i:i+1] for i in range(cond_P512["cond"].shape[0])]
    P512_tokens = self._pool_prior_tokens(
        P512_tokens_list, prior_pooling, prior_pooling_temperature, prior_pooling_agreement_boost
    )
    if pipeline_type != "512":
        cond_O1024 = self.get_cond([image], 1024)
        cond_P1024 = self.get_cond(prior_images, 1024)
        P1024_tokens_list = [cond_P1024["cond"][i:i+1] for i in range(cond_P1024["cond"].shape[0])]
        P1024_tokens = self._pool_prior_tokens(
            P1024_tokens_list, prior_pooling, prior_pooling_temperature, prior_pooling_agreement_boost
        )
    else:
        cond_O1024 = cond_P1024 = P1024_tokens = None

    ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}[pipeline_type]

    gating_args = gating_args or {}
    flow_blend_args = flow_blend_args or {}
    sparse_structure_sampler_params = sparse_structure_sampler_params or {}
    shape_slat_sampler_params = shape_slat_sampler_params or {}
    tex_slat_sampler_params = tex_slat_sampler_params or {}
    gating_schedule = resolve_gating_schedule(gating_schedule, gating_schedule_name, None)
    flow_combine_fn = resolve_flow_blend(flow_combine_fn, flow_blend_name, FLOW_BLEND_FNS["linear"])

    common_kwargs = {
        "gating_schedule": gating_schedule,
        "prior_weight": prior_weight,
        "gating_args": gating_args,
        "inference_steps": stage1_inference_steps,
        "prior_blur_sigma": prior_blur_sigma,
        "blur_attn_type": blur_attn_type,
        "flow_combine_fn": flow_combine_fn,
        "flow_blend_args": flow_blend_args,
    }

    # Stage 1: sparse structure — only request the branches we'll actually use
    ss_bundle = self.sample_sparse_structure_relaxflow(
        cond_O512, P512_tokens, ss_res,
        num_samples=num_samples,
        sampler_params=sparse_structure_sampler_params,
        run_branches=list(active),
        stop_t=sparse_structure_stop_t,
        **common_kwargs,
    )
    coords_blend  = ss_bundle.get("blend")   if "blend"     in active else None
    coords_obs    = ss_bundle.get("obs_only") if "obs_only"  in active else None
    coords_prior  = ss_bundle.get("prior_only") if "prior_only" in active else None

    shape_slat_blend = shape_slat_obs = shape_slat_prior = None
    tex_slat_blend   = tex_slat_obs   = tex_slat_prior   = None

    # Stage 2: shape SLAT (with early stopping, only for active branches)
    def _sample_shape(cond_obs, coords, prior_tokens, flow_model, branch, stop_t):
        if branch not in active or coords is None or coords.shape[0] == 0:
            return None
        bundle = self.sample_shape_slat_relaxflow(
            cond_obs, coords, prior_tokens, flow_model,
            sampler_params=shape_slat_sampler_params,
            run_branches=[branch],
            stop_t=stop_t,
            **common_kwargs,
        )
        return bundle[branch]

    def _sample_tex(cond_obs, shape_slat, prior_tokens, flow_model, branch, stop_t):
        if branch not in active or shape_slat is None:
            return None
        bundle = self.sample_tex_slat_relaxflow(
            cond_obs, shape_slat, prior_tokens, flow_model,
            sampler_params=tex_slat_sampler_params,
            run_branches=[branch],
            stop_t=stop_t,
            **common_kwargs,
        )
        return bundle[branch]

    if pipeline_type == "512":
        fm_shape = self.models["shape_slat_flow_model_512"]
        fm_tex = self.models["tex_slat_flow_model_512"]
        cond_O, P_tokens = cond_O512, P512_tokens
        res = {"blend": 512, "obs_only": 512, "prior_only": 512}

        shape_slat_blend = _sample_shape(cond_O, coords_blend, P_tokens, fm_shape, "blend",     shape_slat_stop_t)
        shape_slat_obs   = _sample_shape(cond_O, coords_obs,   P_tokens, fm_shape, "obs_only",  shape_slat_stop_t)
        shape_slat_prior = _sample_shape(cond_O, coords_prior, P_tokens, fm_shape, "prior_only",shape_slat_stop_t)
        tex_slat_blend   = _sample_tex(cond_O, shape_slat_blend, P_tokens, fm_tex, "blend",     tex_slat_stop_t)
        tex_slat_obs     = _sample_tex(cond_O, shape_slat_obs,   P_tokens, fm_tex, "obs_only",  tex_slat_stop_t)
        tex_slat_prior   = _sample_tex(cond_O, shape_slat_prior, P_tokens, fm_tex, "prior_only",tex_slat_stop_t)

    elif pipeline_type == "1024":
        fm_shape = self.models["shape_slat_flow_model_1024"]
        fm_tex = self.models["tex_slat_flow_model_1024"]
        cond_O, P_tokens = cond_O1024, P1024_tokens
        res = {"blend": 1024, "obs_only": 1024, "prior_only": 1024}

        shape_slat_blend = _sample_shape(cond_O, coords_blend, P_tokens, fm_shape, "blend",     shape_slat_stop_t)
        shape_slat_obs   = _sample_shape(cond_O, coords_obs,   P_tokens, fm_shape, "obs_only",  shape_slat_stop_t)
        shape_slat_prior = _sample_shape(cond_O, coords_prior, P_tokens, fm_shape, "prior_only",shape_slat_stop_t)
        tex_slat_blend   = _sample_tex(cond_O, shape_slat_blend, P_tokens, fm_tex, "blend",     tex_slat_stop_t)
        tex_slat_obs     = _sample_tex(cond_O, shape_slat_obs,   P_tokens, fm_tex, "obs_only",  tex_slat_stop_t)
        tex_slat_prior   = _sample_tex(cond_O, shape_slat_prior, P_tokens, fm_tex, "prior_only",tex_slat_stop_t)

    elif pipeline_type in ("1024_cascade", "1536_cascade"):
        hr_reso = 1024 if pipeline_type == "1024_cascade" else 1536
        fm_tex = self.models["tex_slat_flow_model_1024"]
        cond_O, P_tokens = cond_O1024, P1024_tokens

        def _cascade(coords, branch):
            if branch not in active or coords is None or coords.shape[0] == 0:
                return None, None
            lr_bundle = self.sample_shape_slat_relaxflow(
                cond_O512, coords, P512_tokens, self.models["shape_slat_flow_model_512"],
                sampler_params=shape_slat_sampler_params,
                run_branches=[branch], stop_t=0.0, **common_kwargs,
            )
            lr_slat = lr_bundle[branch]
            if lr_slat is None:
                return None, None
            if self.low_vram:
                self.models["shape_slat_decoder"].to(self.device)
                self.models["shape_slat_decoder"].low_vram = True
            hr_coords_raw = self.models["shape_slat_decoder"].upsample(lr_slat, upsample_times=4)
            if self.low_vram:
                self.models["shape_slat_decoder"].cpu()
                self.models["shape_slat_decoder"].low_vram = False
            hr_resolution = hr_reso
            while True:
                quant = torch.cat([
                    hr_coords_raw[:, :1],
                    ((hr_coords_raw[:, 1:] + 0.5) / 512 * (hr_resolution // 16)).int(),
                ], dim=1)
                coords_hr = quant.unique(dim=0)
                if coords_hr.shape[0] < 49152 or hr_resolution == 1024:
                    break
                hr_resolution -= 128
            hr_bundle = self.sample_shape_slat_relaxflow(
                cond_O, coords_hr, P_tokens, self.models["shape_slat_flow_model_1024"],
                sampler_params=shape_slat_sampler_params,
                run_branches=[branch], stop_t=shape_slat_stop_t, **common_kwargs,
            )
            return hr_bundle[branch], hr_resolution

        shape_slat_blend, res_blend = _cascade(coords_blend, "blend")
        shape_slat_obs,   res_obs   = _cascade(coords_obs,   "obs_only")
        shape_slat_prior, res_prior = _cascade(coords_prior, "prior_only")
        res = {"blend": res_blend, "obs_only": res_obs, "prior_only": res_prior}

        tex_slat_blend   = _sample_tex(cond_O, shape_slat_blend, P_tokens, fm_tex, "blend",     tex_slat_stop_t)
        tex_slat_obs     = _sample_tex(cond_O, shape_slat_obs,   P_tokens, fm_tex, "obs_only",  tex_slat_stop_t)
        tex_slat_prior   = _sample_tex(cond_O, shape_slat_prior, P_tokens, fm_tex, "prior_only",tex_slat_stop_t)

    else:
        raise ValueError(f"Unknown pipeline_type: {pipeline_type}")

    def _safe_decode(shape_slat, tex_slat, resolution):
        if shape_slat is None or tex_slat is None:
            return None
        return self.decode_latent(shape_slat, tex_slat, resolution)

    branch_outputs: Dict[str, dict] = {}
    branch_outputs["relaxflow"] = {
        "mesh": _safe_decode(shape_slat_blend, tex_slat_blend, res.get("blend")),
        "shape_slat": shape_slat_blend, "tex_slat": tex_slat_blend, "res": res.get("blend"),
    }
    branch_outputs["obs_only"] = {
        "mesh": _safe_decode(shape_slat_obs, tex_slat_obs, res.get("obs_only")),
        "shape_slat": shape_slat_obs, "tex_slat": tex_slat_obs, "res": res.get("obs_only"),
    }
    branch_outputs["prior_only"] = {
        "mesh": _safe_decode(shape_slat_prior, tex_slat_prior, res.get("prior_only")),
        "shape_slat": shape_slat_prior, "tex_slat": tex_slat_prior, "res": res.get("prior_only"),
    }
    return branch_outputs


# ---------------------------------------------------------------------------
# Apply patches
# ---------------------------------------------------------------------------

Trellis2ImageTo3DPipelineRelaxFlow._sample_relaxflow_flow = _patched_sample_relaxflow_flow
Trellis2ImageTo3DPipelineRelaxFlow.sample_sparse_structure_relaxflow = _patched_sample_sparse_structure_relaxflow
Trellis2ImageTo3DPipelineRelaxFlow.sample_shape_slat_relaxflow = _patched_sample_shape_slat_relaxflow
Trellis2ImageTo3DPipelineRelaxFlow.sample_tex_slat_relaxflow = _patched_sample_tex_slat_relaxflow
Trellis2ImageTo3DPipelineRelaxFlow.run_relaxflow_partial = run_relaxflow_partial
