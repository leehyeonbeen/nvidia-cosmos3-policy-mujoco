"""Diffusers' Cosmos3 pipeline, changed where it prepares the model's inputs differently from NVIDIA's policy server.

Diffusers source:
https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py
NVIDIA's policy server:
https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/cosmos_framework/scripts/action_policy_server_robolab.py
"""

import json
import types

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm
from diffusers.pipelines.cosmos import pipeline_cosmos3_omni as cosmos3
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import logging as diffusers_logging
from diffusers.video_processor import VideoProcessor


def use_full_precision():
    """Sets torch to use only full-precision GPU math: no TF32, no reduced-precision sums, no autotuned kernels."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_pipeline(
    path,
    flow_shift,
    dtype=torch.bfloat16,
    device="cuda" if torch.cuda.is_available() else "mps",
):
    """Loads a Cosmos3 model on the GPU: NVIDIA (cuda) if available, else the Mac GPU (mps).

    Args:
        path: The model folder.
        flow_shift: The UniPC sampler's flow_shift.
        dtype: bfloat16 (default): compiled; results differ slightly between GPUs.
            float32 (--full-precision): uncompiled; nearly the same results on every GPU.
        device: The GPU. Defaults to "cuda" if available, else "mps".

    Returns:
        The Cosmos3ActionPipeline.
    """
    diffusers_logging.disable_progress_bar()  # no "Loading pipeline components" bar
    if dtype == torch.float32:
        use_full_precision()
    pipe = Cosmos3ActionPipeline.from_pretrained(
        path, dtype=dtype, enable_safety_checker=False
    ).to(device)
    if dtype == torch.bfloat16:
        # Compile each transformer layer, without CUDA graphs, which break on repeated layers:
        # https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/cosmos_framework/model/generator/mot/parallelize_unified_mot.py#L431
        pipe.transformer.compile_repeated_blocks(
            fullgraph=True, dynamic=True, mode=None
        )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=flow_shift, use_karras_sigmas=False
    )
    pipe.set_progress_bar_config(disable=True)  # no progress bar for every model call
    return pipe


class Cosmos3ActionPipeline(Cosmos3OmniPipeline):
    state_row = None  # set only while a call with `state` runs
    view_description = None  # the image layout sentence (Config.view_description)

    def _build_action_json_prompt(self, description, *, view_point, **kwargs):
        """Adds the image layout sentence to Diffusers' prompt.

        Layout sentence, from NVIDIA's policy server:
        https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/cosmos_framework/scripts/action_policy_server_robolab.py#L631

        Args:
            description: The text Diffusers' version builds the prompt from.
            view_point: The camera setup; the layout sentence is added only for "concat_view".
            **kwargs: Passed on to Diffusers' version.

        Returns:
            The prompt, as a JSON string.
        """
        prompt = json.loads(
            Cosmos3OmniPipeline._build_action_json_prompt(
                description, view_point=view_point, **kwargs
            )
        )
        if view_point == "concat_view" and self.view_description:
            prompt["cinematography"]["framing"] += " " + self.view_description
        return json.dumps(prompt)

    def __call__(self, *args, state=None, guidance_interval=None, **kwargs):
        """Runs Diffusers' pipeline, adding the robot's state and a guidance interval when given.

        Args:
            *args: Passed on to Diffusers' pipeline.
            state: Policy-DROID models only: 7 joint angles (rad) + gripper closedness. Else None.
            guidance_interval: (low, high) timesteps where guidance is used, or None for every step.
            **kwargs: Passed on to Diffusers' pipeline; must hold guidance_scale when guidance_interval is given.

        Returns:
            Diffusers' output. With a state, its actions drop the state row.
        """
        if guidance_interval is not None:
            kwargs["callback_on_step_end"] = _guidance_window(
                kwargs["guidance_scale"], *guidance_interval
            )

        if state is not None:
            # The model reads gripper openness (1 - closedness):
            # https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/cosmos_framework/scripts/action_policy_server_robolab.py#L592
            self.state_row = torch.tensor(
                [*state[:7], 1.0 - state[7]], dtype=torch.float32
            )

        # Diffusers' pipeline; it calls the methods below
        try:
            output = super().__call__(*args, **kwargs)
        finally:
            self.state_row = None

        if state is not None:
            output.action = [rows[1:] for rows in output.action]  # drop the state row
        return output

    def prepare_latents(self, *args, **kwargs):
        """Puts the robot's current state in action row 0, marked as given (not generated).

        https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/cosmos_framework/scripts/action_policy_server_robolab.py#L597

        Args:
            *args: Passed on to Diffusers' prepare_latents.
            **kwargs: Passed on to Diffusers' prepare_latents.

        Returns:
            The 12 values of Diffusers' prepare_latents, in its order. With a state, action row 0 holds it,
            marked as given, and only its 8 action dimensions are used.
        """
        # the 12 values Diffusers 0.41 returns, in its order
        (
            latents,
            sound_latents,
            action_latents,  # [chunk, 64]: one row per action, padded to the model's 64 action dimensions
            fps,
            fps_sound,
            vision_condition_mask,
            sound_condition_mask,
            action_condition_mask,  # [chunk, 1]: 1 = given, 0 = generated
            action_domain_id,
            action_image_size,
            raw_action_dim_resolved,  # how many of the 64 action dimensions are used; the rest stay zero
            action_condition_frame_indexes,  # which action rows are given
        ) = super().prepare_latents(*args, **kwargs)

        if self.state_row is not None:
            row = torch.zeros_like(action_latents[:1])
            row[0, : len(self.state_row)] = self.state_row.to(row)
            action_latents = torch.cat([row, action_latents])
            action_condition_mask = torch.cat(
                [torch.ones_like(action_condition_mask[:1]), action_condition_mask]
            )
            action_condition_frame_indexes = [0]

            # Policy-DROID actions have 8 values: 7 joints + gripper. Diffusers would use 10 and leave the last 2 as noise.
            # NVIDIA's policy server uses 8:
            # https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/cosmos_framework/scripts/action_policy_server_robolab.py#L458
            raw_action_dim_resolved = len(self.state_row)
            action_latents[:, raw_action_dim_resolved:] = 0

        return (
            latents,
            sound_latents,
            action_latents,
            fps,
            fps_sound,
            vision_condition_mask,
            sound_condition_mask,
            action_condition_mask,
            action_domain_id,
            action_image_size,
            raw_action_dim_resolved,
            action_condition_frame_indexes,
        )

    def _prepare_action_video_conditioning(self, *args, **kwargs):
        """Pads the camera image without resizing it.

        Diffusers would shrink 640x540 to 640x528. NVIDIA's code pads only:
        https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/cosmos_framework/data/generator/action/utils/transforms.py#L129

        Args:
            *args: Passed on to Diffusers' version.
            **kwargs: Passed on to Diffusers' version.

        Returns:
            Diffusers' result, with the image padded, not resized.
        """
        config = self.video_processor.config
        scale = config.vae_scale_factor
        config.vae_scale_factor = (
            1  # Diffusers resizes to a multiple of this number (16); 1 = no resize
        )
        try:
            return super()._prepare_action_video_conditioning(*args, **kwargs)
        finally:
            config.vae_scale_factor = scale

    def _encode_video(self, frames):
        """Encodes only the camera image, not the placeholder frames after it.

        Same result, faster:
        https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/cosmos_framework/model/generator/omni_mot_model.py#L5389

        Args:
            frames: The camera image, then the placeholder frames, along dim 2.

        Returns:
            The video latent: the camera image's latent frame, then zeros for the placeholder frames.
        """
        first = super()._encode_video(frames[:, :, :1])
        latent_frames = (
            frames.shape[2] - 1
        ) // self.vae.config.scale_factor_temporal + 1
        # zeros for the placeholder frames: Diffusers replaces them with noise anyway
        return torch.cat(
            [
                first,
                first.new_zeros(*first.shape[:2], latent_frames - 1, *first.shape[3:]),
            ],
            dim=2,
        )

    def _prepare_action_segment(
        self,
        input_action_tokens,
        condition_frame_indexes,
        mrope_offset,
        action_fps,
        curr,
        device,
    ):
        """Puts the state row at the camera image's time (Diffusers puts it one frame later).

        https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/cosmos_framework/data/generator/action/utils/transforms.py#L396

        Args:
            input_action_tokens: The action rows, one time step each.
            condition_frame_indexes: Passed on to Diffusers' version.
            mrope_offset: The action rows' time offset.
            action_fps: Actions per second, used when the model modulates by fps.
            curr: Passed on to Diffusers' version.
            device: The device for the action positions.

        Returns:
            Diffusers' action segment; with a state, its action_mrope_ids start at the camera image's frame.
        """
        segment = super()._prepare_action_segment(
            input_action_tokens,
            condition_frame_indexes,
            mrope_offset,
            action_fps,
            curr,
            device,
        )
        if self.state_row is None:
            return segment

        # Copied from Diffusers' _prepare_action_segment, with start_frame_offset=0 instead of 1:
        # https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py#L662
        config = self.transformer.config
        action_len = input_action_tokens.shape[0]
        effective_fps = action_fps if config.enable_fps_modulation else None
        action_mrope_ids, _ = cosmos3.get_3d_mrope_ids_vae_tokens(
            grid_t=action_len,
            grid_h=1,
            grid_w=1,
            temporal_offset=mrope_offset,
            reset_spatial_indices=config.unified_3d_mrope_reset_spatial_ids,
            fps=effective_fps,
            base_fps=float(config.base_fps),
            temporal_compression_factor=1,
            base_temporal_compression_factor=self.vae_scale_factor_temporal,
            start_frame_offset=0,
        )
        segment["action_mrope_ids"] = action_mrope_ids.to(device)
        return segment


def _guidance_window(scale, t_lo, t_hi):
    """Returns a callback to run after each denoising step.

    The next step uses guidance `scale` if its timestep is between t_lo and t_hi, else 1 (no guidance).
    Port of NVIDIA's guidance interval:
    https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/cosmos_framework/model/generator/omni_mot_model.py#L4361

    Args:
        scale: The guidance scale between t_lo and t_hi.
        t_lo: The low end of the timestep interval.
        t_hi: The high end of the timestep interval.

    Returns:
        The callback (pipe, i, t, kwargs), for Diffusers' callback_on_step_end.
    """

    def callback(pipe, i, t, kwargs):
        timesteps = pipe.scheduler.timesteps
        if i + 1 < len(timesteps):
            pipe._guidance_scale = scale if t_lo < timesteps[i + 1] < t_hi else 1.0
        return {}

    return callback


def _rms_norm(self, x):
    return (
        F.normalize(x, dim=1 if self.channel_first else -1) * self.scale * self.gamma
        + self.bias
    )


def accelerate_diffusers_decoding(vae):
    """Replaces Diffusers' stock VAE decoding with cosmos-framework's.

    Channels-last convs and the RMS norm in bfloat16:
    https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/cosmos_framework/model/generator/tokenizers/wan2pt2_vae_4x16x16.py#L123

    Args:
        vae: The model's video decoder (VAE), changed in place; only on CUDA, once.
    """
    if (
        getattr(vae, "_accelerated_decoding", False) or vae.device.type != "cuda"
    ):  # ~5x slower on the Mac GPU
        return
    for module in [*vae.post_quant_conv.modules(), *vae.decoder.modules()]:
        if isinstance(module, torch.nn.Conv3d):
            module.weight.data = module.weight.to(memory_format=torch.channels_last_3d)
        elif isinstance(module, torch.nn.Conv2d):
            module.weight.data = module.weight.to(memory_format=torch.channels_last)
        elif isinstance(module, WanRMS_norm):
            module.forward = types.MethodType(_rms_norm, module)
    vae._accelerated_decoding = True


@torch.inference_mode()
def decode_video_latent(vae, latent):
    """Decodes a predicted video latent into frames, with the steps of Diffusers' pipeline (step 8).

    https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py

    Args:
        vae: The model's video decoder (VAE).
        latent: The predicted video latent [1, 48, T, H, W].

    Returns:
        The frames, uint8 [H, W, 3]: 1 + 4 (T - 1) of them.
    """
    accelerate_diffusers_decoding(vae)
    dtype = vae.dtype
    mean = (
        torch.tensor(vae.config.latents_mean, dtype=dtype)
        .view(1, -1, 1, 1, 1)
        .to(latent.device)
    )
    inv_std = (
        (1.0 / torch.tensor(vae.config.latents_std, dtype=dtype))
        .view(1, -1, 1, 1, 1)
        .to(latent.device)
    )
    scaled = latent.to(dtype) / inv_std + mean
    video = vae.decode(scaled).sample
    video = video.to(latent.dtype)
    video = VideoProcessor.denormalize(video[0]).permute(
        1, 2, 3, 0
    )  # [T,H,W,3] in [0, 1]
    return list((video.float() * 255).round().to(torch.uint8).cpu().numpy())
