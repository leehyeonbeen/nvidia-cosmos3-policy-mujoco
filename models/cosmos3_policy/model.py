"""Run one Cosmos3 model through Diffusers: camera image and prompt in, the next chunk of actions out.
scripts/policy_server.py serves it to the simulations.
"""

import logging
import time

import numpy as np

# hide torch's Mac redirects warning
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").addFilter(
    lambda r: False
)
import torch
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import CosmosActionCondition
from PIL import Image

from models.cosmos3_policy.actions import joint_targets
from models.cosmos3_policy.pipeline import decode_video_latent, load_pipeline
from models.cosmos3_policy.specs import CONFIGS
from utils.episode_log import stored


class Cosmos3Model:
    def __init__(self, model_name, dtype=torch.bfloat16):
        """Loads a Cosmos3 model with its settings.

        Args:
            model_name: A model in CONFIGS (specs.py), e.g. "edge-droid-policy".
            dtype: The model's torch dtype, passed to load_pipeline. Defaults to torch.bfloat16.
        """
        self.cfg = CONFIGS[model_name]
        self.dtype = dtype
        self.pipe = load_pipeline(self.cfg.path, self.cfg.shift, dtype)
        self.pipe.view_description = self.cfg.view_description
        self.generator = torch.Generator("cpu")
        self.generator.seed()  # random: a new Generator would start from torch's fixed default seed

    def reset(self, seed=None):
        """Starts an episode's noise from seed; random when the run gives none.

        Args:
            seed: The episode's noise seed, or None for a random one. Defaults to None.
        """
        if seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(seed)

    def generate(self, image, prompt, state, raw_actions=None):
        """Runs one model call.

        Args:
            image: The model's input image.
            prompt: The text prompt.
            state: Joint angles + gripper closedness; used only by models that read the state (cfg.state).
            raw_actions: Base model actions [chunk, 10] from ee_actions. Given, the model runs forward dynamics
                and predicts only the video.

        Returns:
            The pipeline's output: the actions (.action) and the predicted video latent (.video), not decoded.
        """
        cfg = self.cfg
        state = state if cfg.state else None
        return self.pipe(
            prompt=prompt,
            action=CosmosActionCondition(
                mode="policy" if raw_actions is None else "forward_dynamics",
                raw_actions=raw_actions,
                chunk_size=cfg.chunk,
                domain_name=cfg.domain,
                resolution_tier=cfg.tier,
                image=Image.fromarray(image),
                view_point=cfg.view_point,
            ),
            fps=cfg.fps,
            num_inference_steps=cfg.steps,
            guidance_scale=cfg.guidance,
            guidance_interval=cfg.guidance_interval,
            use_system_prompt=False,
            state=state,
            generator=self.generator,
            output_type="latent",  # decoding is slow: only on request (decode_video_latent)
        )

    def infer(self, image, prompt, state, decode_video=False):
        """Runs one model call.

        Args:
            image: The model's input image.
            prompt: The text prompt.
            state: Joint angles + gripper closedness, for models that read the state.
            decode_video: Whether to decode the predicted video. Defaults to False.

        Returns:
            Joint targets and gripper closedness (Policy-DROID models; None for base models), the call's inputs
            and outputs, and the decoded video frames (empty if not decoded).
        """
        cfg = self.cfg
        t1 = time.perf_counter()
        out = self.generate(image, prompt, state)
        t2 = time.perf_counter()
        print(
            f"{cfg.path} produced {cfg.chunk} {cfg.action} actions (future {cfg.chunk/cfg.fps:.2f}s) in {t2-t1:.2f}s."
        )
        joints = gripper = None
        if cfg.action == "joint_pos":  # Policy-DROID models: joint angles
            joints, gripper = joint_targets(out.action[0].float().numpy())
        call = dict(
            input_image=np.asarray(image, dtype=np.uint8),
            input_state=np.asarray(state, np.float32) if cfg.state and state is not None else np.zeros(0, np.float32),
            output_action=stored(out.action[0]),
            output_video_latent=stored(out.video[0]),
        )
        video = []
        if decode_video:
            t1 = time.perf_counter()
            video = decode_video_latent(self.pipe.vae, out.video)
            print(f"Video latents decoded in {time.perf_counter()-t1:.2f}s.")
        return joints, gripper, call, video
