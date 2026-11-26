"""Settings for each model, all on DROID's Franka. NVIDIA's sources:

Cosmos 3 report, Table 21: https://arxiv.org/abs/2606.02800
Policy server instructions: https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/docs/action_policy_droid_server.md
Diffusers cookbook: https://github.com/NVIDIA/cosmos-framework/tree/bf492fe/.agents/skills/cookbooks/cosmos3/generator/action
"""

from dataclasses import dataclass, replace

import numpy as np


# Rotation from the robot's flange to the ee frame, with z along the fingers. Copied from NVIDIA's DROID data code:
# https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/cosmos_framework/data/generator/action/datasets/droid_lerobot_dataset.py#L55
_DROID_TO_OPENCV: np.ndarray = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

# openpi's DROID keys for the raw camera images in a policy server request, in Config.cameras order
CAMERA_KEYS = (
    "observation/wrist_image_left",
    "observation/exterior_image_1_left",
    "observation/exterior_image_2_left",
)


@dataclass(frozen=True)
class Config:
    path: str  # model folder
    domain: str  # the training dataset whose robot this one stands in for: "droid_lerobot"
    action: str  # "ee_delta" (base models) or "joint_pos" (Policy-DROID)
    state: bool  # whether the model also reads the joint angles and gripper
    fps: int  # actions per second
    chunk: int  # actions per model call
    cameras: tuple[str, ...]  # camera names, in image order
    size: tuple[int, int]  # (width, height) of each camera image
    view_point: str  # camera setup the model was trained with: "concat_view", "third_person_view", "ego_view" or "wrist_view"
    steps: int  # denoising steps per call
    guidance: float  # classifier-free guidance scale (1 = no guidance)
    shift: float  # the sampler's noise schedule shift (UniPC flow_shift)
    guidance_interval: tuple[float, float] | None = (
        None  # use guidance only between these timesteps
    )
    stats: str | None = (
        None  # action normalization stats, for scaling forward dynamics inputs (models/cosmos3_policy/actions.py)
    )
    tier: int = 480  # Diffusers' resolution_tier
    view_description: str | None = (
        None  # sentence on the image layout, added to the prompt
    )


# DROID image: wrist camera on top, two side cameras below. Layout sentence copied from NVIDIA's policy server:
# https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/cosmos_framework/scripts/action_policy_server_robolab.py#L77
DROID_VIEW = dict(
    cameras=("wrist_camera", "exo_camera_1", "exo_camera_2"),  # MolmoSpaces' names
    size=(640, 360),
    view_point="concat_view",
    view_description=(
        "The top row is from the wrist-mounted camera. "
        "The bottom row contains two horizontally concatenated third-person perspective views of the scene from "
        "opposite sides, with the robot visible."
    ),
)
# Base models: values from NVIDIA's Diffusers cookbook
EDGE_BASE = dict(
    path="models/cosmos3_checkpoints/Cosmos3-Edge",
    action="ee_delta",
    state=False,
    steps=30,
    guidance=1.0,
    shift=5.0,
)

# Keys: model names. The Nano and forward dynamics ("-fd") entries are added below.
CONFIGS = {
    # Edge base: as a DROID Franka
    "edge-base-policy": Config(
        **EDGE_BASE,
        **DROID_VIEW,
        domain="droid_lerobot",
        fps=15,
        chunk=16,
        stats="droid_lerobot_stats.json",
    ),
    # Edge Policy-DROID: values from NVIDIA's policy server instructions (the Edge command) and the server's defaults
    "edge-droid-policy": Config(
        **DROID_VIEW,
        path="models/cosmos3_checkpoints/Cosmos3-Edge-Policy-DROID",
        domain="droid_lerobot",
        action="joint_pos",
        state=True,
        fps=15,
        chunk=32,
        steps=4,
        guidance=3.0,
        shift=5.0,
        guidance_interval=(960, 1001),
    ),
}

# Nano: the same settings, from its own folders.
# Nano Policy-DROID uses guidance on every step: report Table 21. NVIDIA's Nano server command has no interval either.
CONFIGS["nano-base-policy"] = replace(CONFIGS["edge-base-policy"], path="models/cosmos3_checkpoints/Cosmos3-Nano")
CONFIGS["nano-droid-policy"] = replace(
    CONFIGS["edge-droid-policy"], path="models/cosmos3_checkpoints/Cosmos3-Nano-Policy-DROID", guidance_interval=None
)

# Base models in forward dynamics mode (scripts/forward_dynamics.py): the same, with shift 10.
# Shift from NVIDIA's forward dynamics notebook (FIXED_SAMPLING):
# https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/.agents/skills/cookbooks/cosmos3/generator/action/run_fd_with_diffusers.ipynb
for model_name in ("edge-base-policy", "nano-base-policy"):
    CONFIGS[model_name.replace("-policy", "-fd")] = replace(CONFIGS[model_name], shift=10.0)
