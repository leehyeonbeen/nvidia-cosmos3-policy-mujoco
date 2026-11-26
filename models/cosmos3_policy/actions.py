"""Convert Policy-DROID outputs into joint targets, and ee poses into base model actions for forward dynamics."""

import functools
import os
import sys

import numpy as np
import torch

FRAMEWORK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cosmos-framework")
sys.path.insert(0, FRAMEWORK)

from cosmos_framework.data.generator.action.action_normalization import load_action_stats
from cosmos_framework.data.generator.action.utils.action_processing import (
    resolve_action_normalization,
)
from cosmos_framework.data.generator.action.utils.pose_utils import pose_abs_to_rel

STATS_DIR = os.path.join(
    FRAMEWORK, "cosmos_framework", "data", "generator", "action", "normalizer_stats"
)


def joint_targets(action):
    """Converts a Policy-DROID model output into joint angles and gripper closedness.

    The model outputs gripper openness. Copied from NVIDIA's policy server's _format_outputs:
    https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/cosmos_framework/scripts/action_policy_server_robolab.py#L667

    Args:
        action: The model output [T, 8].

    Returns:
        Joint angles [T, 7] (rad) and gripper closedness [T].
    """
    return action[:, :7], 1.0 - action[:, 7]


@functools.cache
def action_stats(stats_file):
    """Loads the action stats, once per file.

    Args:
        stats_file: A path, or a file name in NVIDIA's normalizer_stats folder.

    Returns:
        The stats from NVIDIA's load_action_stats, a dict of arrays by name (e.g. "q01", "q99").
    """
    return load_action_stats(
        stats_file
        if os.path.exists(stats_file)
        else os.path.join(STATS_DIR, stats_file)
    )


def ee_actions(poses, gripper, stats_file):
    """Converts ee poses and gripper closedness into base model actions.

    Relative poses and gripper openness, ported from NVIDIA's data code:
    https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/cosmos_framework/data/generator/action/datasets/robomind_ur_dataset.py#L195
    Scaled with "quantile_rot" and clamped to [-1, 1], from NVIDIA's forward dynamics notebook:
    https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/.agents/skills/cookbooks/cosmos3/generator/action/run_fd_with_diffusers.ipynb
    https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/cosmos_framework/data/generator/action/utils/action_processing.py#L219

    Args:
        poses: The ee poses [T+1, 4, 4].
        gripper: Gripper closedness [T].
        stats_file: The action stats file (action_stats), for scaling with "quantile_rot".

    Returns:
        Base model actions [T, 10].
    """
    rel = pose_abs_to_rel(
        poses, rotation_format="rot6d", pose_convention="backward_framewise"
    )
    a = torch.from_numpy(
        np.concatenate([rel, 1.0 - np.asarray(gripper)[:, None]], axis=1)
    ).float()
    stats = {k: torch.from_numpy(v) for k, v in action_stats(stats_file).items()}
    return resolve_action_normalization(
        "quantile_rot", stats, apply_forward_clamp=True
    ).normalize_action(a)
