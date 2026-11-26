"""Forward dynamics: the base model predicts the video of a recorded DROID episode (droid_sample) from its first camera
image, the prompt and the recorded actions. Runs in the policy server's env.

    python scripts/forward_dynamics.py --episodes 0 100

The episode is split into Policy-DROID's model calls (32 actions). Each call starts from the recorded camera image;
a call longer than one forward dynamics chunk (16 actions) goes on from the last predicted frame. One frame per action.

Writes outputs/recording/droid_sample/<episode>/<episode>_droid-sample_<model>_prediction: .parquet (one row per chunk:
its first frame, input actions and predicted video latent), .json and .mp4. A chunk starts from the recorded image at
start_frame, or from the last chunk's final frame.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])  # repository root
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before importing torch

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from models.cosmos3_policy.actions import ee_actions
from models.cosmos3_policy.model import Cosmos3Model
from models.cosmos3_policy.pipeline import decode_video_latent
from models.cosmos3_policy.specs import _DROID_TO_OPENCV, CONFIGS
from scripts.predict_on_recording import DATASETS, load_droid_sample_episode, output_prefix
from utils.episode_log import save_preview, stored, transform


def load_droid_sample(episode):
    """Loads a DROID sample episode: its ee poses, commanded gripper, camera images and info.

    Its recorded gripper poses are turned into ee poses. Port of NVIDIA's DROID data code, which trains DROID on the
    recorded poses:
    https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/cosmos_framework/data/generator/action/datasets/droid_lerobot_dataset.py#L428

    Args:
        episode: The episode's index in the DROID sample.

    Returns:
        The ee poses [T, 4, 4], the gripper commands [T], the camera images, and the episode's info, with its
        success label.
    """
    folder, cameras = DATASETS["droid_sample"]
    images, fps, prompt, robot = load_droid_sample_episode(folder, cameras, episode, 1e9)
    p = robot["cartesian"]  # x y z + extrinsic xyz Euler angles
    R = Rotation.from_euler("xyz", p[:, 3:]).as_matrix() @ _DROID_TO_OPENCV
    poses = np.array([transform(x, r) for x, r in zip(p[:, :3], R)])
    info = dict(dataset="droid_sample", episode=episode, prompt=prompt, success=robot["success"], fps=fps)
    return poses, robot["gripper_command"], images, info


def forward_dynamics(poses, gripper, camera, info, out, model):
    """Predicts an episode's video from its actions and saves it to out.parquet, .json and .mp4.

    Args:
        poses: The ee poses, one per frame [T, 4, 4].
        gripper: The gripper commands, one per frame.
        camera: The camera images, one per frame.
        info: The episode's info (load_droid_sample).
        out: The output path, without the extension.
        model: The base model, a Cosmos3Model in forward dynamics mode.
    """
    cfg = model.cfg
    assert info["fps"] == cfg.fps, "the episode's actions per second must match the base model's"
    gripper = np.round(gripper)  # as commanded: fully open or closed
    call = CONFIGS["edge-droid-policy"].chunk  # Policy-DROID's actions per model call

    frames, rows = [], []
    for i in range((len(poses) - 2) // cfg.chunk):  # whole chunks with T+1 poses
        s = slice(i * cfg.chunk, (i + 1) * cfg.chunk)
        # Each model call starts from the recorded camera image, so the video lines up with Policy-DROID's own
        # predicted video. NVIDIA's notebook instead chains every chunk from the last predicted frame:
        # https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/.agents/skills/cookbooks/cosmos3/generator/action/run_fd_with_diffusers.ipynb
        if s.start % call == 0:
            image = camera[s.start]
        raw_actions = ee_actions(poses[s.start : s.stop + 1], gripper[s], cfg.stats)
        model.generator.manual_seed(i)  # NVIDIA's notebook seeds each chunk with its index
        result = model.generate(image, info["prompt"], None, raw_actions=raw_actions)
        rows.append(
            dict(
                start_frame=s.start,
                input_action=raw_actions.numpy(),
                output_video_latent=stored(result.video[0]),
            )
        )
        video = decode_video_latent(model.pipe.vae, result.video)
        frames += video if i == 0 else video[1:]  # each chunk regenerates its first frame
        image = video[-1]  # a model call longer than one chunk goes on from here
    save_preview(out + ".mp4", frames, cfg.fps)
    shapes = {k: list(v.shape) for k, v in rows[0].items() if k != "start_frame"}
    table = pd.DataFrame({k: [r[k] if k == "start_frame" else r[k].ravel() for r in rows] for k in rows[0]})
    table.to_parquet(out + ".parquet")
    print(out + ".parquet")
    info = dict(
        info,
        model_config=asdict(cfg),
        dtype=str(model.dtype).removeprefix("torch."),
        model_call_shapes=shapes,
        model_call_bfloat16=["output_video_latent"] if rows[0]["output_video_latent"].dtype == np.uint16 else [],
    )
    with open(out + ".json", "w") as f:
        json.dump(info, f, indent=1)
    print(out + ".json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="DROID sample episodes: failures 0-99, then successes 100-199",
    )
    parser.add_argument("--model", dest="model_name", default="edge-base-fd", choices=["edge-base-fd", "nano-base-fd"])
    parser.add_argument("--output-dir", default="outputs/recording", help="where the episodes are saved")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model = None
    for e in args.episodes:
        out = f"{output_prefix(args.output_dir, 'droid_sample', e)}_{args.model_name}_prediction"
        if os.path.exists(out + ".parquet") and not args.overwrite:
            print(f"{out}.parquet exists, skipped")
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        model = model or Cosmos3Model(args.model_name)  # settings: models/cosmos3_policy/specs.py
        forward_dynamics(*load_droid_sample(e), out, model)
