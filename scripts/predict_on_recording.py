"""Run Cosmos3 on a recorded DROID episode, with the recorded camera images instead of a simulation.
Each model call gets the recorded image and the prompt, with no joint angles (except droid_sample, below).
Runs in the policy server's env.

Quick test (3 s, saved to outputs/_test):
    python scripts/predict_on_recording.py --model edge-droid-policy --seconds 3 --output-dir outputs/_test/recording
More:
    python scripts/predict_on_recording.py --model edge-droid-policy                       # NVIDIA's DROID example
    python scripts/predict_on_recording.py --model edge-base-policy --dataset droid_100 --episodes 3 4 5

Datasets (folders and cameras in DATASETS below):
- droid_example: NVIDIA's cookbook example, one episode. Run `git lfs pull` first (README).
- droid_100: 100 DROID episodes from https://huggingface.co/datasets/lerobot/droid_100
- droid_sample: 100 DROID failure/success pairs (scripts/download_droid_pairs.py). Pair i is episodes i and i+100.
  Policy-DROID also gets the recorded joints and gripper.

Saves outputs/recording/<dataset>/<episode>/<episode>_<dataset>_<model> (output_prefix) in EpisodeLog's files:
.parquet, .json, _camera.mp4 and, with --decode-video, _prediction.mp4. Episodes already saved are skipped unless
--overwrite.
"""

import argparse
import glob
import json
import os
import sys
import traceback
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])  # repository root
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before importing torch

import imageio
import numpy as np
import pandas as pd

from models.cosmos3_policy.canvas import _resize_rgb_uint8, compose_droid
from models.cosmos3_policy.model import Cosmos3Model
from utils.episode_log import EpisodeLog
from utils.runner import MODELS, episodes_to_run

EXAMPLES = (
    "models/cosmos-framework/.agents/skills/cookbooks/cosmos3/generator/action/assets"
)
# name: (folder, cameras), tiled as top, bottom left, bottom right (the DROID layout)
DATASETS = {
    "droid_example": (
        f"{EXAMPLES}/droid_lerobot_example",
        [
            "observation.image.wrist_image_left",
            "observation.image.exterior_image_1_left",
            "observation.image.exterior_image_2_left",
        ],
    ),
    "droid_100": (
        "datasets/droid_100",
        [
            "observation.images.wrist_image_left",
            "observation.images.exterior_image_1_left",
            "observation.images.exterior_image_2_left",
        ],
    ),
    "droid_sample": ("datasets/droid_sample", ["wrist", "ext1", "ext2"]),  # the metadata's <camera>_mp4_path
}
DEFAULT_PROMPT = "Pick up the object and place it in the target container."  # NVIDIA's notebook prompt


def output_prefix(output_dir, dataset, episode):
    """Returns the start of an episode's output paths, <output_dir>/<dataset>/<episode>/<episode>_<dataset>.

    Args:
        output_dir: The folder all outputs go under.
        dataset: A name in DATASETS.
        episode: The episode's index in the dataset.

    Returns:
        The path start, with "_" in the last <dataset> written as "-". droid_sample episodes are named by their
        DROID folder, the others ep<index>.
    """
    if dataset == "droid_sample":
        name = os.path.basename(droid_sample_episodes(DATASETS[dataset][0])[episode])
    else:
        name = f"ep{episode}"
    return f"{output_dir}/{dataset}/{name}/{name}_{dataset.replace('_', '-')}"


def tile(views):
    """Returns one image per time step: the three cameras tiled in the DROID layout.

    Args:
        views: The three cameras' frames, one list per camera.

    Returns:
        A list of images.
    """
    return [compose_droid(*(frames[t] for frames in views)) for t in range(min(len(frames) for frames in views))]


def load_episode(folder, cameras, episode, seconds):
    """Loads an episode of a LeRobot dataset: camera images, fps and task text, up to `seconds` long.

    Args:
        folder: The dataset folder.
        cameras: The three camera keys, tiled in the DROID layout.
        episode: The episode_index.
        seconds: The most to load, in seconds.

    Returns:
        The camera images, the fps and the task text (None if the episode has none).
    """
    fps = json.load(open(f"{folder}/meta/info.json"))["fps"]
    episodes = pd.concat(
        pd.read_parquet(p)
        for p in sorted(glob.glob(f"{folder}/meta/episodes/*/*.parquet"))
    )
    ep = episodes[episodes["episode_index"] == episode].iloc[0]
    n = min(int(ep["length"]), round(seconds * fps))

    views = []
    for cam in cameras:
        # one-episode examples may leave out where the episode sits in the video files
        chunk, file = int(ep.get(f"videos/{cam}/chunk_index", 0)), int(
            ep.get(f"videos/{cam}/file_index", 0)
        )
        path = f"{folder}/videos/{cam}/chunk-{chunk:03d}/file-{file:03d}.mp4"
        start = round(float(ep.get(f"videos/{cam}/from_timestamp", 0)) * fps)
        reader = imageio.get_reader(path)
        frames = []
        for i, frame in enumerate(reader):
            if i >= start:
                frames.append(frame)
            if len(frames) == n:
                break
        reader.close()
        views.append(frames)
    # droid_100's cameras are 320x180. Enlarge them to DROID's 640x360: the pipeline never enlarges a small image.
    for frames in views:
        for i, f in enumerate(frames):
            if f.shape[1] < 640:
                frames[i] = _resize_rgb_uint8(f, (f.shape[0] * 640 // f.shape[1], 640))
    images = tile(views)
    tasks = [t for t in ep.get("tasks", []) if any(c.isalpha() for c in str(t))]
    return images, fps, tasks[0] if tasks else None


def droid_sample_episodes(folder):
    return sorted(glob.glob(f"{folder}/failure/*")) + sorted(
        glob.glob(f"{folder}/success/*")
    )


def load_droid_sample_episode(folder, cameras, episode, seconds):
    """Loads a DROID sample episode in load_episode's form, plus its recorded robot and success label.

    Args:
        folder: The DROID sample folder, with pairs.json.
        cameras: The metadata's <camera>_mp4_path names: "wrist", "ext1" and "ext2".
        episode: The episode's index: failures first, then successes.
        seconds: The most to load, in seconds.

    Returns:
        The camera images, the fps (15), the pair's prompt and the recorded robot (joints, gripper,
        cartesian, gripper_command and success).
    """
    import h5py  # DROID sample only

    path, fps = droid_sample_episodes(folder)[episode], 15
    meta = json.load(open(glob.glob(f"{path}/metadata_*.json")[0]))
    views = [
        [
            _resize_rgb_uint8(np.asarray(f), (360, 640))
            for f in imageio.get_reader(
                f"{path}/{os.path.basename(meta[f'{c}_mp4_path'])}"
            )
        ]
        for c in cameras
    ]
    h5 = h5py.File(f"{path}/trajectory.h5")
    s = h5["observation/robot_state"]
    n = min([len(v) for v in views] + [len(s["joint_positions"]), round(seconds * fps)])
    images = tile([frames[:n] for frames in views])
    # DROID annotates only successes: a pair shares the success's prompt
    prompts = {}
    for pair in json.load(open(f"{folder}/pairs.json")):
        prompts[pair["failure"]] = pair["prompt"]
        prompts[pair["success"]] = pair["prompt"]
    robot = dict(
        joints=s["joint_positions"][:n],
        gripper=s["gripper_position"][:n],
        cartesian=s["cartesian_position"][:n],
        gripper_command=h5["action/gripper_position"][:n],
        success=meta["success"],
    )
    return images, fps, prompts[os.path.basename(path)], robot


def load(dataset, episode, seconds):
    """Loads an episode's camera images, fps, task text and recorded robot (DROID sample only, else None).

    Args:
        dataset: A name in DATASETS.
        episode: The episode's index in the dataset.
        seconds: The most to load, in seconds.

    Returns:
        The camera images, fps, task text and recorded robot.
    """
    folder, cameras = DATASETS[dataset]
    if dataset == "droid_sample":
        return load_droid_sample_episode(folder, cameras, episode, seconds)
    return *load_episode(folder, cameras, episode, seconds), None


def run_episode(cosmos, dataset, episode, out_path, prompt=None, seconds=1e9, decode_video=False):
    """Runs the model on one recorded episode and saves it to out_path*.

    Args:
        cosmos: The Cosmos3Model.
        dataset: A name in DATASETS.
        episode: The episode's index in the dataset, also its noise seed.
        out_path: The start of the episode's output paths (EpisodeLog's files).
        prompt: The task instruction. Defaults to None: the episode's own, else DEFAULT_PROMPT.
        seconds: The most to run, in seconds. Defaults to 1e9 (all of it).
        decode_video: Whether to decode and save the predicted video. Defaults to False.

    Returns:
        The episode's success label, if any.
    """
    images, fps, task, robot = load(dataset, episode, seconds)
    cfg = cosmos.cfg
    assert fps == cfg.fps, "the recording's frames per second must match the model's"
    cosmos.reset(episode)  # the episode's noise seed: its index in the dataset
    info = dict(
        dataset=dataset,
        episode=episode,
        seed=episode,
        prompt=prompt or task or DEFAULT_PROMPT,
    )
    log = EpisodeLog(cosmos, info)
    for t, image in enumerate(images):
        if t % cfg.chunk:
            log.add(image)
            continue
        # Policy-DROID on DROID sample: the recorded joints + gripper closedness.
        # No joint state otherwise; NVIDIA's cookbook gives none either:
        # https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/.agents/skills/cookbooks/cosmos3/generator/action/run_policy_with_diffusers.ipynb
        state = (
            np.append(robot["joints"][t], robot["gripper"][t])
            if robot and cfg.state
            else None
        )
        *_, call, video = cosmos.infer(image, info["prompt"], state, decode_video)
        log.add_prediction(call, video)
        log.add(image)
    label = (
        dict(success=robot["success"]) if robot else {}
    )  # DROID sample: the operator's label
    return log.save(out_path, fps, **label).get("success")


def main(dataset, episodes, model_name, prompt, seconds, output_dir, overwrite, decode_video):
    """Runs the model on the chosen episodes of a dataset, skipping those already saved unless overwrite.

    Args:
        dataset: A name in DATASETS.
        episodes: The episodes' indices in the dataset.
        model_name: One of MODELS in utils/runner.py.
        prompt: The task instruction; None: each episode's own, else DEFAULT_PROMPT.
        seconds: The most of each episode to use, in seconds.
        output_dir: The folder all outputs go under.
        overwrite: Whether to run episodes already saved.
        decode_video: Whether to decode and save the predicted video.
    """
    out_path = lambda e: os.path.abspath(
        f"{output_prefix(output_dir, dataset, e)}_{model_name}"
    )
    json_path = lambda e: os.path.relpath(
        out_path(e) + ".json", os.path.abspath(output_dir)
    )
    todo = (
        episodes
        if overwrite
        else episodes_to_run(episodes, json_path, output_dir, 0, 1)
    )
    if todo:
        cosmos = Cosmos3Model(model_name)
        for e in todo:
            try:
                result = run_episode(cosmos, dataset, e, out_path(e), prompt, seconds, decode_video)
            except Exception:  # writes no .json, so the next run retries it
                traceback.print_exc()
                result = "error"
            print(f"--- {os.path.basename(out_path(e))}: {result}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", choices=list(DATASETS), default="droid_example")
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[0],
        help="episode indices in the dataset",
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        choices=MODELS,
        default="edge-droid-policy",

    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="task instruction; default: the episode's own, if it has one",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=1e9,
        help="use at most this much of the episode, for short tests (default: all of it)",
    )
    parser.add_argument("--output-dir", default="outputs/recording", help="where the episodes are saved")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--decode-video", action="store_true", help="save the predicted video (slow)")
    main(**vars(parser.parse_args()))
