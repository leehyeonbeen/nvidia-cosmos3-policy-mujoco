"""Run a Cosmos3 Policy-DROID model on one MolmoSpaces episode through MolmoSpaces' own evaluation
(molmo_spaces/evaluation/README.md), with MolmoSpaces' DROID Franka. run_all.py runs run_episode for each episode.
Needs the policy server (README, "Run").

Quick test (3 s):
    python scripts/molmospaces/run_demo.py --seconds 3
More:
    python scripts/molmospaces/run_demo.py --model nano-droid-policy
    python scripts/molmospaces/run_demo.py --bench Open-v1 --episode 3

Saves outputs/molmospaces_<model>_<bench>_ep<i>/: MolmoSpaces' results and cosmos3*.
"""

import argparse
import functools
import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # repository root
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before importing torch

from molmo_spaces.evaluation.benchmark_schema import load_all_episodes
from molmo_spaces.evaluation.eval_main import run_evaluation
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.utils import save_utils

from models.cosmos3_policy.specs import CONFIGS
from scripts.molmospaces.cosmos3_policy import Cosmos3EvalConfig, Cosmos3MolmoSpacesPolicy
from scripts.molmospaces.load_scene import BENCHES, HARD
from utils.episode_log import SETTLE_SECONDS
from utils.policy import time_limit
from utils.runner import (
    add_model_args,
    add_episode_args,
    skip,
    write_json,
)

# Policy-DROID cameras for the benches that name them differently (pi_policy.py:128)
CAMERAS = {
    b: ["wrist_camera_zed_mini", "droid_shoulder_light_randomization", "exo_camera_2"]
    for b in BENCHES
    if BENCHES[b].startswith(HARD)
}

# Skip MolmoSpaces' camera videos (save_utils.py:987); its scoring reads only .h5
save_utils.save_videos_from_raw_observations = lambda *args, **kwargs: None


@functools.cache
def bench_episodes(bench):
    """Loads the bench's episodes.

    Args:
        bench: One of BENCHES.

    Returns:
        The bench's episodes, from MolmoSpaces' load_all_episodes.
    """
    return load_all_episodes(ASSETS_DIR / "benchmarks" / BENCHES[bench])


def load_policy(
    model_name,
    full_precision,
    host,
    port,
    decode_video,
):
    """Returns the policy for the model.

    The eval config ends up in policy.config.

    Args:
        model_name: The model's name; it picks the checkpoint.
        full_precision: Whether to run the model in float32 instead of bfloat16.
        host: The policy server's host.
        port: The policy server's port.
        decode_video: Whether to decode and save the predicted video.

    Returns:
        A Cosmos3MolmoSpacesPolicy.
    """
    config = Cosmos3EvalConfig()
    config.policy_config.checkpoint_path = CONFIGS[model_name].path
    # one policy for all episodes (allowed by molmo_spaces/evaluation/README.md:241)
    return Cosmos3MolmoSpacesPolicy(config, None, decode_video, full_precision, host, port)


def run_episode(policy, episode, out_path, seconds=None, seed=None):
    """Runs one episode and saves it to out_path*; MolmoSpaces' results go to its folder.

    Args:
        policy: The policy from load_policy.
        episode: (bench, index).
        out_path: The saved files' path without the ending.
        seconds: The time limit; None: the episode's task_horizon_sec.
        seed: The episode's noise seed; None: random.

    Returns:
        Success (True or False), or why it was skipped.
    """
    bench, i = episode
    config = policy.config
    name = os.path.dirname(out_path)
    # the bench's cameras, for our policy and for MolmoSpaces
    cameras = type(config).model_fields["policy_config"].default.camera_names
    config.policy_config.camera_names = CAMERAS.get(bench, cameras)
    spec = bench_episodes(bench)[i]
    # MolmoSpaces' steps: round(seconds / dt) (eval_main.py:347); whole model calls after the settle
    dt = config.policy_dt_ms / 1000
    actions, limit = time_limit(seconds or spec.task["task_horizon_sec"], dt, policy.cosmos.cfg.chunk)
    outcome = dict(benchmark=BENCHES[bench], index=i, house=spec.house_index)
    # one episode (eval_main.py:743, --idx)
    policy.task = None  # MolmoSpaces sets it once the episode is set up
    policy.seed = seed
    results = run_evaluation(
        type(config),
        ASSETS_DIR / "benchmarks" / BENCHES[bench],
        checkpoint_path=config.policy_config.checkpoint_path,
        task_horizon_sec=(round(SETTLE_SECONDS / dt) + actions) * dt,
        output_dir=name,
        preloaded_policy=policy,
        episode_idx=i,
        camera_names_override=config.policy_config.camera_names,
    )
    if policy.task is None:  # MolmoSpaces' own evaluation skips it too
        not_run = "MolmoSpaces could not set it up"
        return skip(f"{out_path}.json", **outcome, not_run=not_run)
    assert results.total_count == 1, f"MolmoSpaces ran {results.total_count}"
    if policy.chunks.log is None:  # ended during the settle
        success = results.success_count == 1
        write_json(f"{out_path}.json", **outcome, success=success, ended_in_settle=True)
    else:
        success = policy.success_time is not None
        policy.save(out_path, limit, success, episode=f"{bench} ep{i}", **outcome)
    return success


def main(
    model_name,
    full_precision,
    host,
    port,
    bench,
    episode,
    seconds,
    decode_video,
    seed,
):
    """Runs one episode and saves it to outputs/molmospaces_<model>_<bench>_ep<i>/.

    Args:
        model_name: One of POLICY_MODELS in utils/runner.py.
        full_precision: Whether to run the model in float32 instead of bfloat16.
        host: The policy server's host.
        port: The policy server's port.
        bench: The bench, a name in BENCHES.
        episode: The episode's index in the bench.
        seconds: The time limit; None: the episode's own.
        decode_video: Whether to decode and save the predicted video.
        seed: The episode's noise seed; None: random.
    """
    policy = load_policy(model_name, full_precision, host, port, decode_video)
    out_path = f"outputs/molmospaces_{model_name}_{bench}_ep{episode}/cosmos3"
    print(f"{out_path}: {run_episode(policy, (bench, episode), out_path, seconds, seed)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_model_args(parser)
    parser.add_argument("--bench", choices=list(BENCHES), default="PickAndPlace-v1")
    parser.add_argument(
        "--episode", type=int, default=0, help="episode: 0-999 (Close-v1: 0-914)"
    )
    add_episode_args(parser)
    main(**vars(parser.parse_args()))
