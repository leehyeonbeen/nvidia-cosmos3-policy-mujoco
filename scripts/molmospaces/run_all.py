"""Run a Cosmos3 Policy-DROID model on every episode of MolmoSpaces' benchmarks through MolmoSpaces' own evaluation, with
MolmoSpaces' DROID Franka. Needs the policy server (README, "Run").
The Filament benches run only with MuJoCo's Filament build installed, the others only without it.
Finished episodes are skipped, so a stopped run can be restarted.

Quick test (one 3 s episode, saved to outputs/_test):
    python scripts/molmospaces/run_all.py --benches PickAndPlace-v1 --episode 239 --seconds 3 --output-dir outputs/_test/molmospaces
Full runs:
    python scripts/molmospaces/run_all.py
    python scripts/molmospaces/run_all.py --model nano-droid-policy

Saves outputs/molmospaces/<model>/<bench>/ep<i>/: MolmoSpaces' results (score them with its
scripts/benchmarks/eval_to_csv.py) and cosmos3*.
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # repository root
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before importing torch

from molmo_spaces.env.env import HAS_FILAMENT

from scripts.molmospaces.load_scene import BENCHES
from scripts.molmospaces.run_demo import bench_episodes, load_policy, run_episode
from utils.runner import (
    add_bulk_args,
    add_episode_args,
    add_model_args,
    episodes_to_run,
)

# Benches for MuJoCo's Filament renderer (mb-bench.md:46); MolmoSpaces renders with the installed one
FILAMENT = [
    "Pick-v2-filament",
    "PickAndPlace-v2",
    "PickAndPlace-NextTo-v2",
    "PickAndPlace-Color-v2",
]


def main(
    model_name,
    full_precision,
    host,
    port,
    benches,
    episode,
    seconds,
    chunk,
    num_chunks,
    output_dir,
    decode_video,
    seed,
):
    """Runs the chosen episodes of the chosen benches, skipping those already saved.

    Args:
        model_name: One of POLICY_MODELS in utils/runner.py.
        full_precision: Whether to run the model in float32 instead of bfloat16.
        host: The policy server's host.
        port: The policy server's port.
        benches: The benches to run, names in BENCHES.
        episode: Only these episode indices, or None for all.
        seconds: The time limit; None: each episode's own.
        chunk: This chunk's index, from 0.
        num_chunks: How many chunks the episodes are split into.
        output_dir: Where the episodes are saved; None: outputs/molmospaces/<model>.
        decode_video: Whether to decode and save the predicted video.
        seed: Every episode's noise seed; None: each episode's own, from its bench and index.
    """
    output_dir = output_dir or f"outputs/molmospaces/{model_name}"
    other = [b for b in benches if (b in FILAMENT) != HAS_FILAMENT]
    if other:
        renderer = "classic" if HAS_FILAMENT else "Filament"
        print(f"Skipped, they need the {renderer} renderer: {', '.join(other)}")
        benches = [b for b in benches if b not in other]
    episodes = []
    for b in benches:
        for i in range(len(bench_episodes(b))):
            if episode is None or i in episode:
                episodes.append((b, i))
    json_path = lambda ep: f"{ep[0]}/ep{ep[1]}/cosmos3.json"
    todo = episodes_to_run(episodes, json_path, output_dir, chunk, num_chunks)
    if not todo:
        return
    policy = load_policy(model_name, full_precision, host, port, decode_video)
    for ep in todo:
        out = f"{output_dir}/{ep[0]}/ep{ep[1]}"
        ep_seed = seed if seed is not None else 1000 * list(BENCHES).index(ep[0]) + ep[1]
        try:
            result = run_episode(policy, ep, f"{out}/cosmos3", seconds, ep_seed)
        except Exception:  # writes no .json, so the next run retries it
            traceback.print_exc()
            result = "error"
        print(f"--- {out}: {result}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_model_args(parser)
    parser.add_argument(
        "--benches", nargs="+", choices=list(BENCHES), default=list(BENCHES)
    )
    parser.add_argument(
        "--episode",
        type=int,
        nargs="+",
        default=None,
        help="only these episodes: 0-999 (Close-v1: 0-914)",
    )
    add_episode_args(parser)
    add_bulk_args(parser, "outputs/molmospaces/<model>")
    main(**vars(parser.parse_args()))
