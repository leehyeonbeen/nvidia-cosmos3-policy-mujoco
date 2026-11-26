"""Command-line options and the episode list shared by the run scripts."""

import json
import os

POLICY_MODELS = ["edge-droid-policy", "nano-droid-policy"]  # the models that drive the robot in MolmoSpaces
MODELS = POLICY_MODELS + ["edge-base-policy", "nano-base-policy"]  # base models: recordings only


def add_model_args(parser):
    """Adds the options for the model and the policy server.

    Args:
        parser: The argparse parser.
    """
    parser.add_argument(
        "--model", dest="model_name", choices=POLICY_MODELS, default="edge-droid-policy"
    )
    parser.add_argument(
        "--host", default="localhost", help="scripts/policy_server.py's host"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--full-precision",
        action="store_true",
        help="float32: the same numbers on every GPU, but slower (default: bfloat16)",
    )


def add_episode_args(parser):
    """Adds the options for each episode: time limit, video decoding and noise seed.

    Args:
        parser: The argparse parser.
    """
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="time limit in sim time (default: the benchmark's)",
    )
    parser.add_argument(
        "--decode-video",
        action="store_true",
        help="save the predicted video (slow)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="every episode's noise seed (default: run_demo random, run_all each episode's own)",
    )


def add_bulk_args(parser, default_dir):
    """Adds the options for runs over many episodes.

    Args:
        parser: The argparse parser.
        default_dir: The default output folder, as --output-dir's help shows it.
    """
    parser.add_argument("--chunk", type=int, default=0, help="which part of the episodes to run, from 0")
    parser.add_argument("--num-chunks", type=int, default=1, help="how many parts the episodes are split into")
    parser.add_argument("--output-dir", default=None, help=f"default: {default_dir}")


def episodes_to_run(episodes, json_path, output_dir, chunk, num_chunks):
    """Returns this chunk's episodes, leaving out those already saved in output_dir.

    Args:
        episodes: All the episodes.
        json_path: json_path(episode) is the episode's .json path inside output_dir.
        output_dir: The run's output folder.
        chunk: This chunk's index, from 0.
        num_chunks: How many chunks the episodes are split into.

    Returns:
        A list of the episodes to run.
    """
    n = len(episodes)
    episodes = episodes[chunk * n // num_chunks : (chunk + 1) * n // num_chunks]
    todo = [e for e in episodes if not os.path.exists(f"{output_dir}/{json_path(e)}")]
    print(f"chunk {chunk} of {num_chunks}: {len(episodes)} episodes, {len(todo)} to run", flush=True)
    return todo


def write_json(path, **info):
    """Writes info to a .json file, making its folder.

    Args:
        path: The .json file's path.
        **info: The entries.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=1)


def skip(path, **info):
    """Returns the reason, for run_all.py to print.

    Args:
        path: The episode's .json path; info is written there.
        **info: The .json's entries; not_run gives the reason.

    Returns:
        "skipped, <not_run>".
    """
    write_json(path, **info)
    return f"skipped, {info['not_run']}"
