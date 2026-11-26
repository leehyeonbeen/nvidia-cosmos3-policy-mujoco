"""Decode the predicted videos of saved episodes into <out>_prediction.mp4, as a run with --decode-video saves them.
Runs in the policy server's env.

    python scripts/decode_video_latents.py outputs/molmospaces/edge-droid-policy/*/ep*/cosmos3.parquet
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])  # repository root
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before importing torch

import torch
from diffusers import AutoencoderKLWan

from models.cosmos3_policy.pipeline import decode_video_latent, use_full_precision
from utils.episode_log import load_model_calls, save_preview

DEVICE = "cuda" if torch.cuda.is_available() else "mps"


def load_decoder(path, dtype):
    """Loads the model's video decoder (VAE), without the rest of the model.

    Args:
        path: The model folder; its vae subfolder is loaded.
        dtype: The decoder's torch dtype; float32 also turns on full-precision GPU math.

    Returns:
        The VAE, on the GPU.
    """
    if dtype == torch.float32:
        use_full_precision()
    return AutoencoderKLWan.from_pretrained(path, subfolder="vae", torch_dtype=dtype).to(DEVICE)


def decode_video_latents(parquet_path, decoders, overwrite):
    """Decodes the predicted video of one saved episode.

    Args:
        parquet_path: The episode's .parquet. Its .json is read, and its _prediction.mp4 written, next to it.
        decoders: Loaded decoders by (model path, dtype), shared across episodes.
        overwrite: Whether to decode again if _prediction.mp4 exists.
    """
    out = parquet_path.removesuffix(".parquet")
    if os.path.exists(f"{out}_prediction.mp4") and not overwrite:
        print(f"{out}: decoded already, skipped")
        return
    with open(out + ".json") as f:
        info = json.load(f)
    key = info["model_config"]["path"], getattr(torch, info["dtype"])
    if key not in decoders:
        decoders[key] = load_decoder(*key)
    latents = load_model_calls(parquet_path, ["output_video_latent"])["output_video_latent"]
    t0 = time.perf_counter()
    frames = []
    for latent in latents:
        video = decode_video_latent(decoders[key], torch.from_numpy(latent)[None].to(DEVICE))
        frames += video[1:] if frames else video  # a call's frame 0 is the same moment as the last call's final frame
    save_preview(f"{out}_prediction.mp4", frames[: info["frames"]], info["model_config"]["fps"])  # drop unplayed
    print(f"Decoded {len(latents)} model calls in {time.perf_counter() - t0:.2f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("parquet", nargs="+", help="episode .parquet files")
    parser.add_argument("--overwrite", action="store_true", help="also remake existing _prediction.mp4")
    args = parser.parse_args()
    decoders = {}  # one per model and dtype
    for parquet_path in args.parquet:
        decode_video_latents(parquet_path, decoders, args.overwrite)
