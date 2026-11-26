"""Serve the Cosmos3 models to the simulations (models/cosmos3_policy/client.py). One model at a time: a request
for another model replaces it. Needs the cosmos3-policy-server env (README, "Setup").

Start (ws://localhost:8000):
    python scripts/policy_server.py
Quick test (3 s), from a benchmark env in a second terminal:
    python scripts/molmospaces/run_demo.py --seconds 3
On another port (give the simulations the same --port):
    python scripts/policy_server.py --port 9000

Saves nothing: the simulations save the outputs.
"""

import argparse
import gc
import os
import sys
import traceback
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])  # repository root
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before importing torch

import torch
import websockets.sync.server

import numpy as np

from models.cosmos3_policy.msgpack_numpy import packb, unpackb

LOADED = {}  # the one model in memory: {(model, full_precision): Cosmos3Model}


def get_model(model_name, full_precision):
    """Returns the model, loading it in place of the one in memory when needed.

    Args:
        model_name: One of POLICY_MODELS in utils/runner.py.
        full_precision: float32 instead of bfloat16; a server runs one precision.

    Returns:
        The Cosmos3Model.
    """
    key = (model_name, full_precision)
    if key not in LOADED:
        # float32 changes torch's global math settings (use_full_precision), so a server runs one precision
        assert all(
            fp == full_precision for _, fp in LOADED
        ), "one precision per server: restart it for the other"
        LOADED.clear()  # one model at a time: free the last one first
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            torch.mps.empty_cache()
        from models.cosmos3_policy.model import Cosmos3Model

        LOADED[key] = Cosmos3Model(model_name, torch.float32 if full_precision else torch.bfloat16)
    return LOADED[key]


def model_image(cosmos, req):
    """Returns the model's input image, built from a request's raw camera images.

    Args:
        cosmos: The Cosmos3Model.
        req: The request, with the camera images under openpi's keys (specs.CAMERA_KEYS) and wrist_fovy.

    Returns:
        The model's input image.
    """
    from models.cosmos3_policy.canvas import compose
    from models.cosmos3_policy.specs import CAMERA_KEYS

    return compose([req[k] for k in CAMERA_KEYS], cosmos.cfg.size, req["wrist_fovy"])


def answer(req):
    """Replies to one request.

    Args:
        req: A request; req["cmd"] is "load", "reset", "compose" or "infer". It also names the model and
            full_precision.

    Returns:
        The reply dict: dtype ("load"), empty ("reset"), image ("compose"), or joints, grip, call and video
        ("infer").
    """
    cosmos = get_model(req["model"], req["full_precision"])
    if req["cmd"] == "load":
        return dict(dtype=str(cosmos.dtype).replace("torch.", ""))
    if req["cmd"] == "reset":
        cosmos.reset(req["seed"])
        return {}
    if req["cmd"] == "compose":
        return dict(image=model_image(cosmos, req))
    joints, gripper, call, video = cosmos.infer(
        model_image(cosmos, req),
        req["prompt"],
        np.append(req["observation/joint_position"], req["observation/gripper_position"]),
        req["decode_video"],
    )
    return dict(joints=joints, grip=gripper, call=call, video=video)


def handle(ws):
    """Answers one client's requests until it disconnects; errors go back as text.

    Args:
        ws: The client's websocket connection.
    """
    ws.send(packb({}))  # openpi's protocol starts with a metadata message
    try:
        for message in ws:
            try:
                reply = packb(answer(unpackb(message)))
            except Exception:
                traceback.print_exc()
                reply = traceback.format_exc()  # sent as text: the client raises it
            ws.send(reply)
    except websockets.ConnectionClosed:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="0.0.0.0", help="address to listen on; 0.0.0.0: every network interface")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    with websockets.sync.server.serve(
        handle, args.host, args.port, compression=None, max_size=None
    ) as server:
        print(f"Cosmos3 policy server on ws://{args.host}:{args.port}", flush=True)
        server.serve_forever()
