"""Save an episode's model calls (EpisodeLog).

    <out>.parquet: one row per model call, its inputs and outputs:
        frame_index: the camera image the call got, a frame of _camera.mp4
        input_image, input_state: the model's input image, and joint angles + gripper (Policy-DROID only)
        output_action: the model's actions, before they are converted into robot commands
        output_video_latent: the predicted video, not decoded (scripts/decode_video_latents.py decodes it)
    <out>_camera.mp4: the camera images, one frame per action
    <out>_prediction.mp4: the predicted video, one frame per action (--decode-video)
    <out>.json: prompt, model settings and the outcome

bfloat16 values are saved as their raw bits (uint16): lossless, half the size.
"""

import json
import os
from dataclasses import asdict, dataclass

import imageio
import numpy as np
import pandas as pd

SETTLE_SECONDS = 3.0  # before the first camera image, in every episode


@dataclass
class Task:
    prompt: str
    object_name: str  # body of the object to pick
    receptacle_name: str | None = None  # body to place it on (pick-and-place only)
    episode: str = ""  # benchmark episode (e.g. "PickAndPlace-v1 ep0")


def transform(position, rotation):
    """Builds a 4x4 transform from a position and a rotation matrix.

    Args:
        position: The position x y z.
        rotation: The rotation matrix, 3x3 or its 9 values.

    Returns:
        The 4x4 transform.
    """
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.reshape(rotation, (3, 3)), position
    return T


class EpisodeLog:
    """Records an episode's camera images and model calls, and saves its files."""

    def __init__(self, cosmos, info):
        """Starts the log.

        Args:
            cosmos: The Cosmos3 model; its settings and dtype are logged.
            info: The episode's entries for the .json, e.g. the prompt and seed.
        """
        self.info = dict(info, settle_seconds=SETTLE_SECONDS, dtype=str(cosmos.dtype).removeprefix("torch."))
        self.info.setdefault("model_config", asdict(cosmos.cfg))
        self.camera_frames, self.prediction_frames, self.model_calls, self.call_frames = [], [], [], []

    def add_prediction(self, call, video, last=False):
        """Adds one model call's inputs and outputs, and its predicted video, before its camera image is added.

        Args:
            call: The call's inputs and outputs {name: array}.
            video: The predicted video frames; empty if not decoded.
            last: The call on the episode's last image. Its frame 0 ends the predicted video.
        """
        self.model_calls.append(call)
        self.call_frames.append(len(self.camera_frames))
        if not video:
            return
        played = self.prediction_frames
        if last:
            self.prediction_frames = played[: len(self.camera_frames)] + list(video[:1])
        else:  # a call's frame 0 is the last call's final frame
            self.prediction_frames = played + list(video[1:] if played else video)

    def add(self, image):
        """Adds a camera image, one per action.

        Args:
            image: The camera image the model got.
        """
        self.camera_frames.append(image)

    def save(self, out_path, fps, **info):
        """Saves the episode's files.

        Args:
            out_path: The files' path without the ending (.parquet, .json, _camera.mp4, _prediction.mp4).
            fps: The videos' frame rate.
            **info: Entries added to the .json.

        Returns:
            The .json contents.
        """
        self.info.update(info, frames=len(self.camera_frames))
        out_path = os.path.abspath(out_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_preview(out_path + "_camera.mp4", self.camera_frames, fps)
        if self.prediction_frames:  # drop the actions not played
            save_preview(out_path + "_prediction.mp4", self.prediction_frames[: len(self.camera_frames)], fps)
        if self.model_calls:
            self.info["model_call_shapes"] = {k: list(v.shape) for k, v in self.model_calls[0].items()}
            self.info["model_call_bfloat16"] = [k for k, v in self.model_calls[0].items() if v.dtype == np.uint16]
            table = pd.DataFrame(
                {k: [call[k].ravel() for call in self.model_calls] for k in self.model_calls[0]}
            )
            table.insert(0, "frame_index", self.call_frames)
            table.to_parquet(out_path + ".parquet")
            print(out_path + ".parquet")
        with open(out_path + ".json", "w") as f:
            json.dump(self.info, f, indent=1)
        print(out_path + ".json")
        return self.info


def save_preview(path, frames, fps):
    """Saves frames as an mp4 (H.264, not lossless), to watch.

    Args:
        path: The mp4's path.
        frames: The images.
        fps: The frame rate.
    """
    imageio.mimwrite(path, frames, fps=fps, macro_block_size=1)
    print(path)


def stored(x):
    """Returns a tensor as saved: bfloat16 as its raw bits (uint16), anything else as float32.

    Args:
        x: A torch tensor.

    Returns:
        A NumPy array, uint16 or float32.
    """
    import torch  # on the policy server only

    x = x.detach().cpu()
    if x.dtype == torch.bfloat16:
        return x.contiguous().view(torch.int16).numpy().view(np.uint16)
    return x.float().numpy()


def unstored(x, bfloat16):
    """Undoes stored(): raw bfloat16 bits -> the same values in float32.

    Args:
        x: A saved array.
        bfloat16: Whether x holds bfloat16 bits.

    Returns:
        x in float32 if it held bfloat16 bits, else x unchanged.
    """
    return (x.astype(np.uint32) << 16).view(np.float32) if bfloat16 else x


def load_model_calls(parquet_path, keys=None):
    """Loads the model calls saved in a .parquet, one array per input or output (bfloat16 values as float32).

    Args:
        parquet_path: The episode's .parquet; its .json gives the shapes.
        keys: Only these inputs and outputs. Defaults to all.

    Returns:
        {key: array}, stacked over the calls.
    """
    with open(parquet_path.removesuffix(".parquet") + ".json") as f:
        info = json.load(f)
    shapes, bfloat16 = info["model_call_shapes"], info["model_call_bfloat16"]
    keys = keys or list(shapes)
    table = pd.read_parquet(parquet_path, columns=keys)
    return {
        k: np.stack([unstored(v, k in bfloat16).reshape(shapes[k]) for v in table[k]])
        for k in keys
    }
