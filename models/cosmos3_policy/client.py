"""Call a Cosmos3 Policy-DROID model on our policy server (scripts/policy_server.py) from a simulation."""

import time

import websockets.sync.client

from models.cosmos3_policy.msgpack_numpy import packb, unpackb
from models.cosmos3_policy.specs import CAMERA_KEYS, CONFIGS


def connect(url, ping_interval=20):
    """Connects to a policy server in openpi's protocol and skips its first message, the metadata.

    Args:
        url: The server's address, ws://<host>:<port>.
        ping_interval: Seconds between keepalive pings; None for none. Defaults to 20, the websockets default.

    Returns:
        The websocket connection.
    """
    ws = websockets.sync.client.connect(url, compression=None, max_size=None, ping_interval=ping_interval)
    unpackb(ws.recv())
    return ws


def send_request(ws, message):
    """Sends one message to a policy server and returns its reply.

    Args:
        ws: The connection from connect.
        message: The request, a dict that may hold NumPy arrays.

    Returns:
        The server's reply, with NumPy arrays unpacked.

    Raises:
        RuntimeError: If the server replies with an error.
    """
    ws.send(packb(message))
    reply = ws.recv()
    if isinstance(reply, str):  # the servers send errors as text
        raise RuntimeError(f"policy server error:\n{reply}")
    return unpackb(reply)


def load_model(model_name, full_precision=False, host="localhost", port=8000):
    """Connects to the model on the policy server.

    Args:
        model_name: One of POLICY_MODELS in utils/runner.py.
        full_precision: float32 instead of bfloat16 (load_pipeline in models/cosmos3_policy/pipeline.py).
            Defaults to False.
        host: The policy server's host. Defaults to "localhost".
        port: The policy server's port. Defaults to 8000.

    Returns:
        A Cosmos3Client.
    """
    return Cosmos3Client(f"ws://{host}:{port}", model_name, full_precision)


class Cosmos3Client:
    def __init__(self, url, model_name, full_precision=False):
        """Connects to the policy server and loads the model there.

        Args:
            url: The policy server's address, ws://<host>:<port>.
            model_name: One of POLICY_MODELS in utils/runner.py.
            full_precision: float32 instead of bfloat16, sent with every request. Defaults to False.
        """
        self.cfg = CONFIGS[model_name]
        self.key = dict(model=model_name, full_precision=full_precision)
        self.ws = connect(url)
        self.dtype = self.request("load")["dtype"]

    def request(self, cmd, **kwargs):
        """Sends one request to the policy server and returns its reply.

        Args:
            cmd: "load", "reset", "compose" or "infer" (scripts/policy_server.py).
            **kwargs: The request's other fields.

        Returns:
            The server's reply, a dict, with NumPy arrays unpacked.

        Raises:
            RuntimeError: If the server replies with an error.
        """
        return send_request(self.ws, dict(cmd=cmd, **self.key, **kwargs))

    def reset(self, seed=None):
        """Starts an episode's noise from seed on the server.

        Args:
            seed: The episode's noise seed; None: random.
        """
        self.request("reset", seed=seed)

    def compose(self, views, wrist_fovy=None):
        """Returns the model's input image, built by the server from the raw camera images.

        Args:
            views: Raw camera images.
            wrist_fovy: The wrist camera's vertical field of view (deg).

        Returns:
            The model's input image.
        """
        return self.request("compose", **camera_images(views, wrist_fovy))["image"]

    def predict(self, views, prompt, state, wrist_fovy=None, decode_video=False):
        """Runs one model call on the policy server.

        Args:
            views: Raw camera images.
            prompt: The text prompt.
            state: Joint angles + gripper closedness.
            wrist_fovy: The wrist camera's vertical field of view (deg).
            decode_video: Whether to decode the predicted video. Defaults to False.

        Returns:
            Joint targets, gripper closedness, the call's inputs and outputs, and the decoded video frames
            (empty if not decoded).
        """
        t1 = time.perf_counter()
        obs = {
            **camera_images(views, wrist_fovy),
            "prompt": prompt,
            "observation/joint_position": state[:-1],
            "observation/gripper_position": state[-1:],
        }
        out = self.request("infer", **obs, decode_video=decode_video)
        print(f"Policy server replied in {time.perf_counter() - t1:.2f}s.")
        return out["joints"], out["grip"], out["call"], out["video"]


def camera_images(views, wrist_fovy=None):
    """Returns the camera images under openpi's observation keys, for a policy server request.

    Args:
        views: Raw camera images, in Config.cameras order.
        wrist_fovy: The wrist camera's vertical field of view (deg), sent along.

    Returns:
        A dict of the images by key, plus "wrist_fovy".
    """
    return {**dict(zip(CAMERA_KEYS, views)), "wrist_fovy": wrist_fovy}
