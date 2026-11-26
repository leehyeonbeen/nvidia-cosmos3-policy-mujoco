"""MolmoSpaces' benches, and the camera mirror for the DROID models' second exterior camera."""

import numpy as np
from scipy.spatial.transform import Rotation

# rad: the Robotiq 2F-85's driver joint fully closed (MolmoSpaces' pi_policy.py:127)
GRIPPER_CLOSED_ANGLE = 0.824033


# Folders under $MLSPACES_ASSETS_DIR/benchmarks, with the names on MolmoSpaces' leaderboard
# (ms-bench.md, mb-bench.md). HARD: the v2 benches.
HARD = "molmospaces-bench-v2/procthor-objaverse"
BENCHES = {
    "Pick-v1": "molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231",
    "PickAndPlace-v1": "molmospaces-bench-v1/procthor-10k/FrankaPickandPlaceDroidMiniBench/FrankaPickandPlaceDroidMiniBench_20260111_json_benchmark",
    "Open-v1": "molmospaces-bench-v1/ithor/FrankaOpenDataGenConfig/FrankaOpenDataGenConfig_20260123_json_benchmark",
    "Close-v1": "molmospaces-bench-v1/ithor/FrankaCloseDataGenConfig/FrankaCloseDataGenConfig_20260123_json_benchmark",
    "Pick-v1.5": "molmospaces-bench-v2/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231",
    "Pick-v2-classic": f"{HARD}/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark",
    "Pick-v2-filament": f"{HARD}/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark",
    "PickAndPlace-v2": f"{HARD}/FrankaPickandPlaceHardBench/FrankaPickandPlaceHardBench_20260206_json_benchmark",
    "PickAndPlace-NextTo-v2": f"{HARD}/FrankaPickandPlaceNextToHardBench/FrankaPickandPlaceNextToHardBench_20260305_json_benchmark",
    "PickAndPlace-Color-v2": f"{HARD}/FrankaPickandPlaceColorHardBench/FrankaPickandPlaceColorHardBench_20260304_json_benchmark",
}


def mirrored_camera(offset, quat):
    """Returns a camera's pose mirrored to the robot's other side, for the DROID models' second exterior camera.

    Args:
        offset: The camera's position in the robot base frame.
        quat: The camera's orientation (w x y z), MuJoCo's camera frame.

    Returns:
        (offset, quat) of the mirrored camera, as lists.
    """
    x, y, z = offset
    turn = Rotation.from_quat(quat, scalar_first=True).as_matrix()
    turn = np.diag([1, -1, 1]) @ turn @ np.diag([-1, 1, 1])  # mirror y -> -y, then flip the camera's x to keep a rotation
    return [x, -y, z], Rotation.from_matrix(turn).as_quat(scalar_first=True).tolist()
