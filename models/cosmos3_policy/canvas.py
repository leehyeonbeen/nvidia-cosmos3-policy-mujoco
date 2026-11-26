"""Build the model's input image from raw camera images, in the layout of the model's training data:

the DROID layout: wrist camera on top, two side cameras below, 640x540
"""

import numpy as np
import torch
import torch.nn.functional as F


# Copied from NVIDIA's policy server:
# https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/cosmos_framework/scripts/action_policy_server_robolab.py#L226
def _resize_rgb_uint8(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()  # [1,3,H,W]
    resized = F.interpolate(
        tensor, size=size, mode="bilinear", align_corners=False
    )  # [1,3,H2,W2]
    return resized.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)  # [H2,W2,3]


# The DROID Franka's wrist camera in MolmoSpaces: vertical field of view (deg) and aspect ratio
DROID_WRIST_FOVY = 56.74
DROID_WRIST_ASPECT = 16 / 9


def crop_to_droid_wrist(image, fovy):
    """Crops a wider wrist camera image to the DROID wrist camera's view, which the DROID models were trained on.

    Args:
        image: The wrist camera image.
        fovy: The camera's vertical field of view (deg).

    Returns:
        The centered crop, with the DROID wrist camera's field of view and 16:9 aspect ratio, or the image
        itself if its view is not wider.
    """
    if fovy <= DROID_WRIST_FOVY:
        return image
    h, w = image.shape[:2]
    crop_h = round(
        h * np.tan(np.radians(DROID_WRIST_FOVY / 2)) / np.tan(np.radians(fovy / 2))
    )
    crop_w = round(crop_h * DROID_WRIST_ASPECT)
    assert (
        crop_w <= w
    ), f"a {w}x{h} image with a {fovy} deg field of view is too narrow for the DROID view"
    top, left = (h - crop_h) // 2, (w - crop_w) // 2
    return np.ascontiguousarray(image[top : top + crop_h, left : left + crop_w])


# Copied from NVIDIA's policy server (the end of _compose_roboarena_views):
# https://github.com/NVIDIA/cosmos-framework/blob/bf492fe/cosmos_framework/scripts/action_policy_server_robolab.py#L232
def compose_droid(
    wrist: np.ndarray, left_raw: np.ndarray, right_raw: np.ndarray
) -> np.ndarray:
    half_h, half_w = wrist.shape[0] // 2, wrist.shape[1] // 2
    left = _resize_rgb_uint8(left_raw, (half_h, half_w))
    right = _resize_rgb_uint8(right_raw, (half_h, half_w))
    return np.concatenate([wrist, np.concatenate([left, right], axis=1)], axis=0)


def compose(views, size, wrist_fovy=None):
    """Builds the model's input image from raw camera images, in the config's camera order.

    Args:
        views: Raw camera images, in the config's camera order.
        size: The (width, height) of each camera image.
        wrist_fovy: The wrist camera's vertical field of view (deg).

    Returns:
        The model's input image.
    """
    views = [crop_to_droid_wrist(views[0], wrist_fovy), *views[1:]]
    resized = []
    for v in views:
        v = np.ascontiguousarray(v)
        resized.append(v if v.shape[:2] == size[::-1] else _resize_rgb_uint8(v, size[::-1]))
    return compose_droid(*resized)
