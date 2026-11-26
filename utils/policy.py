"""Run a Cosmos3 model as a policy: call the model, then play its actions one per step."""

import math
from dataclasses import asdict

from utils.episode_log import EpisodeLog


def whole_calls(actions, chunk):
    """Rounds up a time limit in actions so the last model call plays all its actions.

    Args:
        actions: The time limit in actions.
        chunk: Actions per model call.

    Returns:
        The rounded time limit, a whole number of chunks.
    """
    return math.ceil(actions / chunk) * chunk


def time_limit(seconds, dt, chunk):
    """Rounds a time limit up to whole model calls.

    Args:
        seconds: The time limit.
        dt: Seconds per action.
        chunk: Actions per model call.

    Returns:
        (actions, seconds): the rounded time limit in actions and in seconds.
    """
    actions = whole_calls(round(seconds / dt), chunk)
    return actions, round(actions * dt, 6)


class ChunkPolicy:
    def __init__(self, cosmos, decode_video=False):
        """Sets up the model `cosmos` as a policy.

        Args:
            cosmos: The Cosmos3 model.
            decode_video: Whether to decode and save the predicted video.
        """
        self.cosmos, self.decode_video = cosmos, decode_video

    def reset(self, seed=None):
        """Starts a new episode.

        Args:
            seed: The episode's noise seed; None: random.
        """
        self.actions, self.log, self.model_call = [], None, 0
        self.seed = seed
        self.cosmos.reset(seed)

    def start_log(self, task, cameras):
        """Starts the episode's log.

        Args:
            task: The episode's Task.
            cameras: The camera names, saved with the model settings.
        """
        info = dict(
            prompt=task.prompt,
            episode=task.episode,
            object_name=task.object_name,
            receptacle_name=task.receptacle_name,
            seed=self.seed,
            model_config=dict(asdict(self.cosmos.cfg), cameras=list(cameras)),
        )
        self.log = EpisodeLog(self.cosmos, info)

    def act(self, views, prompt, state, wrist_fovy=None, last=False):
        """Returns the next joint targets and gripper closedness (0 open, 1 closed) from the raw camera images.

        Calls the model again when the last call's actions are used up.

        Args:
            views: The raw camera images.
            prompt: The task's prompt.
            state: The robot state; its last value is the gripper's.
            wrist_fovy: The wrist camera's field of view (fovy), or None.
            last: The episode has ended. Logs its last image and one more model call on it, which is never played.

        Returns:
            (q, g): the joint targets and the gripper closedness; None when last is set.
        """
        image = self.cosmos.compose(views, wrist_fovy)  # the model's image, logged
        if not self.actions or last:
            joints, gripper, call, video = self.cosmos.predict(views, prompt, state, wrist_fovy, self.decode_video)
            self.log.add_prediction(call, video, last)
            self.actions = list(zip(joints, gripper))
            self.model_call += 1
            if not last:
                print(
                    f"Playing {len(self.actions)} of {len(call['output_action'])} actions "
                    f"({len(self.actions) / self.cosmos.cfg.fps:.2f}s)."
                )
        self.log.add(image)
        if last:
            self.log.info["unplayed_model_call"] = self.model_call - 1
            return
        return self.actions.pop(0)
