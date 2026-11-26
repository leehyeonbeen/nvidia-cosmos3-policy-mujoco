"""Cosmos3 as a MolmoSpaces policy on MolmoSpaces' DROID Franka, with its eval config.
Follows molmo_spaces/evaluation/README.md:117-205. run_demo.py loads it.
"""

import os

import numpy as np
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.data_generation import pipeline
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig
from molmo_spaces.evaluation.robot_eval_overrides import register_robot_override
from molmo_spaces.policy.base_policy import InferencePolicy, PolicyFactory
from molmo_spaces.utils import grasps

from models.cosmos3_policy.client import load_model
from models.cosmos3_policy.specs import CONFIGS
from utils.episode_log import SETTLE_SECONDS, Task
from scripts.molmospaces.load_scene import GRIPPER_CLOSED_ANGLE, mirrored_camera
from utils.policy import ChunkPolicy

# Without this, Open-v1 and Close-v1 episodes fail to load: https://github.com/allenai/molmospaces/issues/218
_get_joint_grasp_path = grasps.get_joint_grasp_path
grasps.get_joint_grasp_path = (
    lambda uid, joint_name, grasp_libraries=None: _get_joint_grasp_path(
        uid, joint_name, grasp_libraries or ["droid"]
    )
)


# Follows MolmoSpaces' Pi policy (molmo_spaces/policy/learned_policy/pi_policy.py)
class Cosmos3MolmoSpacesPolicy(InferencePolicy):
    def __init__(
        self,
        config,
        task=None,
        decode_video=False,
        full_precision=False,
        host="localhost",
        port=8000,
    ):
        """Sets up the policy and loads the model.

        Args:
            config: The eval config.
            task: MolmoSpaces' task, or None.
            decode_video: Whether to decode and save the predicted video.
            full_precision: Whether to run the model in float32 instead of bfloat16.
            host: The policy server's host. Defaults to "localhost".
            port: The policy server's port. Defaults to 8000.
        """
        super().__init__(config, task)
        self.decode_video, self.full_precision = decode_video, full_precision
        self.host, self.port = host, port
        self.seed = None  # the episode's noise seed, set by run_episode before MolmoSpaces resets the policy
        self.prepare_model()
        self.reset()

    def prepare_model(self):
        """Connects to the model on the policy server: the one whose checkpoint the eval config names."""
        path = os.path.normpath(self.config.policy_config.checkpoint_path)
        model_name = next(name for name, cfg in CONFIGS.items() if cfg.path == path)
        self.cosmos = load_model(model_name, self.full_precision, self.host, self.port)
        self.chunks = ChunkPolicy(self.cosmos, self.decode_video)

    def reset(self):
        """Starts an episode: new noise from self.seed, the settle, and the task's prompt and objects."""
        self.chunks.reset(self.seed)
        self.settle_steps = round(SETTLE_SECONDS * 1000 / self.config.policy_dt_ms)
        self.hold = None
        self.success_time = None
        if self.task is None:
            return
        task = self.task.config.task_config
        receptacle = getattr(task, "place_receptacle_name", None)  # pick-and-place only
        self.goal = Task(
            self.task.get_task_description(), task.pickup_obj_name, receptacle
        )

    def end_episode(self):
        """Logs the episode's last image, with one more model call on it that is never played."""
        if (
            self.task is None or self.chunks.log is None
        ):  # not set up, or ended during the settle
            return
        views, fov, state = self.obs_to_model_input(self.task.get_observations()[0])
        self.chunks.act(views, self.goal.prompt, state, fov, last=True)

    def get_action(self, observation):
        """Returns the next action: the hold during the settle, then the model's; None ends the episode.

        Args:
            observation: MolmoSpaces' observation, or a list whose first item is it.

        Returns:
            {"arm": joint targets, "gripper": 0 or 255}, or None after a success's model call has played out.
        """
        obs = observation[0] if isinstance(observation, list) else observation
        if self.success_time is None and self.task.judge_success():
            if self.chunks.log is None:  # succeeded during the settle
                return None
            self.success_time = round(
                self.task.env.current_data.time - self.start_time, 6
            )
        if (
            self.success_time is not None and not self.chunks.actions
        ):  # after a success, finish the model call
            return None
        if self.settle_steps:  # settle: the arm held still, gripper open
            self.settle_steps -= 1
            if self.hold is None:
                self.hold = {
                    "arm": np.array(obs["qpos"]["arm"]),
                    "gripper": np.array([0.0]),
                }
            return self.hold
        return super().get_action(obs)

    def obs_to_model_input(self, obs):
        """Returns the model's inputs from an observation.

        Args:
            obs: MolmoSpaces' observation.

        Returns:
            (views, fov, state): the camera images, the wrist camera's vertical field of view (deg), and the
            joint angles + gripper closedness.
        """
        cameras = self.config.policy_config.camera_names
        views = [obs[camera] for camera in cameras]
        fov = self.task.env.camera_manager.registry[cameras[0]].fov
        gripper = np.clip(obs["qpos"]["gripper"][0] / GRIPPER_CLOSED_ANGLE, 0, 1)
        return views, fov, np.append(obs["qpos"]["arm"], gripper)

    def inference_model(self, model_input):
        """Returns the next joint targets and gripper closedness, calling the model when its last chunk is used up.

        Args:
            model_input: (views, fov, state) from obs_to_model_input.

        Returns:
            (q, g): the joint targets and the gripper closedness.
        """
        views, fov, state = model_input
        if self.chunks.log is None:  # the first image after the settle
            self.start_time = self.task.env.current_data.time
            self.chunks.start_log(self.goal, self.config.policy_config.camera_names)
        return self.chunks.act(views, self.goal.prompt, state, fov)

    def model_output_to_action(self, model_output):
        """Returns MolmoSpaces' action from the joint targets and gripper closedness.

        Args:
            model_output: (q, g) from inference_model.

        Returns:
            {"arm": joint targets, "gripper": 0 open or 255 closed}.
        """
        q, g = model_output  # gripper fully open or closed only (pi_policy.py:171)
        return {"arm": np.asarray(q), "gripper": np.array([255.0 if g > 0.5 else 0.0])}

    def save(self, out_path, seconds, success, **info):
        """Saves the episode's videos, .parquet and .json.

        Args:
            out_path: The saved files' path without the ending.
            seconds: The episode's time limit, for the .json.
            success: Whether the episode succeeded, for the .json.
            **info: Extra entries for the .json.
        """
        log = self.chunks.log
        log.save(
            out_path,
            self.cosmos.cfg.fps,
            seconds=seconds,
            success=success,
            success_time=self.success_time,
            **info,
        )


# MolmoSpaces cleans up after each episode while the scene is still open (pipeline.py:1133).
# Our policy logs the episode's last image first.
_cleanup_episode_resources = pipeline.cleanup_episode_resources


def cleanup_after_end_episode(task, policy, *args, **kwargs):
    """Lets our policy log the episode's last image, then runs MolmoSpaces' cleanup with the same arguments."""
    if isinstance(policy, Cosmos3MolmoSpacesPolicy):
        policy.end_episode()
    _cleanup_episode_resources(task, policy, *args, **kwargs)


pipeline.cleanup_episode_resources = cleanup_after_end_episode


# Follows MyPolicyConfig (molmo_spaces/evaluation/README.md:167)
class Cosmos3PolicyConfig(BasePolicyConfig):
    policy_type: str = "learned"
    policy_cls: type = None
    policy_factory: PolicyFactory | None = None

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.policy_cls is None:
            self.policy_cls = Cosmos3MolmoSpacesPolicy
            self.policy_factory = Cosmos3MolmoSpacesPolicy

    checkpoint_path: str = "models/cosmos3_checkpoints/Cosmos3-Edge-Policy-DROID"
    camera_names: list[str] = ["wrist_camera", "exo_camera_1", "exo_camera_2"]


# MolmoSpaces' DROID Franka with a second exterior camera for the DROID models:
# the episode's exterior camera mirrored to the robot's other side.
# The robot overrides below follow cap_robot_eval_override (robot_eval_overrides.py:16).
class DroidFrankaRobotConfig(FrankaRobotConfig):
    pass


def second_exterior_camera(episode_spec, exp_config):
    """Adds the episode's exterior camera mirrored to the robot's other side, under the policy's third camera name.

    Args:
        episode_spec: The episode (unused).
        exp_config: The eval config; its camera list gets the mirrored camera.
    """
    wrist, exterior, second = exp_config.policy_config.camera_names
    exo = next(c for c in exp_config.camera_config.cameras if c.name == exterior)
    offset, quat = mirrored_camera(exo.camera_offset, exo.camera_quaternion)
    exp_config.camera_config.cameras.append(
        exo.model_copy(update=dict(name=second, camera_offset=offset, camera_quaternion=quat))
    )


register_robot_override(DroidFrankaRobotConfig, second_exterior_camera)


# A Policy-DROID model on MolmoSpaces' DROID Franka.
# Follows PiPolicyEvalConfig (molmo_spaces/evaluation/configs/evaluation_configs.py:191).
class Cosmos3EvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaRobotConfig = DroidFrankaRobotConfig()
    policy_config: Cosmos3PolicyConfig = Cosmos3PolicyConfig()
    policy_dt_ms: float = 66.0  # ~15 Hz, the Policy-DROID models' action rate
    end_on_success: bool = False  # the policy ends the episode (get_action)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False
