"""Shared utilities for run_eval.py and run_demo.py (WebSocket policy scripts).
Adapted from robocasa for LIBERO DROID support."""

from typing import Dict

import numpy as np

# ---------------------------------------------------------------------------
# Arm controller mapping:  CLI name  ->  robosuite controller type
# ---------------------------------------------------------------------------
ARM_CONTROLLER_MAP: Dict[str, str] = {
    "cartesian_pose": "OSC_POSE",
    "joint_pos": "JOINT_POSITION",
    "joint_vel": "JOINT_VELOCITY",
}

ARM_CONTROLLER_ACTION_DIMS: Dict[str, int] = {
    "cartesian_pose": 6,
    "joint_pos": 7,
    "joint_vel": 7,
}


def _patch_joint_vel_controller() -> None:
    """Workaround for robosuite <= 1.5.2 bug: JointVelocityController torque_compensation
    property collision. Patch applied at module load when JOINT_VELOCITY may be used.
    """
    try:
        from robosuite.controllers.parts.generic.joint_vel import JointVelocityController as JVC
        from robosuite.controllers.parts.controller import Controller
    except ImportError:
        return

    if getattr(JVC, "_patched_tc", False):
        return

    _orig_init = JVC.__init__

    def _new_init(self, *args, **kwargs):
        _prop = Controller.__dict__.get("torque_compensation")
        if _prop is not None:
            Controller.torque_compensation = property(_prop.fget, lambda self, v: None)
        try:
            _orig_init(self, *args, **kwargs)
        finally:
            if _prop is not None:
                Controller.torque_compensation = _prop
        self._use_torque_compensation = kwargs.get("use_torque_compensation", True)

    def _new_run(self):
        if self.goal_vel is None:
            self.set_goal(np.zeros(self.joint_dim))
        self.update()
        if self.interpolator is not None and self.interpolator.order == 1:
            self.current_vel = self.interpolator.get_interpolated_goal()
        else:
            self.current_vel = np.array(self.goal_vel)
        err = self.current_vel - self.joint_vel
        derr = err - self.last_err
        self.last_err = err
        self.derr_buf.push(derr)
        if not self.saturated:
            self.summed_err += err
        if self._use_torque_compensation:
            torques = (
                self.kp * err + self.ki * self.summed_err
                + self.kd * self.derr_buf.average + self.torque_compensation
            )
        else:
            torques = self.kp * err + self.ki * self.summed_err + self.kd * self.derr_buf.average
        self.torques = self.clip_torques(torques)
        self.saturated = np.any(self.torques != torques)
        super(JVC, self).run_controller()
        return self.torques

    JVC.__init__ = _new_init
    JVC.run_controller = _new_run
    JVC._patched_tc = True


_patch_joint_vel_controller()


def get_arm_action_dim(arm_controller: str) -> int:
    """Return the action dimension of the arm (excluding gripper)."""
    return ARM_CONTROLLER_ACTION_DIMS.get(arm_controller, 7)


DROID_IMG_SIZE = 224


def _ensure_uint8_hwc(
    img: np.ndarray, target_hw: tuple = (DROID_IMG_SIZE, DROID_IMG_SIZE)
) -> np.ndarray:
    """Convert image to uint8 HWC and resize to target if needed."""
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    h, w = target_hw
    if img.shape[0] != h or img.shape[1] != w:
        from PIL import Image
        img = np.array(Image.fromarray(img).resize((w, h), resample=Image.BICUBIC))
    return img


def _gripper_qpos_to_droid(gripper_qpos: np.ndarray) -> np.ndarray:
    """Map Panda robot0_gripper_qpos (2,) to DROID gripper_position (1,) in [0,1]. 0=open, 1=closed."""
    q = np.asarray(gripper_qpos).flatten()
    val = (q[0] + q[1]) / 2.0 if len(q) >= 2 else float(q[0])
    return np.array([np.clip(val / 0.04, 0.0, 1.0)], dtype=np.float64)


def prepare_observation_droid(
    obs: dict,
    task_description: str,
    flip_images: bool = True,
    img_size: int = DROID_IMG_SIZE,
) -> dict:
    """Convert raw LIBERO observation to OpenPI DROID format.

    LIBERO uses agentview_image (not robot0_agentview_left_image).
    Returns a dict with keys expected by DROID policy:
        observation/exterior_image_1_left, observation/wrist_image_left,
        observation/joint_position, observation/gripper_position, prompt
    """
    target_hw = (img_size, img_size)

    exterior = obs.get("agentview_image")
    wrist = obs.get("robot0_eye_in_hand_image")
    if flip_images:
        if exterior is not None:
            exterior = np.flipud(exterior).copy()
        if wrist is not None:
            wrist = np.flipud(wrist).copy()

    exterior = (
        _ensure_uint8_hwc(exterior, target_hw)
        if exterior is not None
        else np.zeros((*target_hw, 3), dtype=np.uint8)
    )
    wrist = (
        _ensure_uint8_hwc(wrist, target_hw)
        if wrist is not None
        else np.zeros((*target_hw, 3), dtype=np.uint8)
    )

    joint_pos = obs.get("robot0_joint_pos")
    if joint_pos is None:
        raise KeyError(
            "robot0_joint_pos not in observation. Enable it when creating env for --droid, "
            "e.g. env.modify_observable(observable_name='robot0_joint_pos', attribute='active', modifier=True)"
        )
    joint_pos = np.asarray(joint_pos, dtype=np.float64).flatten()
    if joint_pos.shape[0] != 7:
        raise ValueError(f"robot0_joint_pos must be 7D, got shape {joint_pos.shape}")

    gripper_qpos = obs["robot0_gripper_qpos"]
    gripper_pos = _gripper_qpos_to_droid(gripper_qpos)

    return {
        "observation/exterior_image_1_left": exterior,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": joint_pos,
        "observation/gripper_position": gripper_pos,
        "prompt": task_description,
    }


def enable_joint_pos_observable(env) -> None:
    """Enable robot0_joint_pos observable for DROID mode.

    LIBERO ControlEnv wraps env.env (robosuite). Access inner env if needed.
    """
    target = getattr(env, "env", env)
    for ob_name in getattr(target, "observation_names", []):
        if "joint_pos" in ob_name:
            target.modify_observable(
                observable_name=ob_name, attribute="active", modifier=True
            )
            break


def pad_action_for_env(
    action: np.ndarray,
    arm_controller: str,
    env_action_dim: int,
) -> np.ndarray:
    """Ensure action has correct shape for env. LIBERO Panda has no mobile base;
    policy output should match expected dim (e.g. 8 for joint_vel)."""
    expected_dim = get_arm_action_dim(arm_controller) + 1  # +1 gripper
    action = np.asarray(action, dtype=np.float64).flatten()
    if action.shape[0] == expected_dim and env_action_dim > expected_dim:
        pad_dim = env_action_dim - expected_dim
        padding = np.zeros(pad_dim, dtype=np.float64)
        padding[-1] = -1.0
        action = np.concatenate([action, padding])
    elif action.shape[0] != expected_dim and action.shape[0] <= env_action_dim:
        padding = np.zeros(env_action_dim - action.shape[0], dtype=np.float64)
        if padding.shape[0] > 0:
            padding[-1] = -1.0
        action = np.concatenate([action, padding])
    return np.array(action[:env_action_dim], dtype=np.float64, copy=True)
