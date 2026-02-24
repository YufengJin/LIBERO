"""Shared utilities for run_eval.py and run_demo.py (WebSocket policy scripts).
Client sends raw robosuite obs; policy server handles all remapping."""

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


def get_expected_policy_action_dim(arm_controller: str) -> int:
    """Return the policy output dimension expected by this arm controller.
    cartesian_pose -> 7 (6 pose + 1 gripper); joint_pos/joint_vel -> 8 (7 joints + 1 gripper).
    """
    return get_arm_action_dim(arm_controller) + 1


def enable_joint_pos_observable(env) -> None:
    """Enable robot0_joint_pos observable. Call once after env creation.
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
