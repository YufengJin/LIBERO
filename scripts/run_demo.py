#!/usr/bin/env python3
"""
run_demo.py -- LIBERO demo client (WebSocket).

Runs a LIBERO task in simulation, delegates action inference to a Policy Server over WebSocket.
For demo only; no eval logs or success-rate tracking. Default: headless, saves videos to demo_log/.

Usage:
    python tests/test_random_policy_server.py --port 8000
    python scripts/run_demo.py --task_suite_name libero_10 --policy_server_addr localhost:8000
    python scripts/run_demo.py --gui --task_suite_name libero_10 --policy_server_addr localhost:8000
    python scripts/run_demo.py --droid --policy_server_addr localhost:8000 --task_suite_name libero_10
    python scripts/run_demo.py --arm_controller joint_pos --policy_server_addr localhost:8000 --task_suite_name libero_10
"""

import argparse
import atexit
import os
import signal
import sys
import time
from datetime import datetime

import imageio
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.env_wrapper import ControlEnv
from libero.libero.utils.run_utils import (
    ARM_CONTROLLER_MAP,
    enable_joint_pos_observable,
    pad_action_for_env,
    prepare_observation_droid,
)
from policy_websocket import WebsocketClientPolicy

TASK_SUITE_CHOICES = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
]

TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

DUMMY_ACTION_OSC = [0, 0, 0, 0, 0, 0, -1]
DUMMY_ACTION_JOINT_VEL = [0, 0, 0, 0, 0, 0, 0, -1]


def _get_env_action_dim(env):
    """Get action dimension from env (handles ControlEnv wrapper)."""
    inner = getattr(env, "env", env)
    action_spec = getattr(inner, "action_spec", None)
    if action_spec is not None and len(action_spec) >= 1:
        return action_spec[0].shape[0]
    action_space = getattr(inner, "action_space", None)
    if action_space is not None and hasattr(action_space, "shape"):
        return int(action_space.shape[0])
    return 8


def _create_env(task, img_res, controller="OSC_POSE", use_gui=False):
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_kwargs = dict(
        bddl_file_name=task_bddl_file,
        controller=controller,
        camera_heights=img_res,
        camera_widths=img_res,
    )
    if use_gui:
        env = ControlEnv(
            **env_kwargs,
            has_renderer=True,
            has_offscreen_renderer=True,
            render_camera="agentview",
        )
    else:
        env = OffScreenRenderEnv(**env_kwargs)
    env.seed(0)
    return env


def _prepare_observation(obs, task_description, flip_images=True):
    img = obs["agentview_image"]
    wrist_img = obs["robot0_eye_in_hand_image"]
    if flip_images:
        img = np.flipud(img)
        wrist_img = np.flipud(wrist_img)
    return {
        "primary_image": img,
        "wrist_image": wrist_img,
        "task_description": task_description,
    }


def save_rollout_video(primary_images, wrist_images, episode_idx, success,
                      task_description, output_dir):
    """Save a concatenated MP4 of primary | wrist camera views."""
    os.makedirs(output_dir, exist_ok=True)
    tag = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:40]
    filename = f"episode={episode_idx}--success={success}--task={tag}.mp4"
    mp4_path = os.path.join(output_dir, filename)
    writer = imageio.get_writer(mp4_path, fps=30, format="FFMPEG", codec="libx264")
    for p, w in zip(primary_images, wrist_images):
        frame = np.concatenate([p, w], axis=1)
        writer.append_data(frame)
    writer.close()
    print(f"Saved rollout video: {mp4_path}")
    return mp4_path


def run_episode(args, env, task_description, policy, episode_idx, max_steps,
                use_gui=False, save_video=False):
    droid = getattr(args, "droid", False)
    arm_controller = getattr(args, "arm_controller", "cartesian_pose")
    dummy = DUMMY_ACTION_JOINT_VEL if arm_controller in ("joint_pos", "joint_vel") else DUMMY_ACTION_OSC

    NUM_WAIT_STEPS = 10
    for _ in range(NUM_WAIT_STEPS):
        obs, _, _, _ = env.step(dummy)

    success = False
    episode_length = 0
    env_action_dim = _get_env_action_dim(env)
    arm_controller = getattr(args, "arm_controller", "cartesian_pose")
    replay_primary, replay_wrist = [], []

    for t in range(max_steps):
        if droid:
            observation = prepare_observation_droid(
                obs, task_description,
                flip_images=args.flip_images, img_size=args.img_res
            )
            if save_video:
                replay_primary.append(observation["observation/exterior_image_1_left"].copy())
                replay_wrist.append(observation["observation/wrist_image_left"].copy())
        else:
            observation = _prepare_observation(obs, task_description, args.flip_images)
            if save_video:
                replay_primary.append(observation["primary_image"].copy())
                replay_wrist.append(observation["wrist_image"].copy())

        start = time.time()
        result = policy.infer(observation)
        action = result["actions"]
        if hasattr(action, "ndim") and action.ndim > 1:
            action = action[0]
        if t % 50 == 0:
            print(f"  t={t}: infer {time.time() - start:.3f}s")

        action = pad_action_for_env(action, arm_controller, env_action_dim)
        obs, reward, done, info = env.step(action.tolist() if hasattr(action, "tolist") else action)
        episode_length += 1

        if use_gui:
            env.env.render()

        if done:
            success = True
            print(f"  Success at t={t}!")
            break

    print(
        f"  Episode {episode_idx}: {'SUCCESS' if success else 'FAILURE'} "
        f"(length={episode_length})"
    )
    return success, episode_length, replay_primary, replay_wrist


def parse_args():
    parser = argparse.ArgumentParser(
        description="LIBERO demo: run policy in sim via WebSocket (no eval)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--policy_server_addr", type=str, default="localhost:8000",
        help="WebSocket policy server address host:port",
    )
    parser.add_argument("--policy", type=str, default="randomPolicy", help="Policy name (for display)")
    parser.add_argument(
        "--task_suite_name", type=str, default="libero_10",
        choices=TASK_SUITE_CHOICES, help="LIBERO task suite",
    )
    parser.add_argument("--task_id", type=int, default=0, help="Task index within the suite")
    parser.add_argument("--num_resets", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--img_res", type=int, default=256, help="Camera image resolution (square)")
    parser.add_argument("--flip_images", action="store_true", default=True,
                        help="Flip images vertically (LIBERO renders upside-down)")
    parser.add_argument("--no_flip_images", action="store_false", dest="flip_images")
    parser.add_argument("--seed", type=int, default=195, help="Random seed")
    parser.add_argument("--droid", action="store_true",
                        help="Use DROID obs format (joint_vel, OpenPI DROID policy)")
    parser.add_argument("--arm_controller", type=str, default="cartesian_pose",
                        choices=list(ARM_CONTROLLER_MAP.keys()),
                        help="Arm controller type (ignored when --droid)")
    parser.add_argument("--gui", action="store_true",
                        help="Enable interactive GUI rendering (default: headless no_gui)")
    parser.add_argument("--demo_log_dir", type=str, default="./demo_log",
                        help="Directory for saved videos in no_gui mode")
    return parser.parse_args()


def main():
    args = parse_args()
    if getattr(args, "droid", False):
        args.arm_controller = "joint_vel"

    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()

    if args.task_id >= task_suite.n_tasks:
        raise ValueError(
            f"task_id {args.task_id} out of range for {args.task_suite_name} "
            f"(has {task_suite.n_tasks} tasks)"
        )

    task = task_suite.get_task(args.task_id)
    task_description = task.language
    max_steps = TASK_MAX_STEPS.get(args.task_suite_name, 520)
    init_states = task_suite.get_task_init_states(args.task_id)

    addr = args.policy_server_addr
    host, port = (addr.rsplit(":", 1) if ":" in addr else (addr, "8000"))
    port = int(port)

    print("=" * 60)
    print("LIBERO Demo (run policy in sim, no eval)")
    print("=" * 60)
    print(f"  task_suite:     {args.task_suite_name}")
    print(f"  task_id:        {args.task_id}")
    print(f"  task:           {task_description}")
    print(f"  num_resets:     {args.num_resets}")
    print(f"  max_steps:      {max_steps}")
    print(f"  policy:         {args.policy}")
    print(f"  policy_server:   ws://{host}:{port}")
    print(f"  img_res:         {args.img_res}")
    print(f"  flip_images:     {args.flip_images}")
    print(f"  droid:           {getattr(args, 'droid', False)}")
    print(f"  arm_controller:  {getattr(args, 'arm_controller', 'cartesian_pose')} ({ARM_CONTROLLER_MAP.get(getattr(args, 'arm_controller', 'cartesian_pose'), 'OSC_POSE')})")
    print(f"  GUI:             {'on (--gui)' if args.gui else 'off (no_gui, videos saved)'}")
    if not args.gui:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.demo_log_dir, f"{args.task_suite_name}--{date_str}")
        os.makedirs(run_dir, exist_ok=True)
        args._run_dir = run_dir
        print(f"  demo_log_dir:    {run_dir}")
    print("=" * 60)

    policy = WebsocketClientPolicy(host=host, port=port)
    print(f"Server metadata: {policy.get_server_metadata()}")

    env = None

    def _cleanup(signum=None, frame=None):
        print("\nCleaning up ...", flush=True)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        policy.close()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if signum else 0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)
    atexit.register(policy.close)

    try:
        use_gui = args.gui
        save_video = not use_gui
        controller = ARM_CONTROLLER_MAP[args.arm_controller]
        env = _create_env(task, args.img_res, controller=controller, use_gui=use_gui)
        if getattr(args, "droid", False):
            enable_joint_pos_observable(env)

        for ep_idx in range(args.num_resets):
            print(f"\n--- Reset {ep_idx + 1}/{args.num_resets} ---")
            env.reset()
            init_state_id = ep_idx % len(init_states)
            obs = env.set_init_state(init_states[init_state_id])
            print(f"Task: {task_description}")

            policy.reset()
            if not getattr(args, "droid", False):
                inner = getattr(env, "env", env)
                action_spec = getattr(inner, "action_spec", None)
                if action_spec is not None:
                    action_low, action_high = action_spec[0], action_spec[1]
                    init_obs = {
                        "action_dim": action_low.shape[0],
                        "action_low": action_low,
                        "action_high": action_high,
                        "task_name": args.task_suite_name,
                        "task_description": task_description,
                    }
                    policy.infer(init_obs)

            success, ep_len, replay_primary, replay_wrist = run_episode(
                args, env, task_description, policy, ep_idx, max_steps,
                use_gui=use_gui, save_video=save_video,
            )
            if save_video and replay_primary and replay_wrist:
                save_rollout_video(
                    replay_primary, replay_wrist, ep_idx, success, task_description,
                    output_dir=getattr(args, "_run_dir", args.demo_log_dir),
                )

        env.close()
        env = None
    finally:
        policy.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
