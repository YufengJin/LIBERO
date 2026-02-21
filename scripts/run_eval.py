#!/usr/bin/env python3
"""
run_eval.py -- LIBERO evaluation client (WebSocket).

Runs the LIBERO simulation loop and delegates action inference to a remote
Policy Server over WebSocket.  Evaluates all tasks in a given task suite,
saves rollout videos and a text log -- matching the robocasa eval output
conventions.

Usage:
    # Start the policy server first:
    python tests/test_random_policy_server.py --port 8000

    # Run evaluation:
    python scripts/run_eval.py \
        --task_suite_name libero_10 \
        --num_trials_per_task 50 \
        --policy randomPolicy \
        --seed 195 \
        --policy_server_addr localhost:8000

    # DROID format for OpenPI DROID policy (joint_vel, DROID obs):
    python scripts/run_eval.py --droid --policy_server_addr localhost:8000 --task_suite_name libero_10

    Logs and videos are written to: <log_dir>/<task_suite_name>--<YYYYMMDD_HHMMSS>/
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
    return 8  # fallback for joint_vel


def log(msg: str, log_file=None):
    """Print a message and optionally write it to a log file."""
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


# -- environment helpers ------------------------------------------------------

def _create_env(task, img_res, controller="OSC_POSE"):
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        controller=controller,
        camera_heights=img_res,
        camera_widths=img_res,
    )
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


# -- video saving ------------------------------------------------------------

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


# -- episode / task runners ---------------------------------------------------

def run_episode(args, env, task_description, policy, episode_idx, max_steps,
                log_file=None):
    """Run a single evaluation episode."""
    droid = getattr(args, "droid", False)
    dummy = DUMMY_ACTION_JOINT_VEL if droid else DUMMY_ACTION_OSC

    NUM_WAIT_STEPS = 10
    for _ in range(NUM_WAIT_STEPS):
        obs, _, _, _ = env.step(dummy)

    success = False
    episode_length = 0
    replay_primary, replay_wrist = [], []

    env_action_dim = _get_env_action_dim(env)
    arm_controller = getattr(args, "arm_controller", "cartesian_pose")

    for t in range(max_steps):
        if droid:
            observation = prepare_observation_droid(
                obs, task_description,
                flip_images=args.flip_images, img_size=args.img_res
            )
            replay_primary.append(observation["observation/exterior_image_1_left"].copy())
            replay_wrist.append(observation["observation/wrist_image_left"].copy())
        else:
            observation = _prepare_observation(obs, task_description, args.flip_images)
            replay_primary.append(observation["primary_image"].copy())
            replay_wrist.append(observation["wrist_image"].copy())

        start = time.time()
        result = policy.infer(observation)
        action = result["actions"]
        if hasattr(action, "ndim") and action.ndim > 1:
            action = action[0]
        query_time = time.time() - start

        if t % 50 == 0:
            log(f"  t={t}: infer {query_time:.3f}s, action[:4]={action[:4]}", log_file)

        action = pad_action_for_env(action, arm_controller, env_action_dim)
        obs, reward, done, info = env.step(
            action.tolist() if hasattr(action, "tolist") else action
        )
        episode_length += 1

        if done:
            success = True
            log(f"  Success at t={t}!", log_file)
            break

    log(
        f"  Episode {episode_idx}: {'SUCCESS' if success else 'FAILURE'} "
        f"(length={episode_length})",
        log_file,
    )
    return success, episode_length, replay_primary, replay_wrist


def run_task(args, task_suite, task_id, policy, global_ep_counter,
             global_successes, log_file=None):
    """Evaluate a single task over multiple episodes."""
    task = task_suite.get_task(task_id)
    task_description = task.language
    max_steps = TASK_MAX_STEPS.get(args.task_suite_name, 520)
    init_states = task_suite.get_task_init_states(task_id)

    log(f"\n{'=' * 60}", log_file)
    log(f"Task {task_id}: {task_description}", log_file)
    log(f"{'=' * 60}", log_file)

    controller = "JOINT_VELOCITY" if getattr(args, "droid", False) else "OSC_POSE"
    env = _create_env(task, args.img_res, controller=controller)
    if getattr(args, "droid", False):
        enable_joint_pos_observable(env)

    task_successes = []
    task_lengths = []

    for ep_idx in range(args.num_trials_per_task):
        log(f"\n--- Task {task_id} | Episode {ep_idx + 1}/{args.num_trials_per_task} ---", log_file)

        env.reset()
        init_state_id = ep_idx % len(init_states)
        obs = env.set_init_state(init_states[init_state_id])
        log(f"Task description: {task_description}", log_file)

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

        success, length, rep_p, rep_w = run_episode(
            args, env, task_description, policy,
            global_ep_counter[0], max_steps, log_file,
        )
        task_successes.append(success)
        task_lengths.append(length)
        global_ep_counter[0] += 1
        if success:
            global_successes[0] += 1

        if args.save_video:
            save_rollout_video(
                rep_p, rep_w,
                global_ep_counter[0] - 1, success, task_description,
                output_dir=args.log_dir,
            )

        sr = sum(task_successes) / len(task_successes) * 100
        log(
            f"Task {task_id} running: {sum(task_successes)}/{len(task_successes)} ({sr:.1f}%)",
            log_file,
        )

    env.close()

    task_sr = np.mean(task_successes) if task_successes else 0.0
    task_avg_len = np.mean(task_lengths) if task_lengths else 0.0
    log(f"\nTask {task_id} ({task_description}):", log_file)
    log(f"  Success rate: {task_sr:.4f} ({task_sr * 100:.1f}%)", log_file)
    log(f"  Avg length:   {task_avg_len:.1f}", log_file)

    return task_sr, task_avg_len


# -- main ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="LIBERO WebSocket evaluation client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--policy_server_addr", type=str, default="localhost:8000",
        help="Address of the WebSocket policy server (host:port)",
    )
    parser.add_argument("--policy", type=str, default="randomPolicy",
                        help="Policy name for logging")
    parser.add_argument(
        "--task_suite_name", type=str, default="libero_10",
        choices=TASK_SUITE_CHOICES, help="LIBERO task suite",
    )
    parser.add_argument("--num_trials_per_task", type=int, default=50,
                        help="Number of evaluation episodes per task")
    parser.add_argument("--img_res", type=int, default=256,
                        help="Camera image resolution (square)")
    parser.add_argument("--flip_images", action="store_true", default=True,
                        help="Flip images vertically (LIBERO renders upside-down)")
    parser.add_argument("--no_flip_images", action="store_false", dest="flip_images")
    parser.add_argument("--seed", type=int, default=195, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic seeding per episode")
    parser.add_argument("--no_deterministic", action="store_false", dest="deterministic")
    parser.add_argument("--log_dir", type=str, default="./eval_logs",
                        help="Directory for logs and rollout videos")
    parser.add_argument("--save_video", action="store_true", default=True,
                        help="Save rollout videos")
    parser.add_argument("--no_save_video", action="store_false", dest="save_video")
    parser.add_argument("--droid", action="store_true",
                        help="Use DROID obs format (joint_vel, OpenPI DROID policy)")
    return parser.parse_args()


def main():
    args = parse_args()
    if getattr(args, "droid", False):
        args.arm_controller = "joint_vel"

    np.random.seed(args.seed)

    # Log directory: <base>/<task_suite_name>--YYYYMMDD_HHMMSS/
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.log_dir, f"{args.task_suite_name}--{date_str}")
    os.makedirs(run_dir, exist_ok=True)
    args.log_dir = run_dir

    log_path = os.path.join(run_dir, "eval.log")
    log_file = open(log_path, "w")

    # Run header
    log("=" * 60, log_file)
    log("LIBERO WebSocket Eval Run", log_file)
    log("=" * 60, log_file)
    log(f"  policy:              {args.policy}", log_file)
    log(f"  task_suite_name:     {args.task_suite_name}", log_file)
    log(f"  num_trials_per_task: {args.num_trials_per_task}", log_file)
    log(f"  seed:                {args.seed}", log_file)
    log(f"  deterministic:       {args.deterministic}", log_file)
    log(f"  policy_server:       {args.policy_server_addr}", log_file)
    log(f"  log_dir (run_dir):   {run_dir}", log_file)
    log(f"  img_res:             {args.img_res}", log_file)
    log(f"  flip_images:         {args.flip_images}", log_file)
    log(f"  save_video:          {args.save_video}", log_file)
    log(f"  droid:               {getattr(args, 'droid', False)}", log_file)
    log(f"  arm_controller:      {getattr(args, 'arm_controller', 'cartesian_pose')} ({ARM_CONTROLLER_MAP.get(getattr(args, 'arm_controller', 'cartesian_pose'), 'OSC_POSE')})", log_file)
    log("=" * 60, log_file)
    log("", log_file)

    addr = args.policy_server_addr
    host, port = (addr.rsplit(":", 1) if ":" in addr else (addr, "8000"))
    port = int(port)

    log(f"Connecting to policy server at ws://{host}:{port} ...", log_file)
    policy = WebsocketClientPolicy(host=host, port=port)
    log(f"Server metadata: {policy.get_server_metadata()}", log_file)

    # Graceful shutdown
    def _cleanup(signum=None, frame=None):
        print("\nCleaning up ...", flush=True)
        policy.close()
        if not log_file.closed:
            log_file.close()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if signum else 0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)
    atexit.register(policy.close)

    # Initialize LIBERO benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    log(f"Task suite: {args.task_suite_name} ({num_tasks} tasks)", log_file)

    # Counters (mutable list so nested functions can update)
    global_ep_counter = [0]
    global_successes = [0]

    per_task_sr = []

    try:
        for task_id in range(num_tasks):
            task_sr, task_avg_len = run_task(
                args, task_suite, task_id, policy,
                global_ep_counter, global_successes, log_file,
            )
            per_task_sr.append(task_sr)

        # Final summary
        total_episodes = global_ep_counter[0]
        total_successes = global_successes[0]
        overall_sr = total_successes / total_episodes if total_episodes > 0 else 0.0
        avg_task_sr = np.mean(per_task_sr) if per_task_sr else 0.0

        log("\n" + "=" * 60, log_file)
        log("FINAL RESULTS", log_file)
        log("=" * 60, log_file)
        log(f"Policy:                {args.policy}", log_file)
        log(f"Task suite:            {args.task_suite_name}", log_file)
        log(f"Total episodes:        {total_episodes}", log_file)
        log(f"Total successes:       {total_successes}", log_file)
        log(f"Overall success rate:  {overall_sr:.4f} ({overall_sr * 100:.1f}%)", log_file)
        log(f"Avg task success rate: {avg_task_sr:.4f} ({avg_task_sr * 100:.1f}%)", log_file)
        log("Per-task success rates:", log_file)
        for tid, sr in enumerate(per_task_sr):
            t = task_suite.get_task(tid)
            log(f"  [{tid}] {t.language}: {sr:.4f} ({sr * 100:.1f}%)", log_file)
        log("=" * 60, log_file)

        log(f"\nLog saved to: {log_path}", log_file)
        print(f"\nLog saved to: {log_path}")
        print(f"Run directory (logs + videos): {run_dir}")
        return overall_sr

    finally:
        policy.close()
        if not log_file.closed:
            log_file.close()


if __name__ == "__main__":
    main()
