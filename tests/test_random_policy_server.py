#!/usr/bin/env python3
"""
Test policy server for LIBERO — returns random actions via WebSocket.

Purpose:
    Random-policy WebSocket server used by the benchmark-generator L3_IL smoke
    test and as a quick connectivity check for users wiring in their own policy
    backends. Ignores the ``__meta__`` envelope on every payload — only the
    action_dim from the init handshake matters.

Example:
    # Cartesian-pose 7D (default, matches OpenVLA-OFT)
    python tests/test_random_policy_server.py --port 8765

    # Joint-vel 8D (matches OpenPI / DROID)
    python tests/test_random_policy_server.py --port 8765 --droid

Then connect with:
    python scripts/run_demo.py --policy_server_addr localhost:8765 --task_suite_name libero_10
    python scripts/run_eval.py --policy_server_addr localhost:8765 --task_suite_name libero_10
"""

import argparse
import logging
import sys
import os
from typing import Dict

import numpy as np

_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from policy_websocket import BasePolicy, WebsocketPolicyServer

logger = logging.getLogger(__name__)


class RandomPolicy(BasePolicy):
    """Returns uniformly random small actions. Adapts to action_dim from init_obs (e.g. joint_pos)."""

    def __init__(self, droid: bool = False) -> None:
        self._action_dim: int = 8 if droid else 7
        self._scale: float = 0.1

    def infer(self, obs: Dict) -> Dict:
        if "action_dim" in obs and "primary_image" not in obs:
            self._action_dim = int(obs["action_dim"])
        action = np.random.uniform(-self._scale, self._scale, size=self._action_dim).astype(np.float64)
        action[-1] = -1.0  # gripper closed
        return {"actions": action}

    def reset(self) -> None:
        pass


def main():
    parser = argparse.ArgumentParser(description="LIBERO test policy server (random actions)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--droid", action="store_true",
                        help="DROID mode: return 8-dim joint_vel actions")
    args = parser.parse_args()

    policy = RandomPolicy(droid=args.droid)
    action_dim = 8 if args.droid else 7
    metadata = {"policy_name": "RandomPolicy", "action_dim": action_dim}

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    # Write PID file so smoke runners can `kill $(cat /tmp/policy_server_<port>.pid)`.
    pid_path = f"/tmp/policy_server_{args.port}.pid"
    try:
        with open(pid_path, "w") as f:
            f.write(str(os.getpid()))
    except OSError:
        pid_path = None

    print(f"Starting RandomPolicy server on ws://{args.host}:{args.port}")
    if pid_path is not None:
        print(f"PID file: {pid_path}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if pid_path is not None:
            try:
                os.remove(pid_path)
            except OSError:
                pass
    print("Server stopped, port released.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
