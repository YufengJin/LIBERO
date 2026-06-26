#!/usr/bin/env bash
# 在 libero.sif 内跑 run_eval（websocket sim client，连 policy server）。
# 策略 server 在 droid.sif 里跑（droid_policy_learning/apptainer/run_policy_server.sh）。
#
# 已验证的关键 flag（本机 RTX4090 + apptainer 1.5.2 实测）：
#   --writable-tmpfs  : sif 只读，entrypoint 要写 /workspace/.libero/config.yaml、run_eval 要写 eval_logs → 需可写覆盖层。
#   MUJOCO_GL=osmesa  : CPU 软件渲染，避开 apptainer --nv 的宿主 GL 库 vs 容器旧 glibc 冲突；sim client 无需 GPU。
#   --log_dir /tmp/...: 只读 sif，日志写到可写的 /tmp。
# 源码已烤入镜像；演示数据集（hdf5）若评测需要可 -B 到 /workspace/libero/datasets（init_states 已在包内，通常不需要）。
set -euo pipefail
SIF="${LIBERO_SIF:-/mnt/ssd2T/yjin/sif/libero.sif}"
POLICY_ADDR="${POLICY_SERVER_ADDR:-localhost:8765}"
SUITE="${TASK_SUITE_NAME:-libero_10}"
NTRIALS="${NUM_TRIALS_PER_TASK:-1}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/mnt/ssd2T/yjin/.apptainer_cache}"

exec apptainer run --writable-tmpfs --env MUJOCO_GL=osmesa "$SIF" \
  bash -lc "cd /workspace/libero && python scripts/run_eval.py \
    --task_suite_name '${SUITE}' --policy_server_addr '${POLICY_ADDR}' \
    --num_trials_per_task '${NTRIALS}' --log_dir /tmp/libero_eval $*"
