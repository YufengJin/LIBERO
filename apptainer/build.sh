#!/usr/bin/env bash
# 构建 libero.sif：先 docker build，再从本地 docker daemon 转 squashfs sif。
# 与 docker/Dockerfile 对齐（源码烤入；/root 软链修复；LIBERO_CONFIG_PATH=/workspace/.libero；MUJOCO_GL=egl）。
# 用法（在 benchmarks/LIBERO 下）：bash apptainer/build.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-vla/libero:latest}"
SIF_OUT="${SIF_OUT:-/mnt/ssd2T/yjin/sif/libero.sif}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/mnt/ssd2T/yjin/.apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/mnt/ssd2T/yjin/.apptainer_tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$(dirname "$SIF_OUT")"

cd "$REPO_ROOT"
echo ">> docker build $IMAGE_TAG"
docker build -f docker/Dockerfile -t "$IMAGE_TAG" .
echo ">> apptainer build $SIF_OUT  <-  docker-daemon://$IMAGE_TAG"
apptainer build --force "$SIF_OUT" "docker-daemon://$IMAGE_TAG"
echo ">> done: $SIF_OUT"
ls -lh "$SIF_OUT"
