#!/usr/bin/env bash
set -e

cd /workspace

# Determine LIBERO project root (may be /workspace/LIBERO when parent is mounted)
LIBERO_ROOT=""
if [ -f /workspace/LIBERO/setup.py ]; then
  LIBERO_ROOT="/workspace/LIBERO"
elif [ -f /workspace/setup.py ]; then
  LIBERO_ROOT="/workspace"
fi

# Install libero in editable mode when source is mounted
if [ -n "${LIBERO_ROOT}" ]; then
  # Create LIBERO config BEFORE any libero import (avoids interactive prompt during pip install)
  # Use workspace path so config persists on the mounted volume
  export LIBERO_CONFIG_PATH="${LIBERO_ROOT}/.libero"
  mkdir -p "${LIBERO_CONFIG_PATH}"
  CONFIG_FILE="${LIBERO_CONFIG_PATH}/config.yaml"
  if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Initializing LIBERO config at ${CONFIG_FILE}..."
    BENCHMARK_ROOT="${LIBERO_ROOT}/libero/libero"
    micromamba run -n libero python -c "
import os
import yaml

config_dir = '${LIBERO_CONFIG_PATH}'
os.makedirs(config_dir, exist_ok=True)
config_file = os.path.join(config_dir, 'config.yaml')
benchmark_root = '${BENCHMARK_ROOT}'
config = {
    'benchmark_root': benchmark_root,
    'bddl_files': os.path.join(benchmark_root, 'bddl_files'),
    'init_states': os.path.join(benchmark_root, 'init_files'),
    'datasets': os.path.join(benchmark_root, '../datasets'),
    'assets': os.path.join(benchmark_root, 'assets'),
}
with open(config_file, 'w') as f:
    yaml.dump(config, f)
print('Config written to', config_file)
"
  fi

  echo "Installing libero from ${LIBERO_ROOT} (editable)..."
  micromamba run -n libero pip install -e "${LIBERO_ROOT}"

  # Check if datasets exist, prompt to download if not
  DATASETS_DIR="${LIBERO_ROOT}/libero/datasets"
  if [ ! -d "${DATASETS_DIR}" ] || [ -z "$(ls -A "${DATASETS_DIR}" 2>/dev/null)" ]; then
    echo "[INFO] Datasets not found at ${DATASETS_DIR}."
    echo "       Run: python benchmark_scripts/download_libero_datasets.py --use-huggingface"
  fi
fi

if [ $# -eq 0 ]; then
  exec bash
else
  exec micromamba run -n libero "$@"
fi
