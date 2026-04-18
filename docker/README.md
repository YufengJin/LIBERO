# LIBERO Docker Guide

This document describes how to build and run LIBERO containers.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker runtime configured (for GPU support)
- For X11 mode: X11 server running on the host

## Python environment

镜像使用 [uv](https://github.com/astral-sh/uv) 在 `/opt/venv` 安装依赖（构建时 `uv sync --frozen`，与仓库根目录 `pyproject.toml` / `uv.lock` 一致）。`PATH` 已包含 `/opt/venv/bin`，进入容器后直接运行 `python` 即可，无需再激活 conda/mamba。

## Build Image

From the project root:

```bash
cd docker
docker-compose -f docker-compose.x11.yaml build
# or
docker-compose -f docker-compose.headless.yaml build
```

With custom image name:

```bash
IMAGE=libero:custom docker-compose -f docker-compose.x11.yaml build
```

## Start Container

### X11 mode (GUI display)

For visualization (run_demo --gui):

```bash
cd docker
DISPLAY=${DISPLAY} docker-compose -f docker-compose.x11.yaml up -d
```

Or foreground:

```bash
DISPLAY=${DISPLAY} docker-compose -f docker-compose.x11.yaml up
```

### Headless mode

For batch evaluation, training, or other non-GUI use:

```bash
cd docker
docker-compose -f docker-compose.headless.yaml up -d
```

Or foreground:

```bash
docker-compose -f docker-compose.headless.yaml up
```

## Attach to Container

```bash
docker exec -it libero_container bash
```

## Stop Container

```bash
cd docker
docker-compose -f docker-compose.x11.yaml down
# or
docker-compose -f docker-compose.headless.yaml down
```

## View Logs

```bash
docker logs libero_container
# or follow
docker logs -f libero_container
```

## Configuration

### Container name
- Fixed name: `libero_container`

### GPU
- Uses all available NVIDIA GPUs by default
- Set `GPU` env var to override (default: `all`)

### Working directory
- Container workdir: `/workspace`

### Network
- Uses `host` network mode

### Environment variables
- **DISPLAY** (X11 only): X11 display
- **GPU**: GPU selection (default: `all`)

## Troubleshooting

### X11 permission denied

```bash
xhost +local:docker
```

### GPU not detected

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

### Container name conflict

```bash
docker stop libero_container
docker rm libero_container
```

### LIBERO config prompt on first run

The entrypoint auto-creates `~/.libero/config.yaml` (or `$LIBERO_ROOT/.libero/config.yaml` when source is mounted). If you see an interactive prompt, ensure the workspace is mounted and entrypoint runs.

## Example Usage

Inside the container:

```bash
# The uv-managed venv at /opt/venv is already on PATH — use `python` directly.

# Run eval (start policy server first in another terminal)
python tests/test_random_policy_server.py --port 8000
python scripts/run_eval.py --task_suite_name libero_10 --policy_server_addr localhost:8000

# Run demo (headless, saves videos to demo_log/)
python scripts/run_demo.py --task_suite_name libero_10 --policy_server_addr localhost:8000

# Run demo with GUI
python scripts/run_demo.py --gui --task_suite_name libero_10 --policy_server_addr localhost:8000

# Download datasets (if not present)
python benchmark_scripts/download_libero_datasets.py --use-huggingface
```

## Notes

- X11 mode requires a running X server and `xhost` access for Docker
- First build can take a while (dependencies and robosuite)
- `/workspace` is the project root; LIBERO is installed editable at entrypoint
- Add volume mounts in docker-compose for persistent data
