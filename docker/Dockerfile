# =============================================================================
# libero Docker image (uv-managed Python environment)
# Build from repo root:  docker build -f docker/Dockerfile .
# =============================================================================

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# -----------------------------------------------------------------------------
# 1. System dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    curl wget tar procps bzip2 ca-certificates git build-essential cmake unzip tmux \
    ffmpeg libgl1-mesa-glx libglib2.0-0 \
    libegl1 libosmesa6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*



# 
# Uncomment if project needs MuJoCo / robosuite / EGL rendering:
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libx11-6 libxext6 libxrender1 libsm6 libegl1 libosmesa6 \
#     && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 2. uv (fast Python package manager)
# -----------------------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
# 运行时常见把宿主机目录 bind 到 /root，会遮住镜像内 /root/.local/bin；复制 uv 到 PATH 固定位置
# （集群 apptainer run -B fakeroot:/root 时，entrypoint 的 `uv pip install -e .` 仍可用）。
RUN cp /root/.local/bin/uv /usr/local/bin/uv && chmod +x /usr/local/bin/uv
# Alternative (faster, always latest):
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# -----------------------------------------------------------------------------
# 3. Python venv via uv
# -----------------------------------------------------------------------------
RUN uv venv --python 3.8 /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
# uv 若下载 Python 到 /root/.local/share/uv/python/...，bind /root 时软链接会失效。
# 复制到 /opt/python38 并重链，保证任何 bind 场景下 venv Python 可用（uv 用系统 py3.8 时此步为空操作）。
RUN _py=$(find /root/.local/share/uv/python -maxdepth 3 -name 'python3.8' -type f 2>/dev/null | head -1) && \
    if [ -n "$_py" ]; then \
        _dir=$(dirname "$(dirname "$_py")") && \
        cp -a "$_dir" /opt/python38 && \
        ln -sf /opt/python38/bin/python3.8 /opt/venv/bin/python && \
        ln -sf /opt/python38/bin/python3.8 /opt/venv/bin/python3 && \
        ln -sf /opt/python38/bin/python3.8 /opt/venv/bin/python3.8; \
    fi



# -----------------------------------------------------------------------------
# 4. Non-Python / make-based third-party dependencies → /opt
# -----------------------------------------------------------------------------
# 
# Example:
# WORKDIR /opt/third_party
# RUN git clone https://github.com/<org>/<repo>.git && \
#     cd <repo> && git checkout <commit> && \
#     mkdir build && cd build && cmake .. && make -j$(nproc) && make install
#
# If the dep has Python bindings:
# RUN uv pip install -e /opt/third_party/<repo>

# -----------------------------------------------------------------------------
# 5. Install block — driven by InstallationPlan (<repo>/.nautilus/install_plan.json)
#    or, when no plan exists, by the legacy probe-only Option A/C heuristic.
#
# NOTE: Every `uv sync` / `uv pip install` line carries `--index-strategy
# unsafe-best-match`. uv defaults to "first-index-wins": once any extra index
# (e.g. the PyTorch CUDA wheel index) mentions a package name, uv treats that
# index as the SOLE source and refuses to consult PyPI for a better version.
# This breaks pins like `packaging==23.0` because the PyTorch index ships
# partial mirrors at different versions. `unsafe-best-match` tells uv to
# consider all indexes and pick the best match — what users intuitively
# expect. Harmless when no extra-index is configured. See:
# https://docs.astral.sh/uv/concepts/indexes/
# -----------------------------------------------------------------------------
WORKDIR /build
COPY pyproject.toml uv.lock requirements.txt setup.py ./
RUN uv pip install --no-deps 'setuptools>=62.3,<81' --index-strategy unsafe-best-match
RUN uv sync --frozen --no-install-project --active

# 
# Slot for downstream sub-skills (benchmark / policy generators) to inject
# extra pip installs via extend_base.py. Leave the placeholder line above —
# it is the insertion anchor.

# -----------------------------------------------------------------------------
# 6. Extension point for sub-skills (benchmark / policy / robot env-generator)
# -----------------------------------------------------------------------------
# Sub-skills insert RUN lines ABOVE this marker via their extend_base.py script.
# Do NOT remove the marker line — it is how downstream tooling locates the insertion point.
# <<<EXTENSION_INSERT_ABOVE>>>

# -----------------------------------------------------------------------------
# 7. Bake libero source into the image (self-contained)
# -----------------------------------------------------------------------------
# 与上游「源码运行时 bind」不同：把 libero 包(含 bddl_files / init_files / assets 等众多小文件)
# 连同源码 COPY 进镜像并就地 editable 安装，使众多小文件封进 .sif（squashfs 单文件 = 1 inode），
# 集群上不必 bind 源码。**只**在运行时 bind 演示数据集到 /workspace/libero/datasets（见 entrypoint
# 生成的 config.yaml）。改 libero 代码需重 build；本地 docker-compose 若仍 bind 源码则兼容退回。
COPY . /workspace/libero
RUN cd /workspace/libero && uv pip install -e . --index-strategy unsafe-best-match

# -----------------------------------------------------------------------------
# 8. Workspace + entrypoint
# -----------------------------------------------------------------------------
WORKDIR /workspace/libero
# libero 从 LIBERO_CONFIG_PATH(默认 ~/.libero)读 config.yaml；entrypoint 把它写到
# /workspace/.libero，故显式对齐，否则 import libero 会触发交互式数据集路径提问(EOFError)。
ENV LIBERO_CONFIG_PATH=/workspace/.libero
# headless EGL 离屏渲染(robosuite/mujoco)，run_eval 仿真必需。
ENV MUJOCO_GL=egl
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
