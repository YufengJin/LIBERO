# =========================================================
# LIBERO Dockerfile (uv-based, GPU-ready)
# =========================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV VIRTUAL_ENV=/opt/venv
# uv sync defaults to .venv in the project dir; pin the pre-created venv (same as robocasa layout intent)
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH=/opt/venv/bin:/root/.local/bin:$PATH

# ---------------------------------------------------------
# System dependencies (MuJoCo/robosuite rendering, SSL, etc.) + debug tools
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y \
    curl \
    bzip2 \
    ca-certificates \
    git \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libegl1 \
    libosmesa6 \
    ffmpeg \
    lsof \
    psmisc \
    tmux \
    vim \
    x11-apps \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Install uv (copy static binary from official image)
# ---------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# ---------------------------------------------------------
# Create Python 3.8 venv managed by uv
# ---------------------------------------------------------
RUN uv venv /opt/venv --python 3.8

# ---------------------------------------------------------
# Pre-install project dependencies (leverages Docker layer cache)
# Only pyproject.toml + uv.lock are copied so this layer is
# re-used unless dependencies change.
# ---------------------------------------------------------
WORKDIR /workspace
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# ---------------------------------------------------------
# Entrypoint (libero itself is installed editable at container start)
# ---------------------------------------------------------
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
