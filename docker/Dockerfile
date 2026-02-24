# =========================================================
# LIBERO Dockerfile (Micromamba-based, GPU-ready)
# =========================================================

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

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
# Install Micromamba
# ---------------------------------------------------------
RUN curl -Ls https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 -o /usr/bin/micromamba \
 && chmod +x /usr/bin/micromamba

# ---------------------------------------------------------
# Create conda environment (Python 3.8 for LIBERO)
# ---------------------------------------------------------
RUN micromamba create -y -n libero python=3.8 -c conda-forge \
 && micromamba clean -a -y

# ---------------------------------------------------------
# Configure shell hook for micromamba
# ---------------------------------------------------------
RUN /bin/bash -c "micromamba shell init -s bash" \
 && echo "micromamba activate libero" >> /root/.bashrc

# ---------------------------------------------------------
# Upgrade pip
# ---------------------------------------------------------
RUN micromamba run -n libero python -m pip install --upgrade pip

# ---------------------------------------------------------
# Install PyTorch (CUDA 11.3 per LIBERO README)
# ---------------------------------------------------------
RUN micromamba run -n libero python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# ---------------------------------------------------------
# Install requirements.txt + robosuite 1.4.0
# ---------------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN micromamba run -n libero python -m pip install -r /tmp/requirements.txt

# ---------------------------------------------------------
# Set up workspace directory (libero installed in entrypoint for in-container dev)
# ---------------------------------------------------------
WORKDIR /workspace

# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
