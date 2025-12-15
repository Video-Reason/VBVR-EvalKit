FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    TRANSFORMERS_CACHE=/workspace/.cache/transformers

# Install system dependencies (Python 3.10 is default in Ubuntu 22.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    wget \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    python3 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /workspace

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# Create necessary directories for weights, cache, and data
RUN mkdir -p /workspace/weights \
    /workspace/.cache/huggingface \
    /workspace/.cache/torch \
    /workspace/.cache/transformers \
    /workspace/data/questions \
    /workspace/data/outputs \
    /workspace/data/scorings

# Expose Web Dashboard port
EXPOSE 5000

# Default command
CMD ["bash"]
