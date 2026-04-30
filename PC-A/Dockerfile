# Segmentation Worker Dockerfile with CUDA Support
# Updated for RTX 5080 (Blackwell sm_120) compatibility
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip3 install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch dengan CUDA 12.8 + Blackwell (sm_120) support untuk RTX 5080
# PyTorch 2.5.1+cu124 TIDAK support sm_120, fallback ke CPU saat .cuda() dipanggil.
# Stable build pertama yang support sm_120: torch 2.7.0 + cu128.
RUN pip3 install --no-cache-dir torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Copy worker code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /tmp/worker-cache /models

# Expose metrics port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run worker
CMD ["python", "worker.py"]
