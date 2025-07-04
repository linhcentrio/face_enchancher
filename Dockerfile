# Face Enhancement Serverless Worker
FROM spxiong/pytorch:2.5.1-py3.10.15-cuda12.1.0-devel-ubuntu22.04 AS base

WORKDIR /app

# Set CUDA environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_MODULE_LOADING=LAZY

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10-dev \
    python3.10-distutils \
    build-essential \
    libgl1 \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app/

# Upgrade pip và install tools
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

# Install core dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    opencv-python==4.9.0.80 \
    numpy==1.26.4 \
    tqdm==4.66.1 \
    onnxruntime-gpu==1.21.0 \
    runpod>=1.6.0 \
    minio>=7.0.0 \
    scikit-image>=0.14.2 \
    Pillow \
    matplotlib \
    scipy \
    easydict \
    cython \
    requests \
    concurrent-futures

# Install InsightFace dependencies and package
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "=== Installing InsightFace ===" && \
    pip install insightface==0.7.3 --no-cache-dir || \
    (echo "=== Pip install failed, downloading wheel ===" && \
    wget --no-check-certificate --timeout=30 --tries=3 \
    "https://huggingface.co/deauxpas/colabrepo/resolve/main/insightface-0.7.3-cp310-cp310-linux_x86_64.whl" \
    -O /tmp/insightface-0.7.3-cp310-cp310-linux_x86_64.whl && \
    pip install /tmp/insightface-0.7.3-cp310-cp310-linux_x86_64.whl --force-reinstall && \
    rm -f /tmp/insightface-0.7.3-cp310-cp310-linux_x86_64.whl)

# Copy source code
COPY . /app/

# Create model directories
RUN echo "=== Creating model directories ===" && \
    mkdir -p /app/enhancers/GFPGAN && \
    mkdir -p /app/enhancers/GPEN && \
    mkdir -p /app/enhancers/Codeformer && \
    mkdir -p /app/enhancers/restoreformer && \
    mkdir -p /app/utils && \
    mkdir -p /app/faceID && \
    mkdir -p /app/outputs

# Download GFPGAN model
RUN echo "=== Downloading GFPGAN model ===" && \
    wget --no-check-certificate --timeout=60 --tries=3 \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx" \
    -O /app/enhancers/GFPGAN/GFPGANv1.4.onnx && \
    echo "✅ GFPGAN model downloaded"

# Download GPEN model
RUN echo "=== Downloading GPEN model ===" && \
    wget --no-check-certificate --timeout=60 --tries=3 \
    "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/GPEN-BFR-256-sim.onnx" \
    -O /app/enhancers/GPEN/GPEN-BFR-256-sim.onnx && \
    echo "✅ GPEN model downloaded"

# Download CodeFormer model
RUN echo "=== Downloading CodeFormer model ===" && \
    wget --no-check-certificate --timeout=60 --tries=3 \
    "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/codeformerfixed.onnx" \
    -O /app/enhancers/Codeformer/codeformerfixed.onnx && \
    echo "✅ CodeFormer model downloaded"

# Download RestoreFormer model
RUN echo "=== Downloading RestoreFormer model ===" && \
    wget --no-check-certificate --timeout=60 --tries=3 \
    "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/restoreformer16.onnx" \
    -O /app/enhancers/restoreformer/restoreformer16.onnx && \
    echo "✅ RestoreFormer model downloaded"

# Download face detection model
RUN echo "=== Downloading face detection model ===" && \
    wget --no-check-certificate --timeout=60 --tries=3 \
    "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/scrfd_2.5g_bnkps.onnx" \
    -O /app/utils/scrfd_2.5g_bnkps.onnx && \
    echo "✅ Face detection model downloaded"

# Download face recognition model
RUN echo "=== Downloading face recognition model ===" && \
    wget --no-check-certificate --timeout=60 --tries=3 \
    "https://huggingface.co/manh-linh/faceID_recognition/resolve/main/recognition.onnx" \
    -O /app/faceID/recognition.onnx && \
    echo "✅ Face recognition model downloaded"

# Verify all model files exist
RUN echo "=== Verifying model files ===" && \
    test -f /app/enhancers/GFPGAN/GFPGANv1.4.onnx && echo "✅ GFPGAN model verified" && \
    test -f /app/enhancers/GPEN/GPEN-BFR-256-sim.onnx && echo "✅ GPEN model verified" && \
    test -f /app/enhancers/Codeformer/codeformerfixed.onnx && echo "✅ CodeFormer model verified" && \
    test -f /app/enhancers/restoreformer/restoreformer16.onnx && echo "✅ RestoreFormer model verified" && \
    test -f /app/utils/scrfd_2.5g_bnkps.onnx && echo "✅ Face detection model verified" && \
    test -f /app/faceID/recognition.onnx && echo "✅ Face recognition model verified"

# Set environment variables
ENV PYTHONPATH="/app"
ENV TORCH_HOME="/app/models"
ENV HF_HOME="/app/models"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

CMD ["python", "rp_handler.py"]
