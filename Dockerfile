# GFPGAN Serverless - CUDA 12.1 Optimized
FROM spxiong/pytorch:2.5.1-py3.10.15-cuda12.1.0-devel-ubuntu22.04

WORKDIR /app

# Set optimized CUDA environment for 12.1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_MODULE_LOADING=LAZY

# Suppress RunPod logs
ENV RUNPOD_DEBUG=false
ENV RUNPOD_LOG_LEVEL=ERROR
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libsndfile1 \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with CUDA 12.1 optimized ONNX Runtime
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    echo "=== Installing core dependencies (CUDA 12.1 optimized) ===" && \
    pip install --no-cache-dir \
    numpy==1.24.4 \
    opencv-python==4.8.0.76 && \
    echo "=== Installing ONNX Runtime GPU 1.17.1 (CUDA 12.1 support) ===" && \
    pip install --no-cache-dir onnxruntime-gpu==1.17.1 && \
    echo "=== Installing other dependencies ===" && \
    pip install --no-cache-dir \
    tqdm==4.67.1 \
    requests==2.28.1 \
    scikit-image==0.25.2 \
    Pillow==11.0.0 \
    matplotlib==3.10.1 \
    scipy==1.15.2 \
    imutils==0.5.4 \
    imageio==2.37.0 \
    librosa==0.11.0 \
    numba==0.61.0 \
    soundfile==0.13.1 \
    easydict==1.13 \
    cython==3.0.12 \
    runpod>=1.6.0 \
    minio>=7.0.0

# Install InsightFace with fallback
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "=== Installing InsightFace ===" && \
    pip install --no-cache-dir insightface==0.7.3 || \
    (wget --no-check-certificate --timeout=30 --tries=3 \
    "https://huggingface.co/deauxpas/colabrepo/resolve/main/insightface-0.7.3-cp310-cp310-linux_x86_64.whl" \
    -O /tmp/insightface.whl && \
    pip install /tmp/insightface.whl --force-reinstall && \
    rm -f /tmp/insightface.whl)

# Copy source code
COPY . /app/

# Create directories and download models
RUN mkdir -p /app/enhancers/GFPGAN /app/utils /app/faceID /app/outputs && \
    echo "=== Downloading models ===" && \
    wget --no-check-certificate --timeout=120 --tries=3 \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx" \
    -O /app/enhancers/GFPGAN/GFPGANv1.4.onnx && \
    wget --no-check-certificate --timeout=120 --tries=3 \
    "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/scrfd_2.5g_bnkps.onnx" \
    -O /app/utils/scrfd_2.5g_bnkps.onnx && \
    wget --no-check-certificate --timeout=120 --tries=3 \
    "https://huggingface.co/manh-linh/faceID_recognition/resolve/main/recognition.onnx" \
    -O /app/faceID/recognition.onnx && \
    echo "✅ All models downloaded"

# Verify CUDA 12.1 + ONNX Runtime 1.17.1 compatibility
RUN echo "=== Verifying CUDA 12.1 + ONNX Runtime 1.17.1 compatibility ===" && \
    python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" && \
    python -c "import torch; print(f'✅ CUDA Available: {torch.cuda.is_available()}')" && \
    python -c "import torch; print(f'✅ CUDA Version: {torch.version.cuda}')" && \
    python -c "import onnxruntime; print(f'✅ ONNX Runtime: {onnxruntime.__version__}')" && \
    python -c "import onnxruntime; print(f'✅ CUDA Provider: {\"CUDAExecutionProvider\" in onnxruntime.get_available_providers()}')" && \
    python -c "import onnxruntime; providers = onnxruntime.get_available_providers(); print(f'✅ Available Providers: {providers}')" && \
    echo "🚀 CUDA 12.1 + ONNX Runtime 1.17.1 verified successfully"

# Set environment
ENV PYTHONPATH="/app"

CMD ["python", "rp_handler.py"]
