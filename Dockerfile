# GFPGAN Serverless - Subprocess Mode (Minimal)
FROM spxiong/pytorch:2.5.1-py3.10.15-cuda12.1.0-devel-ubuntu22.04

WORKDIR /app

# Set CUDA environment
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

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

# Install Python dependencies (minimal for subprocess mode)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --no-cache-dir \
    numpy==1.24.4 \
    opencv-python==4.8.0.76 \
    onnxruntime-gpu==1.14.1 \
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
    insightface==0.7.3 \
    runpod>=1.6.0 \
    minio>=7.0.0

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

# Verify enhancer_cli.py exists and is executable
RUN test -f /app/enhancer_cli.py && echo "✅ enhancer_cli.py verified" || echo "❌ enhancer_cli.py missing"

# Set environment
ENV PYTHONPATH="/app"

CMD ["python", "rp_handler.py"]
