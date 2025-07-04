# Face Enhancement with GFPGAN - Fixed NumPy compatibility
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
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt /app/

# Install Python dependencies with specific versions to avoid conflicts
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --no-cache-dir \
    numpy==1.26.4 \
    opencv-python==4.8.0.76 && \
    pip install --no-cache-dir -r requirements.txt

# Install InsightFace separately to handle potential conflicts
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install insightface==0.7.3 --no-cache-dir || \
    (wget --no-check-certificate --timeout=30 --tries=3 \
    "https://huggingface.co/deauxpas/colabrepo/resolve/main/insightface-0.7.3-cp310-cp310-linux_x86_64.whl" \
    -O /tmp/insightface.whl && \
    pip install /tmp/insightface.whl --force-reinstall && \
    rm -f /tmp/insightface.whl)

# Copy source code
COPY . /app/

# Create directories
RUN mkdir -p /app/enhancers/GFPGAN \
    && mkdir -p /app/utils \
    && mkdir -p /app/faceID \
    && mkdir -p /app/outputs

# Download models
RUN echo "=== Downloading models ===" && \
    wget --no-check-certificate --timeout=60 --tries=3 \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx" \
    -O /app/enhancers/GFPGAN/GFPGANv1.4.onnx && \
    echo "✅ GFPGAN downloaded" && \
    wget --no-check-certificate --timeout=60 --tries=3 \
    "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/scrfd_2.5g_bnkps.onnx" \
    -O /app/utils/scrfd_2.5g_bnkps.onnx && \
    echo "✅ Face detection downloaded" && \
    wget --no-check-certificate --timeout=60 --tries=3 \
    "https://huggingface.co/manh-linh/faceID_recognition/resolve/main/recognition.onnx" \
    -O /app/faceID/recognition.onnx && \
    echo "✅ Face recognition downloaded"

# Verify installations
RUN echo "=== Verifying installations ===" && \
    python -c "import numpy; print(f'NumPy: {numpy.__version__}')" && \
    python -c "import cv2; print(f'OpenCV: {cv2.__version__}')" && \
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python -c "import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}')" && \
    echo "✅ All packages verified"

# Set environment
ENV PYTHONPATH="/app"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import cv2, numpy; print('OK')" || exit 1

CMD ["python", "rp_handler.py"]
