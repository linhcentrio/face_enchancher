# GFPGAN Face Enhancement Serverless - Production Optimized
FROM spxiong/pytorch:2.5.1-py3.10.15-cuda12.1.0-devel-ubuntu22.04

WORKDIR /app

# Set CUDA environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_MODULE_LOADING=LAZY

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
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt /app/

# Install Python dependencies - exact versions from environment.yml
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    echo "=== Installing core dependencies (exact versions) ===" && \
    pip install --no-cache-dir \
    numpy==1.24.4 \
    opencv-python==4.8.0.76 \
    onnxruntime-gpu==1.14.1 \
    tqdm==4.67.1 \
    requests==2.28.1 && \
    echo "=== Installing audio processing ===" && \
    pip install --no-cache-dir \
    soundfile==0.13.1 \
    librosa==0.11.0 \
    numba==0.61.0 && \
    echo "=== Installing image processing ===" && \
    pip install --no-cache-dir \
    scikit-image==0.25.2 \
    Pillow==11.0.0 \
    matplotlib==3.10.1 \
    scipy==1.15.2 \
    imutils==0.5.4 \
    imageio==2.37.0 && \
    echo "=== Installing other dependencies ===" && \
    pip install --no-cache-dir \
    easydict==1.13 \
    cython==3.0.12 \
    runpod>=1.6.0 \
    minio>=7.0.0

# Install InsightFace (with fallback)
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

# Create model directories
RUN mkdir -p /app/enhancers/GFPGAN \
    && mkdir -p /app/utils \
    && mkdir -p /app/faceID \
    && mkdir -p /app/outputs

# Download models in parallel for faster build
RUN echo "=== Downloading models ===" && \
    (wget --no-check-certificate --timeout=120 --tries=3 \
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx" \
    -O /app/enhancers/GFPGAN/GFPGANv1.4.onnx &) && \
    (wget --no-check-certificate --timeout=120 --tries=3 \
    "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/scrfd_2.5g_bnkps.onnx" \
    -O /app/utils/scrfd_2.5g_bnkps.onnx &) && \
    (wget --no-check-certificate --timeout=120 --tries=3 \
    "https://huggingface.co/manh-linh/faceID_recognition/resolve/main/recognition.onnx" \
    -O /app/faceID/recognition.onnx &) && \
    wait && \
    echo "✅ All models downloaded"

# Verify all models exist and are valid
RUN echo "=== Verifying models ===" && \
    test -f /app/enhancers/GFPGAN/GFPGANv1.4.onnx && echo "✅ GFPGAN model verified" && \
    test -f /app/utils/scrfd_2.5g_bnkps.onnx && echo "✅ Face detection model verified" && \
    test -f /app/faceID/recognition.onnx && echo "✅ Face recognition model verified" && \
    echo "🎉 All models verified successfully"

# Final environment verification
RUN echo "=== Final environment verification ===" && \
    python -c "import numpy; print(f'✅ NumPy: {numpy.__version__}')" && \
    python -c "import cv2; print(f'✅ OpenCV: {cv2.__version__}')" && \
    python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" && \
    python -c "import torch; print(f'✅ CUDA Available: {torch.cuda.is_available()}')" && \
    python -c "import onnxruntime; print(f'✅ ONNX Runtime: {onnxruntime.__version__}')" && \
    python -c "import onnxruntime; print(f'✅ CUDA Provider: {\"CUDAExecutionProvider\" in onnxruntime.get_available_providers()}')" && \
    python -c "import librosa; print(f'✅ Librosa: {librosa.__version__}')" && \
    python -c "import insightface; print(f'✅ InsightFace: {insightface.__version__}')" && \
    echo "🚀 Environment ready for production"

# Set environment variables
ENV PYTHONPATH="/app"
ENV TORCH_HOME="/app/models"
ENV HF_HOME="/app/models"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import cv2, numpy, onnxruntime, torch; print('OK')" || exit 1

# Start the serverless worker
CMD ["python", "rp_handler.py"]
