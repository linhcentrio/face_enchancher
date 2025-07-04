# Face Enhancement with GFPGAN - Fixed PyTorch CUDA compatibility
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
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app/

# Install exact versions from environment.yaml (excluding PyTorch)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    echo "=== Installing core dependencies ===" && \
    pip install --no-cache-dir \
    numpy==1.24.4 \
    opencv-python==4.8.0.76 \
    opencv-contrib-python==4.11.0.86 \
    opencv-python-headless==4.11.0.86 && \
    echo "=== Installing ONNX Runtime GPU ===" && \
    pip install --no-cache-dir onnxruntime-gpu==1.14.1 && \
    echo "=== Installing audio processing ===" && \
    pip install --no-cache-dir \
    soundfile==0.13.1 \
    soxr==0.5.0.post1 \
    audioread==3.0.1 \
    numba==0.61.0 \
    librosa==0.11.0 && \
    echo "=== Installing image processing ===" && \
    pip install --no-cache-dir \
    Pillow==11.0.0 \
    scikit-image==0.25.2 \
    matplotlib==3.10.1 \
    scipy==1.15.2 \
    imageio==2.37.0 \
    imutils==0.5.4 && \
    echo "=== Installing other dependencies ===" && \
    pip install --no-cache-dir \
    tqdm==4.67.1 \
    requests==2.28.1 \
    easydict==1.13 \
    cython==3.0.12 \
    ffmpeg-python==0.2.0 \
    joblib==1.4.2 \
    networkx==3.3 \
    packaging==24.2 \
    python-dateutil==2.9.0.post0 \
    pyyaml==6.0.2 \
    six==1.17.0 \
    sympy==1.13.1 \
    typing-extensions==4.12.2 \
    urllib3==1.26.13 && \
    echo "=== Installing RunPod dependencies ===" && \
    pip install --no-cache-dir \
    runpod>=1.6.0 \
    minio>=7.0.0

# Use PyTorch from base image (2.5.1 with CUDA 12.1 support)
# Skip installing PyTorch 2.4.0+cu118 to avoid CUDA version conflict
RUN echo "=== Using PyTorch from base image (CUDA 12.1 compatible) ===" && \
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" && \
    python -c "import torch; print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Install InsightFace (exact version from environment.yaml)
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
    echo "✅ GFPGAN model downloaded" && \
    wget --no-check-certificate --timeout=60 --tries=3 \
    "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/scrfd_2.5g_bnkps.onnx" \
    -O /app/utils/scrfd_2.5g_bnkps.onnx && \
    echo "✅ Face detection model downloaded" && \
    wget --no-check-certificate --timeout=60 --tries=3 \
    "https://huggingface.co/manh-linh/faceID_recognition/resolve/main/recognition.onnx" \
    -O /app/faceID/recognition.onnx && \
    echo "✅ Face recognition model downloaded"

# Final verification
RUN echo "=== Final environment verification ===" && \
    python -c "import numpy; print(f'✅ NumPy: {numpy.__version__}')" && \
    python -c "import cv2; print(f'✅ OpenCV: {cv2.__version__}')" && \
    python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" && \
    python -c "import torch; print(f'✅ PyTorch CUDA: {torch.cuda.is_available()}')" && \
    python -c "import onnxruntime; print(f'✅ ONNX Runtime: {onnxruntime.__version__}')" && \
    python -c "import onnxruntime; print(f'✅ ONNX GPU Providers: {onnxruntime.get_available_providers()}')" && \
    python -c "import librosa; print(f'✅ Librosa: {librosa.__version__}')" && \
    python -c "import insightface; print(f'✅ InsightFace: {insightface.__version__}')" && \
    echo "🎮 GPU Status:" && \
    python -c "import onnxruntime; print(f'  ONNX GPU: {\"CUDAExecutionProvider\" in onnxruntime.get_available_providers()}')" && \
    python -c "import torch; print(f'  PyTorch GPU: {torch.cuda.is_available()}')" && \
    echo "✅ Environment setup completed"

# Set environment
ENV PYTHONPATH="/app"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import cv2, numpy, onnxruntime; print('OK')" || exit 1

CMD ["python", "rp_handler.py"]
