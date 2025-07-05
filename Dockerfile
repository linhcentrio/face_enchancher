# Sử dụng base image PyTorch chuẩn, hỗ trợ CUDA 11.8 và Python 3.8
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Thiết lập noninteractive để tránh prompt khi cài tzdata
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-dev \
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
        git \
    && rm -rf /var/lib/apt/lists/*

# Sao chép requirements.txt và cài đặt Python packages
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . /app/

# Tạo thư mục chứa model và tải model nếu chưa có
RUN mkdir -p /app/enhancers/GFPGAN /app/utils /app/faceID /app/outputs && \
    if [ ! -f /app/enhancers/GFPGAN/GFPGANv1.4.onnx ]; then \
        wget -O /app/enhancers/GFPGAN/GFPGANv1.4.onnx "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx"; \
    fi && \
    if [ ! -f /app/utils/scrfd_2.5g_bnkps.onnx ]; then \
        wget -O /app/utils/scrfd_2.5g_bnkps.onnx "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/scrfd_2.5g_bnkps.onnx"; \
    fi && \
    if [ ! -f /app/faceID/recognition.onnx ]; then \
        wget -O /app/faceID/recognition.onnx "https://huggingface.co/manh-linh/faceID_recognition/resolve/main/recognition.onnx"; \
    fi

# Thiết lập biến môi trường CUDA và Python
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

# Lệnh mặc định khi chạy container (có thể sửa thành enhancer_cli.py nếu muốn)
CMD ["python", "rp_handler.py"]
