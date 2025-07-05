FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Thiết lập biến môi trường CUDA
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Cài đặt các phụ thuộc hệ thống
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

# Sao chép file môi trường và cài đặt Python dependencies
COPY environment.yml /app/environment.yml
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn
COPY . /app/

# Tạo thư mục và tải model (tối ưu: chỉ tải nếu chưa tồn tại)
RUN mkdir -p /app/enhancers/GFPGAN /app/utils /app/faceID /app/outputs && \
    [ -f /app/enhancers/GFPGAN/GFPGANv1.4.onnx ] || wget -O /app/enhancers/GFPGAN/GFPGANv1.4.onnx "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx" && \
    [ -f /app/utils/scrfd_2.5g_bnkps.onnx ] || wget -O /app/utils/scrfd_2.5g_bnkps.onnx "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/scrfd_2.5g_bnkps.onnx" && \
    [ -f /app/faceID/recognition.onnx ] || wget -O /app/faceID/recognition.onnx "https://huggingface.co/manh-linh/faceID_recognition/resolve/main/recognition.onnx"

ENV PYTHONPATH="/app"

CMD ["python", "rp_handler.py"]
