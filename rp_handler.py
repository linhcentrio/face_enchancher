#!/usr/bin/env python3
"""
RunPod Serverless Handler cho GFPGAN Video Face Enhancement
- Tích hợp MinIO (biến môi trường)
- Gọi subprocess tới enhancer_cli.py (dùng GFPGAN)
- Tối ưu hiệu suất, log rõ ràng, kiểm tra lỗi
"""

import os
import sys
import tempfile
import uuid
import requests
import logging
import subprocess
import time
from pathlib import Path
from minio import Minio
from urllib.parse import quote
from datetime import datetime
import runpod

# ---------------------- Cấu hình MinIO ----------------------
MINIO_ENDPOINT   = os.environ.get("MINIO_ENDPOINT",   "127.0.0.1:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET     = os.environ.get("MINIO_BUCKET",     "aiclipdfl")
MINIO_SECURE     = bool(os.environ.get("MINIO_SECURE", "0") == "1")

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE
)

# ---------------------- Logging ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------- Tiện ích ----------------------
def download_file(url: str, local_path: str) -> bool:
    """Tải file từ URL về local_path."""
    try:
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        file_size = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"Downloaded: {file_size:.1f} MB")
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False

def upload_to_minio(local_path: str, object_name: str) -> str:
    """Upload file lên MinIO, trả về URL."""
    try:
        minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
        file_url = f"http://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
        logger.info(f"Uploaded to MinIO: {file_url}")
        return file_url
    except Exception as e:
        logger.error(f"Upload to MinIO failed: {e}")
        raise

# ---------------------- Xử lý chính ----------------------
def handler(job):
    """Xử lý 1 job RunPod: tải video, enhance, upload MinIO, trả về URL."""
    job_id = job.get("id", "unknown")
    start_time = time.time()
    try:
        job_input = job.get("input", {})
        video_url = job_input.get("video_url")
        enhancer_name = job_input.get("enhancer", "gfpgan")  # default GFPGAN

        if not video_url:
            return {"error": "Missing video_url parameter"}

        logger.info(f"🚀 Job {job_id}: {enhancer_name.upper()} Face Enhancement")
        logger.info(f"📺 Video URL: {video_url}")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Đường dẫn file
            video_path = os.path.join(temp_dir, "input.mp4")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(temp_dir, f"enhanced_{timestamp}.mp4")

            # Bước 1: Tải video
            logger.info("📥 Step 1/3: Downloading video...")
            if not download_file(video_url, video_path):
                return {"error": "Failed to download input video"}
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                return {"error": "Downloaded video file is empty or corrupted"}

            # Bước 2: Enhance video bằng subprocess tới enhancer_cli.py
            logger.info(f"✨ Step 2/3: Enhancing video with {enhancer_name.upper()} (subprocess)...")
            try:
                result = subprocess.run(
                    [
                        sys.executable, "enhancer_cli.py",
                        "--video_path", video_path,
                        "--enhancer", enhancer_name,
                        "--output_path", output_path
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
                logger.info(result.stdout)
                if result.stderr:
                    logger.warning(result.stderr)
            except subprocess.CalledProcessError as e:
                logger.error(f"Enhancement subprocess failed: {e.stderr}")
                return {"error": f"Enhancement subprocess failed: {e.stderr}"}
            except Exception as e:
                logger.error(f"Enhancement failed: {e}")
                return {"error": f"Enhancement failed: {str(e)}"}

            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                return {"error": "Enhancement failed - no output file generated"}

            # Bước 3: Upload lên MinIO
            logger.info("📤 Step 3/3: Uploading enhanced video to MinIO...")
            output_filename = f"enhanced_{enhancer_name}_{job_id}_{uuid.uuid4().hex[:8]}.mp4"
            try:
                output_url = upload_to_minio(output_path, output_filename)
            except Exception as e:
                return {"error": f"Upload to MinIO failed: {str(e)}"}

            processing_time = time.time() - start_time
            response = {
                "status": "completed",
                "output_video_url": output_url,
                "processing_time_seconds": round(processing_time, 2),
                "enhancer_used": enhancer_name,
                "job_id": job_id,
                "file_size_mb": round(os.path.getsize(output_path) / (1024*1024), 2)
            }
            logger.info(f"✅ Job {job_id} completed successfully in {processing_time:.2f}s")
            return response

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Job {job_id} failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "processing_time_seconds": round(processing_time, 2),
            "job_id": job_id
        }

# ---------------------- Khởi động Worker ----------------------
if __name__ == "__main__":
    logger.info("🚀 Starting GFPGAN Face Enhancement Serverless Worker...")
    logger.info(f"🗄️ MinIO Storage: {MINIO_ENDPOINT}/{MINIO_BUCKET}")
    # Kiểm tra model, môi trường nếu cần ở đây
    runpod.serverless.start({"handler": handler})
