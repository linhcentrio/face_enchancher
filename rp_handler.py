#!/usr/bin/env python3
"""
RunPod Serverless Handler for GFPGAN Face Enhancement
GPU-optimized version with onnxruntime-gpu 1.14.1
"""

import runpod
import os
import tempfile
import uuid
import requests
import subprocess
import logging
from pathlib import Path
from minio import Minio
from urllib.parse import quote
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MinIO Configuration
MINIO_ENDPOINT = "108.181.198.160:9000"
MINIO_ACCESS_KEY = "a9TFRtBi8q3Nvj5P5Ris"
MINIO_SECRET_KEY = "fCFngM7YTr6jSkBKXZ9BkfDdXrStYXm43UGa0OZQ"
MINIO_BUCKET = "aiclipdfl"
MINIO_SECURE = False

# Initialize MinIO client
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE
)

def download_file(url: str, local_path: str) -> bool:
    """Download file from URL"""
    try:
        logger.info(f"📥 Downloading: {url}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"✅ Downloaded: {file_size:.1f} MB")
        return True
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False

def upload_to_minio(local_path: str, object_name: str) -> str:
    """Upload file to MinIO"""
    try:
        file_size = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"📤 Uploading: {file_size:.1f} MB")
        
        minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
        file_url = f"http://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
        
        logger.info(f"✅ Upload successful")
        return file_url
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise e

def run_gfpgan_enhancement(video_path: str, output_path: str) -> bool:
    """Run GFPGAN enhancement using enhancer_cli.py"""
    try:
        logger.info("✨ Running GFPGAN enhancement...")
        
        # Run enhancer_cli.py with GFPGAN
        cmd = [
            "python", "/app/enhancer_cli.py",
            "--video_path", video_path,
            "--enhancer", "gfpgan",
            "--output_path", output_path
        ]
        
        # Set environment variables for GPU usage
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes timeout
            env=env
        )
        
        if result.returncode == 0:
            logger.info("✅ GFPGAN enhancement completed successfully")
            return True
        else:
            logger.error(f"❌ GFPGAN enhancement failed:")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ GFPGAN enhancement timed out")
        return False
    except Exception as e:
        logger.error(f"❌ GFPGAN enhancement error: {e}")
        return False

def handler(job):
    """Main RunPod handler"""
    job_id = job.get("id", "unknown")
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        video_url = job_input.get("video_url")
        
        if not video_url:
            return {"error": "Missing video_url"}
        
        logger.info(f"🚀 Job {job_id}: GFPGAN Face Enhancement")
        logger.info(f"📺 Video: {video_url}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # File paths
            video_path = os.path.join(temp_dir, "input.mp4")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(temp_dir, f"enhanced_{timestamp}.mp4")
            
            # Step 1: Download video
            logger.info("📥 Step 1/3: Downloading video...")
            if not download_file(video_url, video_path):
                return {"error": "Failed to download video"}
            
            # Step 2: Run GFPGAN enhancement
            logger.info("✨ Step 2/3: Enhancing with GFPGAN...")
            if not run_gfpgan_enhancement(video_path, output_path):
                return {"error": "GFPGAN enhancement failed"}
            
            # Check if output exists
            if not os.path.exists(output_path):
                return {"error": "Enhancement output not found"}
            
            # Step 3: Upload result
            logger.info("📤 Step 3/3: Uploading result...")
            output_filename = f"gfpgan_enhanced_{job_id}_{uuid.uuid4().hex[:8]}.mp4"
            output_url = upload_to_minio(output_path, output_filename)
            
            processing_time = time.time() - start_time
            
            # Success response
            response = {
                "status": "completed",
                "output_video_url": output_url,
                "processing_time_seconds": round(processing_time, 2),
                "enhancer_used": "gfpgan",
                "job_id": job_id
            }
            
            logger.info(f"✅ Job {job_id} completed in {processing_time:.1f}s")
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

def verify_gpu_support():
    """Verify GPU support for ONNX Runtime"""
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        
        if 'CUDAExecutionProvider' in providers:
            logger.info("✅ CUDA Execution Provider available")
            return True
        else:
            logger.warning("⚠️ CUDA Execution Provider not available")
            logger.info(f"Available providers: {providers}")
            return False
            
    except Exception as e:
        logger.error(f"❌ GPU verification failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting GFPGAN Face Enhancement Worker (GPU-optimized)...")
    
    # Verify environment
    try:
        import numpy
        import cv2
        import torch
        import onnxruntime
        
        logger.info(f"✅ NumPy: {numpy.__version__}")
        logger.info(f"✅ OpenCV: {cv2.__version__}")
        logger.info(f"✅ PyTorch: {torch.__version__}")
        logger.info(f"✅ ONNX Runtime: {onnxruntime.__version__}")
        logger.info(f"✅ PyTorch CUDA: {torch.cuda.is_available()}")
        
        # Verify GPU support
        gpu_available = verify_gpu_support()
        if gpu_available:
            logger.info("🎮 GPU acceleration enabled")
        else:
            logger.warning("⚠️ Running on CPU mode")
            
    except Exception as e:
        logger.error(f"❌ Environment check failed: {e}")
    
    # Verify models exist
    required_models = [
        "/app/enhancers/GFPGAN/GFPGANv1.4.onnx",
        "/app/utils/scrfd_2.5g_bnkps.onnx",
        "/app/faceID/recognition.onnx"
    ]
    
    for model_path in required_models:
        if os.path.exists(model_path):
            logger.info(f"✅ Model found: {os.path.basename(model_path)}")
        else:
            logger.error(f"❌ Model missing: {model_path}")
    
    logger.info(f"🗄️ Storage: {MINIO_ENDPOINT}/{MINIO_BUCKET}")
    logger.info("🎬 Ready to process requests...")
    
    # Start RunPod worker
    runpod.serverless.start({"handler": handler})
