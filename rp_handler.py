#!/usr/bin/env python3
"""
RunPod Serverless Handler for GFPGAN Face Enhancement
Minimalist version - Direct subprocess call to enhancer_cli.py
Focus: Maximum performance, minimal overhead
"""

import runpod
import os
import tempfile
import uuid
import requests
import subprocess
import time
from minio import Minio
from urllib.parse import quote
from datetime import datetime

# Minimal logging
import logging
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
    """Download file from URL - optimized"""
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False

def upload_to_minio(local_path: str, object_name: str) -> str:
    """Upload file to MinIO - optimized"""
    try:
        minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
        return f"http://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise e

def run_gfpgan_enhancement(video_path: str, output_path: str) -> bool:
    """Run GFPGAN enhancement via subprocess call to enhancer_cli.py"""
    try:
        # Direct subprocess call to enhancer_cli.py
        cmd = [
            "python", "/app/enhancer_cli.py",
            "--video_path", video_path,
            "--enhancer", "gfpgan",
            "--output_path", output_path
        ]
        
        # Set environment for GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        
        # Run enhancer_cli.py - capture output for debugging only
        result = subprocess.run(
            cmd,
            cwd="/app",
            env=env,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        if result.returncode == 0:
            return True
        else:
            logger.error(f"GFPGAN failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("GFPGAN enhancement timed out")
        return False
    except Exception as e:
        logger.error(f"GFPGAN error: {e}")
        return False

def handler(job):
    """Main RunPod handler - minimalist approach"""
    job_id = job.get("id", "unknown")
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        video_url = job_input.get("video_url")
        
        if not video_url:
            return {"error": "Missing video_url"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # File paths
            video_path = os.path.join(temp_dir, "input.mp4")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(temp_dir, f"enhanced_{timestamp}.mp4")
            
            # Step 1: Download video
            if not download_file(video_url, video_path):
                return {"error": "Failed to download video"}
            
            # Step 2: Run GFPGAN via subprocess
            if not run_gfpgan_enhancement(video_path, output_path):
                return {"error": "GFPGAN enhancement failed"}
            
            # Check output exists
            if not os.path.exists(output_path):
                return {"error": "Enhancement output not found"}
            
            # Step 3: Upload result
            output_filename = f"gfpgan_enhanced_{job_id}_{uuid.uuid4().hex[:8]}.mp4"
            output_url = upload_to_minio(output_path, output_filename)
            
            processing_time = time.time() - start_time
            
            # Success response
            return {
                "status": "completed",
                "output_video_url": output_url,
                "processing_time_seconds": round(processing_time, 2),
                "enhancer_used": "gfpgan",
                "job_id": job_id
            }
            
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Job {job_id} failed: {e}")
        
        return {
            "status": "failed",
            "error": str(e),
            "processing_time_seconds": round(processing_time, 2),
            "job_id": job_id
        }

if __name__ == "__main__":
    logger.info("🚀 GFPGAN Face Enhancement Worker (Subprocess Mode)")
    
    # Verify enhancer_cli.py exists
    if not os.path.exists("/app/enhancer_cli.py"):
        logger.error("❌ enhancer_cli.py not found")
        exit(1)
    
    # Verify output directory
    if not os.path.exists("/app/outputs"):
        os.makedirs("/app/outputs")
    
    logger.info("✅ Ready to process requests")
    
    # Start RunPod worker
    runpod.serverless.start({"handler": handler})
