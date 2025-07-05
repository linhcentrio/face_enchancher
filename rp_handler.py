#!/usr/bin/env python3
"""
RunPod Serverless Handler for GFPGAN Face Enhancement
Production Optimized - EXACT Copy of enhancer_cli.py logic
Performance Target: 3.8-4.0 FPS (match local environment)
"""

import runpod
import os
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

# EXACT imports like enhancer_cli.py
import numpy as np
import cv2
import sys
from tqdm import tqdm
import onnxruntime

# Suppress ONNX warnings like original
onnxruntime.set_default_logger_severity(3)

# Add path for local modules
sys.path.append('/app')

# Import face processing modules
try:
    from utils.retinaface import RetinaFace
    from utils.face_alignment import get_cropped_head_256
    from faceID.faceID import FaceRecognition
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Configure minimal logging for performance
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
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

# GLOBAL model initialization like enhancer_cli.py - PERFORMANCE CRITICAL
print("🤖 Initializing global models...")

# Face detection model initialization - EXACT COPY
detector = RetinaFace(
    "/app/utils/scrfd_2.5g_bnkps.onnx",
    provider=[
        ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
        "CPUExecutionProvider"
    ],
    session_options=None
)

# Face recognition model initialization - EXACT COPY  
recognition = FaceRecognition('/app/faceID/recognition.onnx')

print("✅ Global models initialized")

def load_enhancer(enhancer_name, device):
    """Load the specified enhancer model - EXACT COPY"""
    if enhancer_name == 'gpen':
        from enhancers.GPEN.GPEN import GPEN
        return GPEN(model_path="/app/enhancers/GPEN/GPEN-BFR-256-sim.onnx", device=device)
    elif enhancer_name == 'codeformer':
        from enhancers.Codeformer.Codeformer import CodeFormer
        return CodeFormer(model_path="/app/enhancers/Codeformer/codeformerfixed.onnx", device=device)
    elif enhancer_name == 'restoreformer':
        from enhancers.restoreformer.restoreformer16 import RestoreFormer
        return RestoreFormer(model_path="/app/enhancers/restoreformer/restoreformer16.onnx", device=device)
    elif enhancer_name == 'gfpgan':
        from enhancers.GFPGAN.GFPGAN import GFPGAN
        return GFPGAN(model_path="/app/enhancers/GFPGAN/GFPGANv1.4.onnx", device=device)
    else:
        raise ValueError(f"Unknown enhancer: {enhancer_name}")

def process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height):
    """Process a batch of frames - EXACT COPY (NO ThreadPoolExecutor for performance)"""
    frames, aligned_faces, mats = zip(*frame_buffer)

    # Enhance faces in batch
    enhanced_faces = enhancer.enhance_batch(aligned_faces)

    # Simple for loop - NO threading overhead like original
    for frame, aligned_face, mat, enhanced_face in zip(frames, aligned_faces, mats, enhanced_faces):
        # Resize enhanced face back to the original size of aligned face
        enhanced_face_resized = cv2.resize(enhanced_face, (aligned_face.shape[1], aligned_face.shape[0]))

        # Resize face mask to match the size of enhanced face
        face_mask_resized = cv2.resize(face_mask, (enhanced_face_resized.shape[1], enhanced_face_resized.shape[0]))

        # Blend enhanced face back into the original frame
        blended_face = (face_mask_resized * enhanced_face_resized + (1 - face_mask_resized) * aligned_face).astype(np.uint8)

        # Warp blended face back to original frame
        mat_rev = cv2.invertAffineTransform(mat)
        dealigned_face = cv2.warpAffine(blended_face, mat_rev, (frame_width, frame_height))

        mask = cv2.warpAffine(face_mask_resized, mat_rev, (frame_width, frame_height))
        final_frame = (mask * dealigned_face + (1 - mask) * frame).astype(np.uint8)

        out.write(final_frame)

def enhance_video(video_path, enhancer_name, output_path=None):
    """Enhance the video - EXACT COPY of enhancer_cli.py"""
    device = 'cpu'
    if onnxruntime.get_device() == 'GPU':
        device = 'cuda'

    print(f"Running on {device}")

    # Load enhancer
    enhancer = load_enhancer(enhancer_name, device)

    # Open the video stream
    video_stream = cv2.VideoCapture(video_path)
    if not video_stream.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    # Get video properties
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path is None:
        output_path = os.path.join('outputs', os.path.basename(video_path).replace('.', f'_{enhancer_name}.'))

    # Temporary video file without audio
    temp_video_path = output_path.replace('.', '_temp.')

    # Create video writer with correct FPS
    out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Precompute face mask
    face_mask = np.zeros((256, 256), dtype=np.uint8)
    face_mask = cv2.rectangle(face_mask, (66, 69), (190, 240), (255, 255, 255), -1)
    face_mask = cv2.GaussianBlur(face_mask.astype(np.uint8), (19, 19), cv2.BORDER_DEFAULT)
    face_mask = cv2.cvtColor(face_mask, cv2.COLOR_GRAY2RGB)
    face_mask = face_mask / 255

    batch_size = 1  # GFPGAN batch size is always 1
    frame_buffer = []

    # Simple frame processing loop - NO chunking for performance
    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = video_stream.read()
        if not ret:
            break

        # Detect and align face
        bboxes, kpss = detector.detect(frame, input_size=(320, 320), det_thresh=0.3)
        if len(kpss) == 0:
            out.write(frame)
            continue

        aligned_face, mat = get_cropped_head_256(frame, kpss[0], size=256, scale=1.0)
        frame_buffer.append((frame, aligned_face, mat))

        if len(frame_buffer) >= batch_size:
            process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height)
            frame_buffer = []

    # Process remaining frames in the buffer
    if frame_buffer:
        process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height)

    video_stream.release()
    out.release()
    print(f"Enhanced video frames saved to {temp_video_path}")

    # Extract audio from the original video
    audio_path = os.path.splitext(output_path)[0] + '.aac'
    try:
        subprocess.call([
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'aac', '-b:a', '192k', audio_path
        ])
    except Exception as e:
        print(f"Error extracting audio: {e}")
        raise

    # Combine the enhanced video frames with the original audio
    try:
        subprocess.call([
            'ffmpeg', '-y', '-i', temp_video_path, '-i', audio_path,
            '-c:v', 'libx264', '-crf', '23', '-preset', 'medium',
            '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', output_path
        ])
    except Exception as e:
        print(f"Error combining video and audio: {e}")
        raise

    # Remove temporary files
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    if os.path.exists(audio_path):
        os.remove(audio_path)

    print(f"Enhanced video with original audio saved to {output_path}")

def download_file(url: str, local_path: str) -> bool:
    """Download file from URL with minimal logging"""
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size = os.path.getsize(local_path) / (1024 * 1024)
        print(f"Downloaded: {file_size:.1f} MB")
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False

def upload_to_minio(local_path: str, object_name: str) -> str:
    """Upload file to MinIO storage"""
    try:
        file_size = os.path.getsize(local_path) / (1024 * 1024)
        print(f"Uploading: {file_size:.1f} MB")
        
        minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
        file_url = f"http://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
        
        print("Upload successful")
        return file_url
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise e

def handler(job):
    """Main RunPod handler - optimized for performance"""
    job_id = job.get("id", "unknown")
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        video_url = job_input.get("video_url")
        
        if not video_url:
            return {"error": "Missing video_url parameter"}
        
        # Optional parameters
        enhancer_name = job_input.get("enhancer", "gfpgan")  # Default to GFPGAN
        
        logger.info(f"🚀 Job {job_id}: {enhancer_name.upper()} Face Enhancement")
        logger.info(f"📺 Video URL: {video_url}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # File paths
            video_path = os.path.join(temp_dir, "input.mp4")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(temp_dir, f"enhanced_{timestamp}.mp4")
            
            # Step 1: Download input video
            logger.info("📥 Step 1/3: Downloading video...")
            if not download_file(video_url, video_path):
                return {"error": "Failed to download input video"}
            
            # Verify video file
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                return {"error": "Downloaded video file is empty or corrupted"}
            
            # Step 2: Enhance video using EXACT enhancer_cli.py logic
            logger.info(f"✨ Step 2/3: Enhancing video with {enhancer_name.upper()}...")
            
            try:
                # Call the enhance_video function - EXACT like enhancer_cli.py
                enhance_video(video_path, enhancer_name, output_path)
                
                if not os.path.exists(output_path):
                    return {"error": "Enhancement failed - no output file generated"}
                
                # Check output file size
                output_size = os.path.getsize(output_path)
                if output_size == 0:
                    return {"error": "Enhancement failed - output file is empty"}
                
                logger.info(f"✅ Enhancement completed: {output_size / (1024*1024):.1f} MB")
                
            except Exception as e:
                logger.error(f"❌ Enhancement failed: {e}")
                return {"error": f"Enhancement processing failed: {str(e)}"}
            
            # Step 3: Upload result to MinIO
            logger.info("📤 Step 3/3: Uploading enhanced video...")
            
            try:
                output_filename = f"enhanced_{enhancer_name}_{job_id}_{uuid.uuid4().hex[:8]}.mp4"
                output_url = upload_to_minio(output_path, output_filename)
                
            except Exception as e:
                logger.error(f"❌ Upload failed: {e}")
                return {"error": f"Upload failed: {str(e)}"}
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare success response
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

if __name__ == "__main__":
    logger.info("🚀 Starting GFPGAN Face Enhancement Serverless Worker...")
    logger.info("🎯 Performance Target: 3.8-4.0 FPS (optimized)")
    
    # Environment verification
    try:
        logger.info(f"🐍 Python: {sys.version}")
        
        import torch
        logger.info(f"🔥 PyTorch: {torch.__version__}")
        logger.info(f"⚡ CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"🎮 GPU: {torch.cuda.get_device_name()}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        
        logger.info(f"🔧 ONNX Runtime: {onnxruntime.__version__}")
        providers = onnxruntime.get_available_providers()
        logger.info(f"🎯 ONNX Providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            logger.info("✅ GPU acceleration enabled")
        else:
            logger.warning("⚠️ Running on CPU mode")
            
    except Exception as e:
        logger.warning(f"⚠️ Environment check failed: {e}")
    
    # Verify model files exist
    required_models = [
        "/app/enhancers/GFPGAN/GFPGANv1.4.onnx",
        "/app/utils/scrfd_2.5g_bnkps.onnx",
        "/app/faceID/recognition.onnx"
    ]
    
    models_ok = True
    for model_path in required_models:
        if os.path.exists(model_path):
            model_size = os.path.getsize(model_path) / (1024*1024)
            logger.info(f"✅ Model verified: {os.path.basename(model_path)} ({model_size:.1f} MB)")
        else:
            logger.error(f"❌ Model missing: {model_path}")
            models_ok = False
    
    if not models_ok:
        logger.error("❌ Required models missing. Exiting...")
        sys.exit(1)
    
    logger.info(f"🗄️ Storage: {MINIO_ENDPOINT}/{MINIO_BUCKET}")
    logger.info("🎬 Ready to process face enhancement requests...")
    
    # Start RunPod serverless worker
    runpod.serverless.start({"handler": handler})
