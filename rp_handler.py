#!/usr/bin/env python3
"""
RunPod Serverless Handler for Face Enhancement
Supports multiple enhancement models: GFPGAN, GPEN, CodeFormer, RestoreFormer
"""

import runpod
import os
import tempfile
import uuid
import requests
import time
import cv2
import numpy as np
import torch
import sys
import logging
import gc
import subprocess
from pathlib import Path
from minio import Minio
from urllib.parse import quote
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import onnxruntime

# Suppress ONNX runtime warnings
onnxruntime.set_default_logger_severity(3)

# Add path for local modules
sys.path.append('/app')

# Import required modules
try:
    from utils.retinaface import RetinaFace
    from utils.face_alignment import get_cropped_head_256
    from faceID.faceID import FaceRecognition
except ImportError as e:
    logging.error(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced model configurations
ENHANCER_CONFIGS = {
    "gfpgan": {
        "model_path": "/app/enhancers/GFPGAN/GFPGANv1.4.onnx",
        "class_name": "GFPGAN",
        "module_path": "enhancers.GFPGAN.GFPGAN",
        "recommended_batch_size": 1,
        "description": "General Face Restoration - Best for overall quality"
    },
    "gpen": {
        "model_path": "/app/enhancers/GPEN/GPEN-BFR-256-sim.onnx",
        "class_name": "GPEN",
        "module_path": "enhancers.GPEN.GPEN",
        "recommended_batch_size": 8,
        "description": "GAN Prior Enhanced Network - Best for natural results"
    },
    "codeformer": {
        "model_path": "/app/enhancers/Codeformer/codeformerfixed.onnx",
        "class_name": "CodeFormer",
        "module_path": "enhancers.Codeformer.Codeformer",
        "recommended_batch_size": 8,
        "description": "Transformer-based - Best for severely degraded faces"
    },
    "restoreformer": {
        "model_path": "/app/enhancers/restoreformer/restoreformer16.onnx",
        "class_name": "RestoreFormer",
        "module_path": "enhancers.restoreformer.restoreformer16",
        "recommended_batch_size": 16,
        "description": "Transformer Enhanced - Best for high-resolution results"
    }
}

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

# Global model instances
detector = None
recognition = None
enhancer_cache = {}

def validate_enhancer(enhancer_name: str) -> str:
    """Validate and normalize enhancer name"""
    if enhancer_name.lower() in ENHANCER_CONFIGS:
        return enhancer_name.lower()
    else:
        logger.warning(f"Invalid enhancer '{enhancer_name}', defaulting to gfpgan")
        return "gfpgan"

def get_enhancer_config(enhancer_name: str) -> dict:
    """Get configuration for specified enhancer"""
    enhancer_name = validate_enhancer(enhancer_name)
    return ENHANCER_CONFIGS[enhancer_name]

def initialize_face_detection():
    """Initialize face detection and recognition models"""
    global detector, recognition
    
    try:
        # Initialize face detector
        detector_path = "/app/utils/scrfd_2.5g_bnkps.onnx"
        if not os.path.exists(detector_path):
            raise FileNotFoundError(f"Face detector model not found: {detector_path}")
        
        detector = RetinaFace(
            detector_path,
            provider=[
                ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
                "CPUExecutionProvider"
            ],
            session_options=None
        )
        logger.info("✅ Face detector initialized")
        
        # Initialize face recognition
        recognition_path = "/app/faceID/recognition.onnx"
        if not os.path.exists(recognition_path):
            raise FileNotFoundError(f"Face recognition model not found: {recognition_path}")
        
        recognition = FaceRecognition(recognition_path)
        logger.info("✅ Face recognition initialized")
        
    except Exception as e:
        logger.error(f"❌ Face detection initialization failed: {e}")
        raise e

def load_enhancer(enhancer_name: str, device: str):
    """Load enhancer model dynamically"""
    global enhancer_cache
    
    # Check if enhancer is already loaded
    if enhancer_name in enhancer_cache:
        logger.info(f"✅ Using cached {enhancer_name} enhancer")
        return enhancer_cache[enhancer_name]
    
    try:
        config = get_enhancer_config(enhancer_name)
        model_path = config["model_path"]
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Enhancer model not found: {model_path}")
        
        # Dynamic import
        module_path = config["module_path"]
        class_name = config["class_name"]
        
        module = __import__(module_path, fromlist=[class_name])
        enhancer_class = getattr(module, class_name)
        
        # Initialize enhancer
        enhancer = enhancer_class(model_path=model_path, device=device)
        enhancer_cache[enhancer_name] = enhancer
        
        logger.info(f"✅ {enhancer_name.upper()} enhancer loaded on {device}")
        return enhancer
        
    except Exception as e:
        logger.error(f"❌ Failed to load {enhancer_name} enhancer: {e}")
        raise e

def download_file(url: str, local_path: str) -> bool:
    """Download file from URL with progress tracking"""
    try:
        logger.info(f"📥 Downloading {url}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        logger.info(f"✅ Downloaded: {local_path} ({downloaded/1024/1024:.1f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False

def upload_to_minio(local_path: str, object_name: str) -> str:
    """Upload file to MinIO with enhanced error handling"""
    try:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
        file_url = f"http://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
        
        logger.info(f"✅ Uploaded successfully: {file_url}")
        return file_url
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise e

def process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height):
    """Process batch of frames with face enhancement"""
    frames, aligned_faces, mats = zip(*frame_buffer)
    
    # Convert to numpy array for batch processing
    aligned_faces_array = np.array(aligned_faces)
    
    # Enhance faces in batch
    enhanced_faces = enhancer.enhance_batch(aligned_faces_array)
    
    # Process each frame
    def process_single_frame(data):
        frame, aligned_face, mat, enhanced_face = data
        
        # Resize enhanced face
        enhanced_face_resized = cv2.resize(enhanced_face, (aligned_face.shape[1], aligned_face.shape[0]))
        
        # Resize face mask
        face_mask_resized = cv2.resize(face_mask, (enhanced_face_resized.shape[1], enhanced_face_resized.shape[0]))
        
        # Blend enhanced face with original
        blended_face = (face_mask_resized * enhanced_face_resized + 
                       (1 - face_mask_resized) * aligned_face).astype(np.uint8)
        
        # Warp face back to original position
        mat_rev = cv2.invertAffineTransform(mat)
        dealigned_face = cv2.warpAffine(blended_face, mat_rev, (frame_width, frame_height))
        
        # Apply mask and blend with original frame
        mask = cv2.warpAffine(face_mask_resized, mat_rev, (frame_width, frame_height))
        final_frame = (mask * dealigned_face + (1 - mask) * frame).astype(np.uint8)
        
        return final_frame
    
    # Process frames with threading
    frame_data = zip(frames, aligned_faces, mats, enhanced_faces)
    
    with ThreadPoolExecutor() as executor:
        final_frames = list(executor.map(process_single_frame, frame_data))
    
    # Write processed frames
    for final_frame in final_frames:
        out.write(final_frame)

def enhance_video(video_path: str, enhancer_name: str, output_path: str, batch_size: int = 0) -> tuple[bool, dict]:
    """Enhanced video processing with specified enhancer"""
    global detector, recognition
    
    stats = {
        "total_frames": 0,
        "frames_with_faces": 0,
        "frames_without_faces": 0,
        "faces_enhanced": 0,
        "enhancement_applied": False,
        "enhancer_used": enhancer_name
    }
    
    try:
        # Determine device
        device = 'cuda' if torch.cuda.is_available() and onnxruntime.get_device() == 'GPU' else 'cpu'
        logger.info(f"🔥 Running on {device}")
        
        # Load enhancer
        enhancer = load_enhancer(enhancer_name, device)
        
        # Open video stream
        video_stream = cv2.VideoCapture(video_path)
        if not video_stream.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        # Get video properties
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        
        stats["total_frames"] = total_frames
        
        # Create temporary video file
        temp_video_path = output_path.replace('.mp4', '_temp.mp4')
        out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (frame_width, frame_height))
        
        # Create face mask for blending
        face_mask = np.zeros((256, 256), dtype=np.uint8)
        face_mask = cv2.rectangle(face_mask, (66, 69), (190, 240), (255, 255, 255), -1)
        face_mask = cv2.GaussianBlur(face_mask.astype(np.uint8), (19, 19), cv2.BORDER_DEFAULT)
        face_mask = cv2.cvtColor(face_mask, cv2.COLOR_GRAY2RGB)
        face_mask = face_mask / 255
        
        # Determine optimal batch size
        if batch_size <= 0:
            config = get_enhancer_config(enhancer_name)
            if device == 'cuda':
                batch_size = config["recommended_batch_size"]
            else:
                batch_size = 1
        
        logger.info(f"Using batch size: {batch_size}")
        
        # Process video in chunks for memory optimization
        MAX_FRAMES_IN_MEMORY = 1000
        
        for frame_idx in tqdm(range(0, total_frames, MAX_FRAMES_IN_MEMORY), desc="Processing chunks"):
            chunk_frames = min(MAX_FRAMES_IN_MEMORY, total_frames - frame_idx)
            frame_buffer = []
            
            for _ in tqdm(range(chunk_frames), desc="Processing frames", leave=False):
                ret, frame = video_stream.read()
                if not ret:
                    break
                
                # Detect faces
                try:
                    bboxes, kpss = detector.detect(frame, input_size=(320, 320), det_thresh=0.3)
                except Exception as e:
                    logger.debug(f"Face detection failed: {e}")
                    bboxes, kpss = [], []
                
                if len(kpss) == 0:
                    # No face detected, keep original frame
                    stats["frames_without_faces"] += 1
                    out.write(frame)
                    continue
                
                # Face detected, process with enhancer
                stats["frames_with_faces"] += 1
                
                try:
                    aligned_face, mat = get_cropped_head_256(frame, kpss[0], size=256, scale=1.0)
                    frame_buffer.append((frame, aligned_face, mat))
                    
                    if len(frame_buffer) >= batch_size:
                        process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height)
                        stats["faces_enhanced"] += len(frame_buffer)
                        frame_buffer = []
                        
                except Exception as e:
                    logger.debug(f"Face processing failed: {e}")
                    # If face processing fails, keep original frame
                    out.write(frame)
                    stats["frames_with_faces"] -= 1
                    stats["frames_without_faces"] += 1
            
            # Process remaining frames
            if frame_buffer:
                process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height)
                stats["faces_enhanced"] += len(frame_buffer)
            
            # Memory cleanup
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        video_stream.release()
        out.release()
        
        if stats["frames_with_faces"] > 0:
            stats["enhancement_applied"] = True
            logger.info(f"✅ Face enhancement applied to {stats['frames_with_faces']}/{stats['total_frames']} frames")
        else:
            logger.info("ℹ️ No faces detected in any frame")
            stats["enhancement_applied"] = False
        
        # Extract and combine audio
        audio_path = os.path.splitext(output_path)[0] + '.aac'
        
        # Extract audio
        subprocess.run([
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'aac', '-b:a', '192k', audio_path
        ], check=True, capture_output=True)
        
        # Combine video and audio
        subprocess.run([
            'ffmpeg', '-y', '-i', temp_video_path, '-i', audio_path,
            '-c:v', 'libx264', '-crf', '23', '-preset', 'medium',
            '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', output_path
        ], check=True, capture_output=True)
        
        # Cleanup temporary files
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        logger.info(f"✅ Video enhancement completed: {output_path}")
        return True, stats
        
    except Exception as e:
        logger.error(f"❌ Video enhancement failed: {e}")
        return False, stats
    finally:
        if 'video_stream' in locals():
            video_stream.release()
        if 'out' in locals():
            out.release()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def handler(job):
    """Main RunPod handler for face enhancement"""
    job_id = job.get("id", "unknown")
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        video_url = job_input.get("video_url")
        
        if not video_url:
            return {"error": "Missing video_url"}
        
        # Enhancement parameters
        enhancer_name = validate_enhancer(job_input.get("enhancer", "gfpgan"))
        batch_size = job_input.get("batch_size", 0)
        
        enhancer_config = get_enhancer_config(enhancer_name)
        
        logger.info(f"🚀 Job {job_id}: Face Enhancement with {enhancer_name.upper()}")
        logger.info(f"📺 Video: {video_url}")
        logger.info(f"🎨 Enhancer: {enhancer_config['description']}")
        logger.info(f"⚙️ Batch size: {batch_size if batch_size > 0 else 'auto'}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "input_video.mp4")
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(temp_dir, f"enhanced_{enhancer_name}_{current_time}.mp4")
            
            # Step 1: Download input video
            logger.info("📥 Step 1/3: Downloading input video...")
            if not download_file(video_url, video_path):
                return {"error": "Failed to download video"}
            
            # Step 2: Enhance video
            logger.info(f"✨ Step 2/3: Enhancing video with {enhancer_name.upper()}...")
            enhancement_success, stats = enhance_video(video_path, enhancer_name, output_path, batch_size)
            
            if not enhancement_success:
                return {"error": f"Face enhancement with {enhancer_name} failed"}
            
            if not os.path.exists(output_path):
                return {"error": "Enhancement output not generated"}
            
            # Step 3: Upload result
            logger.info("📤 Step 3/3: Uploading result...")
            output_filename = f"enhanced_{enhancer_name}_{job_id}_{uuid.uuid4().hex[:8]}.mp4"
            output_url = upload_to_minio(output_path, output_filename)
            
            processing_time = time.time() - start_time
            
            # Prepare response
            response = {
                "output_video_url": output_url,
                "processing_time_seconds": round(processing_time, 2),
                "enhancer_used": enhancer_name,
                "status": "completed",
                "enhancement_stats": {
                    "total_frames": stats["total_frames"],
                    "frames_enhanced": stats["frames_with_faces"],
                    "enhancement_rate": round(stats["frames_with_faces"] / stats["total_frames"] * 100, 1) if stats["total_frames"] > 0 else 0,
                    "enhancement_applied": stats["enhancement_applied"]
                }
            }
            
            return response
            
    except Exception as e:
        logger.error(f"❌ Handler error: {e}")
        return {
            "error": str(e),
            "status": "failed",
            "processing_time_seconds": round(time.time() - start_time, 2)
        }
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    logger.info("🚀 Starting Face Enhancement Serverless Worker...")
    logger.info(f"🗄️ Storage: {MINIO_ENDPOINT}/{MINIO_BUCKET}")
    logger.info(f"🎨 Supported enhancers: {', '.join(ENHANCER_CONFIGS.keys())}")
    
    # Verify environment
    try:
        logger.info(f"🐍 Python: {sys.version}")
        logger.info(f"🔥 PyTorch: {torch.__version__}")
        logger.info(f"⚡ CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"🎮 GPU: {torch.cuda.get_device_name()}")
            logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    except Exception as e:
        logger.warning(f"⚠️ Environment check failed: {e}")
    
    # Verify model files exist
    for enhancer_name, config in ENHANCER_CONFIGS.items():
        model_path = Path(config["model_path"])
        if model_path.exists():
            logger.info(f"✅ {enhancer_name.upper()} model verified")
        else:
            logger.warning(f"⚠️ {enhancer_name.upper()} model missing: {model_path}")
    
    # Initialize face detection
    try:
        initialize_face_detection()
        logger.info("✅ All models initialized successfully")
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        sys.exit(1)
    
    # Start RunPod serverless worker
    logger.info("🎬 Ready to process face enhancement requests...")
    runpod.serverless.start({"handler": handler})
