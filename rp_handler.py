#!/usr/bin/env python3
"""
RunPod Serverless Handler for GFPGAN Face Enhancement
All-in-one version with comprehensive GPU logging
"""

import runpod
import os
import tempfile
import uuid
import requests
import logging
import time
import gc
import subprocess
from pathlib import Path
from minio import Minio
from urllib.parse import quote
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Import required libraries
import numpy as np
import cv2
import torch
import onnxruntime
from tqdm import tqdm
import sys

# Add path for local modules
sys.path.append('/app')

# Import face processing modules
try:
    from utils.retinaface import RetinaFace
    from utils.face_alignment import get_cropped_head_256
    from enhancers.GFPGAN.GFPGAN import GFPGAN
    from faceID.faceID import FaceRecognition
except ImportError as e:
    logging.error(f"Import error: {e}")
    sys.exit(1)

# Configure logging
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

# Global model instances
detector = None
enhancer = None
recognition = None

def log_gpu_status(step_name: str):
    """Log detailed GPU status"""
    logger.info(f"🔍 GPU Status - {step_name}:")
    
    # PyTorch GPU info
    try:
        logger.info(f"  PyTorch CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  PyTorch CUDA Version: {torch.version.cuda}")
            logger.info(f"  PyTorch GPU Count: {torch.cuda.device_count()}")
            logger.info(f"  PyTorch Current Device: {torch.cuda.current_device()}")
            logger.info(f"  PyTorch Device Name: {torch.cuda.get_device_name()}")
            
            # GPU Memory info
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"  PyTorch GPU Memory - Allocated: {memory_allocated:.2f} GB")
            logger.info(f"  PyTorch GPU Memory - Reserved: {memory_reserved:.2f} GB")
    except Exception as e:
        logger.error(f"  PyTorch GPU check failed: {e}")
    
    # ONNX Runtime GPU info
    try:
        providers = onnxruntime.get_available_providers()
        logger.info(f"  ONNX Available Providers: {providers}")
        logger.info(f"  ONNX CUDA Provider: {'CUDAExecutionProvider' in providers}")
        logger.info(f"  ONNX TensorRT Provider: {'TensorrtExecutionProvider' in providers}")
    except Exception as e:
        logger.error(f"  ONNX GPU check failed: {e}")
    
    # System GPU info
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info(f"  nvidia-smi: {result.stdout.strip()}")
        else:
            logger.warning(f"  nvidia-smi failed: {result.stderr}")
    except Exception as e:
        logger.warning(f"  nvidia-smi not available: {e}")

def determine_device() -> str:
    """Determine optimal device for processing"""
    logger.info("🔧 Determining processing device...")
    
    # Check ONNX Runtime GPU support
    providers = onnxruntime.get_available_providers()
    cuda_available = 'CUDAExecutionProvider' in providers
    
    # Check PyTorch GPU support
    pytorch_cuda = torch.cuda.is_available()
    
    if cuda_available:
        device = 'cuda'
        logger.info(f"✅ Using CUDA device (ONNX: {cuda_available}, PyTorch: {pytorch_cuda})")
    else:
        device = 'cpu'
        logger.warning(f"⚠️ Falling back to CPU (ONNX: {cuda_available}, PyTorch: {pytorch_cuda})")
    
    log_gpu_status("Device Selection")
    return device

def initialize_models():
    """Initialize face detection and enhancement models"""
    global detector, enhancer, recognition
    
    logger.info("🤖 Initializing models...")
    log_gpu_status("Before Model Initialization")
    
    device = determine_device()
    
    try:
        # Initialize face detector
        detector_path = "/app/utils/scrfd_2.5g_bnkps.onnx"
        if not os.path.exists(detector_path):
            raise FileNotFoundError(f"Face detector model not found: {detector_path}")
        
        logger.info("📥 Loading face detector...")
        if device == 'cuda':
            providers = [
                ("CUDAExecutionProvider", {
                    "cudnn_conv_algo_search": "DEFAULT",
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB
                    "cudnn_conv_use_max_workspace": '1'
                }),
                "CPUExecutionProvider"
            ]
        else:
            providers = ["CPUExecutionProvider"]
        
        detector = RetinaFace(detector_path, provider=providers, session_options=None)
        logger.info("✅ Face detector loaded")
        
        # Initialize face recognition
        recognition_path = "/app/faceID/recognition.onnx"
        if not os.path.exists(recognition_path):
            raise FileNotFoundError(f"Face recognition model not found: {recognition_path}")
        
        logger.info("📥 Loading face recognition...")
        recognition = FaceRecognition(recognition_path)
        logger.info("✅ Face recognition loaded")
        
        # Initialize GFPGAN enhancer
        enhancer_path = "/app/enhancers/GFPGAN/GFPGANv1.4.onnx"
        if not os.path.exists(enhancer_path):
            raise FileNotFoundError(f"GFPGAN model not found: {enhancer_path}")
        
        logger.info("📥 Loading GFPGAN enhancer...")
        enhancer = GFPGAN(model_path=enhancer_path, device=device)
        logger.info(f"✅ GFPGAN enhancer loaded on {device}")
        
        log_gpu_status("After Model Initialization")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        log_gpu_status("Model Initialization Failed")
        return False

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

def process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height, device):
    """Process batch of frames with face enhancement"""
    frames, aligned_faces, mats = zip(*frame_buffer)
    
    logger.info(f"🎨 Processing batch of {len(frames)} frames on {device}...")
    log_gpu_status("Before Batch Processing")
    
    # Convert to numpy array for batch processing
    aligned_faces_array = np.array(aligned_faces)
    
    # Enhance faces in batch
    start_time = time.time()
    enhanced_faces = enhancer.enhance_batch(aligned_faces_array)
    enhancement_time = time.time() - start_time
    
    logger.info(f"✅ Batch enhancement completed in {enhancement_time:.2f}s")
    log_gpu_status("After Batch Processing")
    
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

def enhance_video(video_path: str, output_path: str) -> tuple[bool, dict]:
    """Enhanced video processing with GFPGAN"""
    global detector, enhancer, recognition
    
    stats = {
        "total_frames": 0,
        "frames_with_faces": 0,
        "frames_without_faces": 0,
        "faces_enhanced": 0,
        "enhancement_applied": False,
        "device_used": "unknown"
    }
    
    try:
        # Determine device
        device = determine_device()
        stats["device_used"] = device
        
        logger.info(f"🎬 Starting video enhancement on {device}...")
        log_gpu_status("Video Enhancement Start")
        
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
        logger.info(f"📺 Video info: {frame_width}x{frame_height}, {fps} fps, {total_frames} frames")
        
        # Create temporary video file
        temp_video_path = output_path.replace('.mp4', '_temp.mp4')
        out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (frame_width, frame_height))
        
        # Create face mask for blending
        face_mask = np.zeros((256, 256), dtype=np.uint8)
        face_mask = cv2.rectangle(face_mask, (66, 69), (190, 240), (255, 255, 255), -1)
        face_mask = cv2.GaussianBlur(face_mask.astype(np.uint8), (19, 19), cv2.BORDER_DEFAULT)
        face_mask = cv2.cvtColor(face_mask, cv2.COLOR_GRAY2RGB)
        face_mask = face_mask / 255
        
        # Determine batch size based on device
        if device == 'cuda':
            batch_size = 1  # GFPGAN works best with batch size 1
        else:
            batch_size = 1
        
        logger.info(f"⚙️ Using batch size: {batch_size}")
        
        # Process video in chunks for memory optimization
        MAX_FRAMES_IN_MEMORY = 1000
        
        for frame_idx in tqdm(range(0, total_frames, MAX_FRAMES_IN_MEMORY), desc="Processing chunks"):
            chunk_frames = min(MAX_FRAMES_IN_MEMORY, total_frames - frame_idx)
            frame_buffer = []
            
            logger.info(f"📦 Processing chunk {frame_idx//MAX_FRAMES_IN_MEMORY + 1}: frames {frame_idx}-{frame_idx+chunk_frames}")
            log_gpu_status(f"Chunk {frame_idx//MAX_FRAMES_IN_MEMORY + 1} Start")
            
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
                        process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height, device)
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
                process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height, device)
                stats["faces_enhanced"] += len(frame_buffer)
            
            # Memory cleanup
            gc.collect()
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("🧹 GPU memory cleaned")
        
        video_stream.release()
        out.release()
        
        if stats["frames_with_faces"] > 0:
            stats["enhancement_applied"] = True
            logger.info(f"✅ Face enhancement applied to {stats['frames_with_faces']}/{stats['total_frames']} frames")
        else:
            logger.info("ℹ️ No faces detected in any frame")
            stats["enhancement_applied"] = False
        
        # Extract and combine audio
        logger.info("🎵 Processing audio...")
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
        log_gpu_status("Video Enhancement Completed")
        return True, stats
        
    except Exception as e:
        logger.error(f"❌ Video enhancement failed: {e}")
        log_gpu_status("Video Enhancement Failed")
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
        log_gpu_status("Job Start")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # File paths
            video_path = os.path.join(temp_dir, "input.mp4")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(temp_dir, f"enhanced_{timestamp}.mp4")
            
            # Step 1: Download video
            logger.info("📥 Step 1/3: Downloading video...")
            if not download_file(video_url, video_path):
                return {"error": "Failed to download video"}
            
            # Step 2: Enhance video
            logger.info("✨ Step 2/3: Enhancing with GFPGAN...")
            enhancement_success, stats = enhance_video(video_path, output_path)
            
            if not enhancement_success:
                return {"error": "Face enhancement failed"}
            
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
                "device_used": stats["device_used"],
                "job_id": job_id,
                "enhancement_stats": {
                    "total_frames": stats["total_frames"],
                    "frames_enhanced": stats["frames_with_faces"],
                    "enhancement_rate": round(stats["frames_with_faces"] / stats["total_frames"] * 100, 1) if stats["total_frames"] > 0 else 0,
                    "enhancement_applied": stats["enhancement_applied"]
                }
            }
            
            logger.info(f"✅ Job {job_id} completed in {processing_time:.1f}s on {stats['device_used']}")
            log_gpu_status("Job Completed")
            return response
            
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Job {job_id} failed: {e}")
        log_gpu_status("Job Failed")
        
        return {
            "status": "failed",
            "error": str(e),
            "processing_time_seconds": round(processing_time, 2),
            "job_id": job_id
        }

if __name__ == "__main__":
    logger.info("🚀 Starting GFPGAN Face Enhancement Worker (All-in-One + GPU Logging)...")
    
    # Initial environment check
    log_gpu_status("Worker Startup")
    
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
    
    # Initialize models
    logger.info("🤖 Initializing models...")
    if not initialize_models():
        logger.error("❌ Model initialization failed")
        sys.exit(1)
    
    logger.info(f"🗄️ Storage: {MINIO_ENDPOINT}/{MINIO_BUCKET}")
    logger.info("🎬 Ready to process requests...")
    
    # Start RunPod worker
    runpod.serverless.start({"handler": handler})
