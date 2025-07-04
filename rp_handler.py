#!/usr/bin/env python3
"""
RunPod Serverless Handler - EXACT COPY of enhancer_cli.py logic
"""

import runpod
import os
import tempfile
import uuid
import requests
import logging
import subprocess
from pathlib import Path
from minio import Minio
from urllib.parse import quote
from datetime import datetime

# EXACT imports như enhancer_cli.py
import numpy as np
import cv2
import sys
from tqdm import tqdm
import onnxruntime

# Suppress ONNX warnings như gốc
onnxruntime.set_default_logger_severity(3)

# Add path for local modules
sys.path.append('/app')

# Import face processing modules
from utils.retinaface import RetinaFace
from utils.face_alignment import get_cropped_head_256
from faceID.faceID import FaceRecognition

# Configure minimal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MinIO Configuration
MINIO_ENDPOINT = "108.181.198.160:9000"
MINIO_ACCESS_KEY = "a9TFRtBi8q3Nvj5P5Ris"
MINIO_SECRET_KEY = "fCFngM7YTr6jSkBKXZ9BkfDdXrStYXm43UGa0OZQ"
MINIO_BUCKET = "aiclipdfl"
MINIO_SECURE = False

minio_client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE)

# GLOBAL model initialization như enhancer_cli.py
detector = RetinaFace(
    "/app/utils/scrfd_2.5g_bnkps.onnx",
    provider=[
        ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
        "CPUExecutionProvider"
    ],
    session_options=None
)

recognition = FaceRecognition('/app/faceID/recognition.onnx')

# Load the specified enhancer model - EXACT COPY
def load_enhancer(enhancer_name, device):
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

# Process a batch of frames - EXACT COPY (NO ThreadPoolExecutor)
def process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height):
    frames, aligned_faces, mats = zip(*frame_buffer)

    # Enhance faces in batch
    enhanced_faces = enhancer.enhance_batch(aligned_faces)

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

# Enhance the video - EXACT COPY
def enhance_video(video_path, enhancer_name, output_path=None):
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

    batch_size = 1  # Adjust batch size based on memory and performance
    frame_buffer = []

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
    try:
        minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
        return f"http://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise e

def handler(job):
    job_id = job.get("id", "unknown")
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        video_url = job_input.get("video_url")
        
        if not video_url:
            return {"error": "Missing video_url"}
        
        logger.info(f"🚀 Job {job_id}: GFPGAN Face Enhancement")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "input.mp4")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(temp_dir, f"enhanced_{timestamp}.mp4")
            
            # Download video
            if not download_file(video_url, video_path):
                return {"error": "Failed to download video"}
            
            # Enhance video - EXACT như enhancer_cli.py
            enhance_video(video_path, "gfpgan", output_path)
            
            if not os.path.exists(output_path):
                return {"error": "Enhancement output not found"}
            
            # Upload result
            output_filename = f"gfpgan_enhanced_{job_id}_{uuid.uuid4().hex[:8]}.mp4"
            output_url = upload_to_minio(output_path, output_filename)
            
            processing_time = time.time() - start_time
            
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
    logger.info("🚀 Starting GFPGAN Face Enhancement Worker (EXACT Copy)...")
    logger.info("🎬 Ready to process requests...")
    runpod.serverless.start({"handler": handler})
