import subprocess
import platform
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import shutil
from tqdm import tqdm

import onnxruntime
onnxruntime.set_default_logger_severity(3)

# Face detection and alignment
from utils.retinaface import RetinaFace
from utils.face_alignment import get_cropped_head_256

# Face detection model initialization
detector = RetinaFace(
    "utils/scrfd_2.5g_bnkps.onnx",
    provider=[
        ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
        "CPUExecutionProvider"
    ],
    session_options=None
)

# Specific face selector
from faceID.faceID import FaceRecognition
recognition = FaceRecognition('faceID/recognition.onnx')

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Enhance video using specified model')
    parser.add_argument('--video_path', type=str, help='Filepath of video to enhance', required=True)
    parser.add_argument('--enhancer', type=str, choices=['gpen', 'gfpgan', 'codeformer', 'restoreformer'], required=True, help='Model to use for enhancement')
    parser.add_argument('--output_path', type=str, help='Filepath to save the enhanced video', default=None)
    return parser.parse_args()

# Load the specified enhancer model
def load_enhancer(enhancer_name, device):
    if enhancer_name == 'gpen':
        from enhancers.GPEN.GPEN import GPEN
        return GPEN(model_path="enhancers/GPEN/GPEN-BFR-256-sim.onnx", device=device)
    elif enhancer_name == 'codeformer':
        from enhancers.Codeformer.Codeformer import CodeFormer
        return CodeFormer(model_path="enhancers/Codeformer/codeformerfixed.onnx", device=device)
    elif enhancer_name == 'restoreformer':
        from enhancers.restoreformer.restoreformer16 import RestoreFormer
        return RestoreFormer(model_path="enhancers/restoreformer/restoreformer16.onnx", device=device)
    elif enhancer_name == 'gfpgan':
        from enhancers.GFPGAN.GFPGAN import GFPGAN
        return GFPGAN(model_path="enhancers/GFPGAN/GFPGANv1.4.onnx", device=device)
    else:
        raise ValueError(f"Unknown enhancer: {enhancer_name}")

# Enhance the video
def enhance_video(video_path, enhancer_name, output_path=None):
    device = 'cpu'
    if onnxruntime.get_device() == 'GPU':
        device = 'cuda'
    print(f"Running on {device}")

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
        #subprocess.call(['ffmpeg', '-y', '-i', temp_video_path, '-i', audio_path, '-c:v', 'libx264', '-preset', 'slow', '-crf', '23', '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', '-map', '0:v:0', '-map', '1:a:0', output_path])

        subprocess.call([
             'ffmpeg', '-y', '-i', temp_video_path, '-i', audio_path, '-c:v', 'libx264', '-crf', '23', '-preset', 'medium', '-c:a', 'aac',
             '-b:a', '192k', '-movflags', '+faststart', output_path
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

# Process a batch of frames
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

# Main entry point
if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    enhance_video(args.video_path, args.enhancer, args.output_path)