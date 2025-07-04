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
import gc
from concurrent.futures import ThreadPoolExecutor

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
    parser.add_argument('--batch_size', type=int, help='Batch size for processing', default=0)
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
def enhance_video(video_path, enhancer_name, output_path=None, user_batch_size=0):
    device = 'cpu'
    if onnxruntime.get_device() == 'GPU':
        device = 'cuda'

    print(f"Running on {device}")

    # Tải enhancer
    enhancer = load_enhancer(enhancer_name, device)

    # Mở video stream
    video_stream = cv2.VideoCapture(video_path)
    if not video_stream.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    # Lấy thuộc tính video
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path is None:
        output_path = os.path.join('outputs', os.path.basename(video_path).replace('. ', f'_{enhancer_name}.'))

    # Tạo file tạm có tên đặc biệt để tránh trùng lặp
    base_name = os.path.splitext(output_path)[0]
    extension = os.path.splitext(output_path)[1]
    temp_video_path = f"{base_name}_temp_video{extension}"
    final_output_path = f"{base_name}_final{extension}"

    # Tạo video writer
    out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (frame_width, frame_height))

    # Tính toán mặt nạ khuôn mặt
    face_mask = np.zeros((256, 256), dtype=np.uint8)
    face_mask = cv2.rectangle(face_mask, (66, 69), (190, 240), (255, 255, 255), -1)
    face_mask = cv2.GaussianBlur(face_mask.astype(np.uint8), (19, 19), cv2.BORDER_DEFAULT)
    face_mask = cv2.cvtColor(face_mask, cv2.COLOR_GRAY2RGB)
    face_mask = face_mask / 255

    # Xác định batch size tối ưu
    if user_batch_size > 0:
        batch_size = user_batch_size
    else:
        if device == 'cuda':
            if hasattr(enhancer, 'recommended_batch_size'):
                batch_size = enhancer.recommended_batch_size
            elif enhancer_name == 'gfpgan':
                batch_size = 1  # GFPGAN có vấn đề với batch > 1
            elif enhancer_name == 'codeformer':
                batch_size = 8  # Tối ưu cho RTX A4000
            elif enhancer_name == 'restoreformer':
                batch_size = 16  # RestoreFormer tối ưu nhất ở batch size lớn
            else:
                batch_size = 8   # Giá trị mặc định an toàn
        else:
            batch_size = 1  # Trên CPU nên giữ batch nhỏ

    print(f"Using batch size: {batch_size}")

    # Xử lý video theo đoạn để tối ưu bộ nhớ
    MAX_FRAMES_IN_MEMORY = 1000  # Điều chỉnh theo RAM có sẵn

    try:
        for frame_idx in tqdm(range(0, total_frames, MAX_FRAMES_IN_MEMORY), desc="Xử lý theo đoạn"):
            chunk_frames = min(MAX_FRAMES_IN_MEMORY, total_frames - frame_idx)
            frame_buffer = []

            for _ in tqdm(range(chunk_frames), desc="Đọc frames"):
                ret, frame = video_stream.read()
                if not ret:
                    break

                # Phát hiện khuôn mặt
                bboxes, kpss = detector.detect(frame, input_size=(320, 320), det_thresh=0.3)
                if len(kpss) == 0:
                    out.write(frame)
                    continue

                aligned_face, mat = get_cropped_head_256(frame, kpss[0], size=256, scale=1.0)
                frame_buffer.append((frame, aligned_face, mat))

                if len(frame_buffer) >= batch_size:
                    process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height)
                    frame_buffer = []

            # Xử lý frame còn lại
            if frame_buffer:
                process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height)

            # Dọn bộ nhớ
            gc.collect()
            if device == 'cuda':
                try:
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass
    except Exception as e:
        print(f"Lỗi xử lý video: {e}")
        raise
    finally:
        video_stream.release()
        out.release()
        print(f"Enhanced video frames saved to {temp_video_path}")

    # Trích xuất âm thanh từ video gốc
    audio_path = os.path.splitext(output_path)[0] + '.aac'
    try:
        subprocess.call([
            'ffmpeg',
            '-i', video_path,
            '-vn',
            '-acodec', 'aac',
            '-b:a', '192k',
            audio_path
        ])
    except Exception as e:
        print(f"Error extracting audio: {e}")
        raise

    # Kết hợp video đã xử lý với âm thanh gốc
    try:
        subprocess.call([
            'ffmpeg', '-y',
            '-i', temp_video_path,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-crf', '23',
            '-preset', 'medium',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-movflags', '+faststart',
            final_output_path
        ])
        
        # Sau khi tạo thành công, đổi tên file final thành output cuối cùng
        if os.path.exists(final_output_path):
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(final_output_path, output_path)
    except Exception as e:
        print(f"Error combining video and audio: {e}")
        raise

    # Xóa file tạm
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    if os.path.exists(audio_path):
        os.remove(audio_path)

    print(f"Enhanced video with original audio saved to {output_path}")

# Process a batch of frames
def process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height):
    frames, aligned_faces, mats = zip(*frame_buffer)

    # Chuyển thành numpy array để xử lý batch
    aligned_faces_array = np.array(aligned_faces)

    # Nâng cao khuôn mặt theo batch
    enhanced_faces = enhancer.enhance_batch(aligned_faces_array)

    # Xử lý từng frame
    def process_single_frame(data):
        frame, aligned_face, mat, enhanced_face = data

        # Resize enhanced face
        enhanced_face_resized = cv2.resize(enhanced_face, (aligned_face.shape[1], aligned_face.shape[0]))

        # Resize face mask
        face_mask_resized = cv2.resize(face_mask, (enhanced_face_resized.shape[1], enhanced_face_resized.shape[0]))

        # Trộn enhanced face vào aligned face
        blended_face = (face_mask_resized * enhanced_face_resized + (1 - face_mask_resized) * aligned_face).astype(np.uint8)

        # Warp face trở lại vị trí gốc
        mat_rev = cv2.invertAffineTransform(mat)
        dealigned_face = cv2.warpAffine(blended_face, mat_rev, (frame_width, frame_height))

        # Áp dụng mặt nạ để blend vào frame gốc
        mask = cv2.warpAffine(face_mask_resized, mat_rev, (frame_width, frame_height))
        final_frame = (mask * dealigned_face + (1 - mask) * frame).astype(np.uint8)

        return final_frame

    # Xử lý đa luồng với ThreadPoolExecutor
    frame_data = zip(frames, aligned_faces, mats, enhanced_faces)

    with ThreadPoolExecutor() as executor:
        final_frames = list(executor.map(process_single_frame, frame_data))

    # Ghi các frame đã xử lý ra video
    for final_frame in final_frames:
        out.write(final_frame)

# Main entry point
if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    enhance_video(args.video_path, args.enhancer, args.output_path, args.batch_size)
