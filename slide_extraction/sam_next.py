"""
SAM3 Video Cropping Tool
Crops videos using SAM3 object detection with FFmpeg acceleration.
"""

import argparse
import shutil
import subprocess
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Crop videos using SAM3 object detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input', help='Input video file or directory')
    parser.add_argument('output', help='Output directory')
    parser.add_argument('--prompt', default='slide', help='Text prompt for object detection')
    parser.add_argument('--bpe-path', default='./assets/bpe_simple_vocab_16e6.txt.gz',
                        help='Path to BPE vocabulary file')
    parser.add_argument('--frames', type=int, default=32,
                        help='Number of frames to sample for detection')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for filtering outliers')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold for detections')
    parser.add_argument('--padding', type=float, default=0.01,
                        help='Padding percentage around detected object')
    parser.add_argument('--resolution', type=str, default=None,
                        help='Output resolution as WIDTHxHEIGHT (e.g., 1920x1080)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for inference')
    return parser.parse_args()


def check_ffmpeg():
    return shutil.which('ffmpeg') is not None


def compute_iou(box1, box2):
    if torch.is_tensor(box1):
        box1 = box1.cpu().numpy()
    if torch.is_tensor(box2):
        box2 = box2.cpu().numpy()
    
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def extract_frames(video_path, num_frames):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = (np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int) 
                     if total_frames >= num_frames else list(range(total_frames)))
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    
    cap.release()
    return frames


def batch_inference(processor, frames, text_prompt, batch_size):
    all_detections = []
    
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        for frame in batch:
            state = processor.set_image(frame)
            detection = processor.set_text_prompt(state=state, prompt=text_prompt)
            all_detections.append({
                "boxes": detection["boxes"].detach().cpu(),
                "scores": detection["scores"].detach().cpu()
            })
        torch.cuda.empty_cache()
    
    return all_detections


def filter_boxes(boxes, iou_threshold, min_overlap_ratio=0.25):
    if len(boxes) == 0:
        return torch.empty((0, 4))
    
    boxes_np = boxes.numpy() if torch.is_tensor(boxes) else np.array(boxes)
    n = len(boxes_np)
    
    iou_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            iou = compute_iou(boxes_np[i], boxes_np[j])
            iou_matrix[i, j] = iou_matrix[j, i] = iou
    
    overlap_scores = np.sum(iou_matrix >= iou_threshold, axis=1) / n
    valid_indices = np.where(overlap_scores >= min_overlap_ratio)[0]
    
    if len(valid_indices) == 0:
        valid_indices = np.arange(n)
    
    return torch.from_numpy(boxes_np[valid_indices])


def get_average_bbox(video_path, processor, text_prompt, num_frames, iou_threshold, confidence, batch_size):
    print(f"Processing: {video_path.name}")
    
    frames = extract_frames(video_path, num_frames)
    print(f"  Extracted {len(frames)} frames")
    
    detections = batch_inference(processor, frames, text_prompt, batch_size)
    
    boxes = []
    for det in detections:
        for box, score in zip(det["boxes"], det["scores"]):
            if score >= confidence:
                boxes.append(box)
    
    if len(boxes) == 0:
        raise ValueError(f"No detections above confidence threshold {confidence}")
    
    print(f"  Found {len(boxes)} detections")
    
    boxes_tensor = torch.stack(boxes)
    filtered = filter_boxes(boxes_tensor, iou_threshold)
    print(f"  Kept {len(filtered)} after filtering")
    
    avg_box = filtered.numpy().mean(axis=0)
    return avg_box


def add_padding(bbox, width, height, padding_pct):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    
    pad_x, pad_y = w * padding_pct, h * padding_pct
    
    return [
        int(max(0, x1 - pad_x)),
        int(max(0, y1 - pad_y)),
        int(min(width, x2 + pad_x)),
        int(min(height, y2 + pad_y))
    ]


def crop_video_ffmpeg(input_path, output_path, crop_x, crop_y, crop_w, crop_h, target_res=None):
    vf_filters = [f'crop={crop_w}:{crop_h}:{crop_x}:{crop_y}']
    if target_res:
        vf_filters.append(f'scale={target_res[0]}:{target_res[1]}')
    
    cmd = [
        'ffmpeg', '-y', '-i', str(input_path),
        '-vf', ','.join(vf_filters),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'copy',
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")


def crop_video_opencv(input_path, output_path, crop_x, crop_y, crop_w, crop_h, target_res=None):
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out_w, out_h = target_res if target_res else (crop_w, crop_h)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        if target_res:
            cropped = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        
        out.write(cropped)
    
    cap.release()
    out.release()


def crop_video(input_path, output_path, bbox, padding_pct, target_res, use_ffmpeg):
    cap = cv2.VideoCapture(str(input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    x1, y1, x2, y2 = add_padding(bbox, width, height, padding_pct)
    crop_w, crop_h = x2 - x1, y2 - y1
    
    print(f"  Crop region: [{x1}, {y1}, {x2}, {y2}] ({crop_w}x{crop_h})")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if use_ffmpeg:
        crop_video_ffmpeg(input_path, output_path, x1, y1, crop_w, crop_h, target_res)
    else:
        crop_video_opencv(input_path, output_path, x1, y1, crop_w, crop_h, target_res)
    
    print(f"  Saved: {output_path}")
    
    return {
        'input': str(input_path),
        'output': str(output_path),
        'bbox': bbox.tolist(),
        'crop_region': [x1, y1, x2, y2],
        'resolution': f"{crop_w}x{crop_h}"
    }


def process_video(video_path, output_dir, args, processor, use_ffmpeg):
    try:
        avg_bbox = get_average_bbox(
            video_path, processor, args.prompt, 
            args.frames, args.iou_threshold, args.confidence, args.batch_size
        )
        
        output_path = output_dir / video_path.name
        crop_info = crop_video(
            video_path, output_path, avg_bbox, 
            args.padding, args.resolution, use_ffmpeg
        )
        
        info_path = output_dir / f"{video_path.stem}_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Input: {crop_info['input']}\n")
            f.write(f"Output: {crop_info['output']}\n")
            f.write(f"Prompt: {args.prompt}\n")
            f.write(f"BBox: {crop_info['bbox']}\n")
            f.write(f"Crop: {crop_info['crop_region']}\n")
            f.write(f"Resolution: {crop_info['resolution']}\n")
        
        print(f"  Info: {info_path}\n")
        return True
        
    except Exception as e:
        print(f"  Error: {e}\n")
        return False


def main():
    args = parse_args()
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    
    output_resolution = None
    if args.resolution:
        w, h = map(int, args.resolution.split('x'))
        output_resolution = (w, h)
    
    use_ffmpeg = check_ffmpeg()
    print(f"Using {'FFmpeg (fast)' if use_ffmpeg else 'OpenCV (slow)'}\n")
    
    print("Loading SAM3 model...")
    model = build_sam3_image_model(bpe_path=args.bpe_path)
    processor = Sam3Processor(model, confidence_threshold=args.confidence)
    print("Model loaded\n")
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        videos = [input_path]
    elif input_path.is_dir():
        videos = list(input_path.glob('*.mp4')) + list(input_path.glob('*.avi')) + \
                 list(input_path.glob('*.mov')) + list(input_path.glob('*.mkv'))
    else:
        raise ValueError(f"Invalid input path: {input_path}")
    
    print(f"Processing {len(videos)} video(s)\n")
    
    success_count = 0
    for video in videos:
        if process_video(video, output_dir, args, processor, use_ffmpeg):
            success_count += 1
    
    print(f"\nCompleted: {success_count}/{len(videos)} videos processed successfully")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
