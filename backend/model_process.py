import os
import sys
import torch
import cv2
import numpy as np
from datetime import timedelta
from moviepy.editor import VideoFileClip
from dataclasses import dataclass
from typing import List, Dict
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.eca_res50_model import eca_resnet50

# device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'device detected: {device}')

# load model
ckpt = '../model/model_ckpts/eca_resnet50_best.pth'
model = eca_resnet50(k_size=[3, 5, 5, 7], num_classes=2)
model.load_state_dict(torch.load(ckpt, map_location=device))
model.to(device)
model.eval()

@dataclass
class ClipInfo:
    clip_mode: str
    before_secs: int
    after_secs: int

@dataclass
class ShotResult:
    video_path: str
    frame_idx: list[int]
    timestamp: list[float]
    confidence: list[float]
    clip_path: list[str]

def get_metadata(clip_mode, before_secs, after_secs):
    clip_info = ClipInfo(clip_mode, before_secs, after_secs)
    return clip_info

def preprocess_frame(frame):
    # Resize frame to match model input size
    frame = cv2.resize(frame, (224, 224))
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Normalize pixel values
    frame = frame.astype(np.float32) / 255.0
    # Convert to tensor and add batch dimension
    frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
    return frame.to(device)

def temporal_smooth(predictions: List[float], window_size: int = 5, threshold: float = 0.7) -> List[bool]:
    """Apply temporal smoothing to predictions"""
    smoothed = []
    half_window = window_size // 2
    
    # Pad the predictions array
    padded_preds = [0.0] * half_window + predictions + [0.0] * half_window
    
    for i in range(len(predictions)):
        # Get window around current prediction
        window = padded_preds[i:i + window_size]
        # If average confidence in window exceeds threshold, mark as shot
        smoothed.append(sum(window) / window_size > threshold)
    
    return smoothed

def process_video(clip_info: ClipInfo, video_path: str) -> ShotResult:
    if not os.path.exists(video_path):
        return None
    
    # 1. Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'video fps: {fps}')
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Total frames: {total_frames}')
    # Load video with moviepy for clip extraction
    video = VideoFileClip(video_path)
    
    # Store frame predictions and results
    frame_predictions = []
    
    # Initialize lists for collecting results
    frame_indices = []
    timestamps = []
    confidences = []
    clip_paths = []
    
    # 2. Process frames
    frame_idx = 0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 3. Predict frame type
            processed_frame = preprocess_frame(frame)
            output = model(processed_frame)
            prediction = torch.softmax(output, dim=1)
            shot_confidence = prediction[0][1].item()
            frame_predictions.append(shot_confidence)
            
            frame_idx += 1

    print(f'Frame predictions: {len(frame_predictions)}')
    print(f'frame_predictions > 0.75: {sum([1 for conf in frame_predictions if conf > 0.75])}')
    
    # 4. Apply temporal smoothing
    is_shot_frames = temporal_smooth(frame_predictions, window_size=4, threshold=0.75)
    # is_shot_frames = [True if conf > 0.75 else False for conf in frame_predictions]
    # print(f'Is shot frames: {len(is_shot_frames)}')

    # 5. Process shot frames and save clips
    clips_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clips')
    os.makedirs(clips_folder, exist_ok=True)
    
    frame_idx = 0
    for is_shot, confidence in tqdm(zip(is_shot_frames, frame_predictions)):
        if is_shot:
            # Get timestamp
            timestamp = frame_idx / fps
            
            # Save video clip
            start_time = max(0, timestamp - clip_info.before_secs)
            end_time = min(video.duration, timestamp + clip_info.after_secs)
            
            try:
                # Use standard position parameters for subclip
                clip = video.subclip(start_time, end_time)
                output_path = os.path.join(clips_folder, f"shot_{frame_idx}.mp4")
                clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
                clip.close()
                
                # Add results to our lists
                frame_indices.append(frame_idx)
                timestamps.append(timestamp)
                confidences.append(confidence)
                clip_paths.append(f"shot_{frame_idx}.mp4")  # 只保存文件名，不包含路径
                
                print(f"Saved clip for frame {frame_idx} at {output_path}")
            except Exception as e:
                print(f"Error saving clip at frame {frame_idx}: {e}")
                continue
        
        frame_idx += 1
    
    # Create a single ShotResult containing all results
    shot_result = None
    if frame_indices:
        shot_result = ShotResult(
            video_path=video_path,
            frame_idx=frame_indices,
            timestamp=timestamps,
            confidence=confidences,
            clip_path=clip_paths
        )
    
    # Cleanup
    cap.release()
    video.close()
    
    return shot_result



