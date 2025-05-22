import os
import sys
import torch
import cv2
import numpy as np
from datetime import timedelta
from moviepy import VideoFileClip
from dataclasses import dataclass
from typing import List, Dict

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
    frame_idx: int
    timestamp: float
    confidence: float
    clip_path: str

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

def process_video(clip_info: ClipInfo, video_path: str) -> List[ShotResult]:
    if not os.path.exists(video_path):
        return None
    
    # 1. Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Load video with moviepy for clip extraction
    video = VideoFileClip(video_path)
    
    # Store frame predictions and results
    frame_predictions = []
    shot_results = []
    
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
    
    # 4. Apply temporal smoothing
    is_shot_frames = temporal_smooth(frame_predictions, window_size=5, threshold=0.7)
    
    # 5. Process shot frames and save clips
    os.makedirs("clips", exist_ok=True)
    
    frame_idx = 0
    for is_shot, confidence in zip(is_shot_frames, frame_predictions):
        if is_shot:
            # Get timestamp
            timestamp = frame_idx / fps
            
            # Save video clip
            start_time = max(0, timestamp - clip_info.before_secs)
            end_time = min(video.duration, timestamp + clip_info.after_secs)
            
            try:
                # Use subclipped instead of subclip
                clip = video.subclipped(t_start=start_time, t_end=end_time)
                output_path = f"clips/shot_{frame_idx}.mp4"
                clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
                clip.close()
                
                # Record result
                shot_results.append(ShotResult(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    confidence=confidence,
                    clip_path=output_path
                ))
            except Exception as e:
                print(f"Error saving clip at frame {frame_idx}: {e}")
                continue
        
        frame_idx += 1
    
    # Cleanup
    cap.release()
    video.close()
    
    return shot_results



