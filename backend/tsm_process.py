import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from dataclasses import dataclass
from collections import OrderedDict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.eca_res50_model import eca_resnet50
from model.temporal_shift import make_temporal_shift

# device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'device detected: {device}')

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

# 数据预处理变换
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image, n_segment=8):
    """
    预处理单张图片，复制为时间序列
    Args:
        image: OpenCV格式的图片 (BGR)
        n_segment: 时间片段数
    Returns:
        tensor: 预处理后的tensor [n_segment, 3, 224, 224]
    """
    # BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 应用变换
    image_tensor = transform(image_rgb)
    
    # 复制为时间序列 [n_segment, 3, 224, 224]
    image_sequence = image_tensor.unsqueeze(0).repeat(n_segment, 1, 1, 1)
    
    return image_sequence

def preprocess_video_frames(video_frames):
    """
    预处理视频帧序列
    Args:
        video_frames: list of frames (OpenCV format)
    Returns:
        tensor: 预处理后的tensor [n_frames, 3, 224, 224]
    """
    processed_frames = []
    for frame in video_frames:
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = transform(frame_rgb)
        processed_frames.append(frame_tensor)
    
    return torch.stack(processed_frames)

def load_video_frames(video_path, n_segment=8):
    """
    从视频文件加载指定数量的帧
    Args:
        video_path: 视频文件路径
        n_segment: 需要的帧数
    Returns:
        list: 视频帧列表
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, n_segment, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            # 如果读取失败，用最后一帧填充
            if frames:
                frames.append(frames[-1])
            else:
                break
    
    cap.release()
    return frames

def analyze_video_for_shots(video_path, model, n_segment=8, step_size=4, shot_threshold=0.7, clip_info: ClipInfo = None):
    """
    使用滑动窗口分析视频，找到打门关键帧
    Args:
        video_path: 视频文件路径
        model: 训练好的TSM模型
        n_segment: 每个窗口的帧数
        step_size: 滑动窗口步长
        shot_threshold: 打门检测的置信度阈值
    Returns:
        list: 包含打门信息的字典列表，每个字典包含frame_index, timestamp, confidence
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return []
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
    
    # 读取所有帧
    all_frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
        frame_count += 1
    
    cap.release()
    print(f"Successfully loaded {len(all_frames)} frames")
    
    shot_detections = []
    
    # 滑动窗口分析
    for start_idx in range(0, len(all_frames) - n_segment + 1, step_size):
        end_idx = start_idx + n_segment
        window_frames = all_frames[start_idx:end_idx]
        
        # 预处理当前窗口的帧
        input_tensor = preprocess_video_frames(window_frames)
        input_tensor = input_tensor.to(device)
        
        # 推理
        with torch.no_grad():
            outputs = model(input_tensor)
            # 对8个帧的预测结果进行平均
            outputs_mean = outputs.mean(dim=0, keepdim=True)
            probabilities = F.softmax(outputs_mean, dim=1)
            shot_confidence = probabilities[0][0].item()  # class 0 是 'shot'
            
            # 如果检测到打门且置信度超过阈值
            if shot_confidence > shot_threshold:
                # 计算关键帧索引（窗口中间的帧）
                key_frame_idx = start_idx + n_segment // 2
                timestamp = key_frame_idx / fps
                
                shot_info = {
                    'frame_index': key_frame_idx,
                    'timestamp': timestamp,
                    'confidence': shot_confidence,
                    'window_start': start_idx,
                    'window_end': end_idx - 1
                }
                
                shot_detections.append(shot_info)
                print(f"Shot detected at frame {key_frame_idx} (t={timestamp:.2f}s), confidence: {shot_confidence:.4f}")
    
    # 去除重复检测（基于时间距离）
    filtered_detections = remove_duplicate_detections(shot_detections, min_time_gap=clip_info.before_secs + clip_info.after_secs)
    
    return filtered_detections

def remove_duplicate_detections(detections, min_time_gap=2.0):
    """
    去除时间上过于接近的重复检测
    Args:
        detections: 检测结果列表
        min_time_gap: 最小时间间隔（秒）
    Returns:
        list: 过滤后的检测结果
    """
    if not detections:
        return []
    
    # 按置信度排序
    sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    filtered = []
    
    for detection in sorted_detections:
        # 检查是否与已有检测过于接近
        is_duplicate = False
        for existing in filtered:
            time_diff = abs(detection['timestamp'] - existing['timestamp'])
            if time_diff < min_time_gap:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered.append(detection)
    
    # 按时间排序
    return sorted(filtered, key=lambda x: x['timestamp'])

def extract_shot_clips(video_path, shot_detections, before_seconds=2.0, after_seconds=2.0, output_dir="clips"):
    """
    根据检测到的打门关键帧提取视频片段，避免重叠片段
    Args:
        video_path: 原视频路径
        shot_detections: 打门检测结果
        before_seconds: 关键帧前提取的秒数
        after_seconds: 关键帧后提取的秒数
        output_dir: 输出目录
    """
    if not shot_detections:
        print("No shot detections to extract clips from")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 记录已提取的视频片段范围 [(start_frame, end_frame), ...]
    extracted_ranges = []
    extracted_count = 0
    
    for i, detection in enumerate(shot_detections):
        key_frame = detection['frame_index']
        timestamp = detection['timestamp']
        confidence = detection['confidence']
        
        # 计算提取范围
        start_frame = max(0, int(key_frame - before_seconds * fps))
        end_frame = int(key_frame + after_seconds * fps)
        
        # 检查是否与已提取的片段重叠
        is_overlapping = False
        for existing_start, existing_end in extracted_ranges:
            # 检查是否有重叠：新片段的开始 < 已有片段的结束 AND 新片段的结束 > 已有片段的开始
            if start_frame < existing_end and end_frame > existing_start:
                is_overlapping = True
                print(f"Skipping shot {i+1} (frame {key_frame}, t={timestamp:.1f}s): overlaps with previous extraction")
                break
        
        # 如果没有重叠，则提取视频片段
        if not is_overlapping:
            extracted_count += 1
            # 输出文件名
            output_filename = f"shot_{extracted_count}_frame{key_frame}_t{timestamp:.1f}s_conf{confidence:.3f}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            # 提取视频片段
            extract_video_segment(video_path, start_frame, end_frame, output_path)
            print(f"Extracted clip: {output_filename}")
            
            # 记录已提取的范围
            extracted_ranges.append((start_frame, end_frame))
    
    cap.release()
    print(f"Total extracted clips: {extracted_count} out of {len(shot_detections)} detections")

def extract_video_segment(input_path, start_frame, end_frame, output_path):
    """
    提取视频片段
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 设置视频编码器 使用H.264编码以获得更好的浏览器兼容性
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 跳到开始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()

def load_model():
    # 初始化模型
    model = eca_resnet50(k_size=[3, 5, 5, 7], num_classes=2)
    make_temporal_shift(model, n_segment=8, n_div=8, place='blockres')

    # 加载预训练权重
    ckpt = torch.load('../model/model_ckpts/eca_tsm_best.pth', map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']

    new_state = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state[name] = v

    model.load_state_dict(new_state)
    model.to(device)
    model.eval()  # 设置为评估模式

    print(f'Model loaded successfully on {device}')
    return model

def process_video(clip_info: ClipInfo, video_path: str) -> ShotResult:
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return None
    
    model = load_model()
    
    # params
    n_segment = 8
    step_size = 4
    shot_threshold = 0.997
    
    # analyze video
    shot_detections = analyze_video_for_shots(
        video_path,
        model,
        n_segment=n_segment,
        step_size=step_size, 
        shot_threshold=shot_threshold,
        clip_info=clip_info
    )

    print(f"共检测到 {len(shot_detections)} 个打门动作:")
    for i, detection in enumerate(shot_detections):
        print(f"Shot {i+1}: Frame {detection['frame_index']}, "
              f"Time {detection['timestamp']:.2f}s, "
              f"Confidence {detection['confidence']:.4f}")

    # 提取视频片段
    clips_dir = 'clips'
    if shot_detections:
        print(f"\n=== 提取视频片段 ===")
        extract_shot_clips(
            video_path, 
            shot_detections, 
            before_seconds=clip_info.before_secs, 
            after_seconds=clip_info.after_secs,
            output_dir=clips_dir
        )
        print(f"视频片段已保存到 {clips_dir} 目录")
    else:
        print("未检测到打门动作")

    
    shot_result = ShotResult(
        video_path=video_path,
        frame_idx=[detection['frame_index'] for detection in shot_detections],
        timestamp=[detection['timestamp'] for detection in shot_detections],
        confidence=[detection['confidence'] for detection in shot_detections],
        clip_path=[f'backend/clips/{f}' for f in os.listdir(clips_dir) if f.endswith('.mp4')] if os.path.exists(clips_dir) else []
    )

    return shot_result