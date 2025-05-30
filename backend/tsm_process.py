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
    model.eval()  # inference

    # clean cache
    del ckpt, state_dict, new_state
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()

    print(f'ECA-TSM-ResNet50 loaded successfully on {device}')
    return model

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

def analyze_video_for_shots_and_extract(video_path, model, n_segment=8, step_size=16, shot_threshold=0.7, clip_info: ClipInfo = None, output_dir="clips", batch_size=256):
    """
    超高效的视频分析: 批量预读取帧、减少IO操作、优化步长
    Args:
        video_path: 视频文件路径
        model: 训练好的TSM模型
        n_segment: 每个窗口的帧数
        step_size: 滑动窗口步长
        shot_threshold: 打门检测的置信度阈值
        clip_info: 包含提取片段前后秒数的信息
        output_dir: 输出目录
        batch_size: 预读取的帧数
    Returns:
        tuple: (shot_detections, extracted_clip_paths)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return [], []
    
    # 创建输出目录
    video_name = video_path.split('/')[-1].split('.')[0]
    output_dir = os.path.join(output_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # print video info
        print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s, {width}x{height}")
        
        shot_detections = []
        extracted_clip_paths = []
        extracted_count = 0
        
        # 计算跳过间隔（以帧为单位）
        skip_frames = int((clip_info.before_secs + clip_info.after_secs) * fps)
        
        current_idx = 0
        frame_cache = {}  # 缓存读取的帧，避免重复读取
        
        while current_idx < total_frames - n_segment + 1:
            # 批量预读取帧
            batch_end = min(current_idx + batch_size, total_frames)
            
            # 读取这一批的所有帧到缓存中
            print(f"Pre-loading frames {current_idx} to {batch_end-1} of {total_frames}...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
            
            for frame_idx in range(current_idx, batch_end):
                if frame_idx not in frame_cache:
                    ret, frame = cap.read()
                    if ret:
                        frame_cache[frame_idx] = frame
                    else:
                        print(f"Warning: Failed to read frame {frame_idx}")
                        break
            
            # 在这一批帧中进行滑动窗口分析
            batch_current = current_idx
            while batch_current < batch_end - n_segment + 1:
                # 检查窗口中的所有帧是否都在缓存中
                window_frames = []
                all_frames_available = True
                
                for i in range(batch_current, batch_current + n_segment):
                    if i in frame_cache:
                        window_frames.append(frame_cache[i])
                    else:
                        all_frames_available = False
                        break
                
                if not all_frames_available or len(window_frames) < n_segment:
                    batch_current += step_size
                    continue
                
                # 预处理当前窗口的帧
                try:
                    input_tensor = preprocess_video_frames(window_frames)
                    input_tensor = input_tensor.to(device)
                    
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        outputs_mean = outputs.mean(dim=0, keepdim=True)
                        probabilities = F.softmax(outputs_mean, dim=1)
                        shot_confidence = probabilities[0][0].item()
                        
                        # 如果检测到打门且置信度超过阈值
                        if shot_confidence > shot_threshold:
                            key_frame_idx = batch_current + n_segment // 2
                            timestamp = key_frame_idx / fps
                            
                            shot_info = {
                                'frame_index': key_frame_idx,
                                'timestamp': timestamp,
                                'confidence': shot_confidence,
                                'window_start': batch_current,
                                'window_end': batch_current + n_segment - 1
                            }
                            
                            shot_detections.append(shot_info)
                            print(f"Shot detected at frame {key_frame_idx} (t={timestamp:.2f}s), confidence: {shot_confidence:.4f}")
                            
                            # 使用缓存的帧直接提取片段
                            start_frame = max(0, int(key_frame_idx - clip_info.before_secs * fps))
                            end_frame = int(key_frame_idx + clip_info.after_secs * fps)
                            
                            extracted_count += 1
                            output_filename = f"shot_{extracted_count}_frame{key_frame_idx}_t{timestamp:.1f}s_conf{shot_confidence:.3f}.mp4"
                            output_path = os.path.join(output_dir, output_filename)
                            
                            # 如果提取范围的帧都在缓存中，直接从缓存提取
                            frames_in_cache = all(f_idx in frame_cache for f_idx in range(start_frame, min(end_frame, batch_end)))
                            
                            if frames_in_cache and end_frame <= batch_end:
                                # 从缓存中提取片段
                                success = extract_video_segment_from_cache(
                                    frame_cache, start_frame, end_frame, fps, width, height, output_path
                                )
                            else:
                                # 回退到原始方法
                                success = extract_video_segment(video_path, start_frame, end_frame, output_path)
                            
                            if success:
                                extracted_clip_paths.append(output_path)
                                print(f"Extracted clip: {output_filename}")
                            else:
                                print(f"Failed to extract clip: {output_filename}")
                                extracted_count -= 1
                            
                            # 跳过已处理的时间范围
                            next_start_frame = end_frame + step_size
                            batch_current = max(batch_current + step_size, next_start_frame)
                            
                            print(f"Skipping to frame {batch_current} to avoid overlap")
                        else:
                            # 没有检测到shot，正常步进
                            batch_current += step_size
                    
                    # 清理GPU内存
                    del input_tensor, outputs, outputs_mean, probabilities
                    if device.type == 'mps':
                        torch.mps.empty_cache()
                    elif device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error processing window starting at frame {batch_current}: {e}")
                    batch_current += step_size
                    continue
            
            # 清理过期的帧缓存，保留可能重叠的帧
            keep_threshold = current_idx + batch_size - n_segment - skip_frames
            frame_cache = {k: v for k, v in frame_cache.items() if k >= keep_threshold}
            
            # 更新当前位置
            current_idx = batch_current if batch_current > current_idx else current_idx + batch_size
            
            # 垃圾回收
            import gc
            gc.collect()
            print(f"Processed up to frame {current_idx}/{total_frames}")
        
        print(f"Successfully analyzed video with {len(shot_detections)} shot detections and {len(extracted_clip_paths)} extracted clips")
        
    except Exception as e:
        print(f"Error during video analysis: {e}")
        return [], []
    finally:
        cap.release()
        import gc
        gc.collect()
    
    return shot_detections, extracted_clip_paths

def extract_video_segment(input_path, start_frame, end_frame, output_path):
    """
    提取视频片段
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open input video {input_path}")
        return False
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 验证视频参数
    if fps <= 0 or width <= 0 or height <= 0:
        print(f"Error: Invalid video parameters - fps:{fps}, width:{width}, height:{height}")
        cap.release()
        return False
    
    # 使用更兼容的编码器设置
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 改用 mp4v 编码器，更兼容
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 检查 VideoWriter 是否成功初始化
    if not out.isOpened():
        print(f"Error: Cannot create output video writer for {output_path}")
        cap.release()
        return False
    
    # 跳到开始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames_written = 0
    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Cannot read frame {frame_idx}")
            break
        
        # 验证帧数据
        if frame is None or frame.size == 0:
            print(f"Warning: Empty frame at {frame_idx}")
            continue
            
        out.write(frame)
        frames_written += 1
    
    cap.release()
    out.release()
    
    # 验证输出文件
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size < 1000:  # 如果文件小于 1KB 认为是损坏的
            print(f"Error: Output file {output_path} is too small ({file_size} bytes), possibly corrupted")
            try:
                os.remove(output_path)  # 删除损坏的文件
            except:
                pass
            return False
        else:
            print(f"Successfully extracted {frames_written} frames to {output_path} ({file_size} bytes)")
            return True
    else:
        print(f"Error: Output file {output_path} was not created")
        return False

def extract_video_segment_from_cache(frame_cache, start_frame, end_frame, fps, width, height, output_path):
    """
    从帧缓存中直接提取视频片段，避免重新读取视频文件
    """
    try:
        # 使用更兼容的编码器设置
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Cannot create output video writer for {output_path}")
            return False
        
        frames_written = 0
        for frame_idx in range(start_frame, end_frame):
            if frame_idx in frame_cache:
                frame = frame_cache[frame_idx]
                if frame is not None and frame.size > 0:
                    out.write(frame)
                    frames_written += 1
                else:
                    print(f"Warning: Invalid frame at {frame_idx}")
            else:
                print(f"Warning: Frame {frame_idx} not in cache")
        
        out.release()
        
        # 验证输出文件
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size < 1000:
                print(f"Error: Output file {output_path} is too small ({file_size} bytes)")
                try:
                    os.remove(output_path)
                except:
                    pass
                return False
            else:
                print(f"Successfully extracted {frames_written} frames from cache to {output_path} ({file_size} bytes)")
                return True
        else:
            print(f"Error: Output file {output_path} was not created")
            return False
            
    except Exception as e:
        print(f"Error extracting from cache: {e}")
        return False

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

"""
Start entry point
"""
def process_video(clip_info: ClipInfo, video_path: str, use_optimized=True) -> ShotResult:
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return None
    
    model = load_model()
    
    n_segment = 8
    step_size = 16  # 增大步长，减少重复计算
    batch_size = 512  # 批量预读取帧数
    shot_threshold = 0.997

    # analyze video and extract clips in one pass
    shot_detections, extracted_clip_paths = analyze_video_for_shots_and_extract(
        video_path,
        model,
        n_segment=n_segment,
        step_size=step_size, 
        shot_threshold=shot_threshold,
        clip_info=clip_info,
        output_dir='clips',
        batch_size=batch_size
    )

    print(f"共检测到 {len(shot_detections)} 个打门动作:")
    for i, detection in enumerate(shot_detections):
        print(f"Shot {i+1}: Frame {detection['frame_index']}, "
              f"Time {detection['timestamp']:.2f}s, "
              f"Confidence {detection['confidence']:.4f}")

    if extracted_clip_paths:
        print(f"\n=== 已提取 {len(extracted_clip_paths)} 个视频片段 ===")
        for i, clip_path in enumerate(extracted_clip_paths):
            print(f"Clip {i+1}: {clip_path}")
    else:
        print("未检测到打门动作")

    shot_result = ShotResult(
        video_path=video_path,
        frame_idx=[detection['frame_index'] for detection in shot_detections],
        timestamp=[detection['timestamp'] for detection in shot_detections],
        confidence=[detection['confidence'] for detection in shot_detections],
        clip_path=extracted_clip_paths
    )

    # 清理资源
    try:
        del model
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    except Exception as e:
        print(f"Warning: Error during resource cleanup: {e}")

    return shot_result    