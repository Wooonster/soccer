#!/usr/bin/env python3
"""
测试修复后的视频片段提取功能
"""

import os
import cv2
from tsm_process import extract_video_segment

def test_video_extraction():
    # 检查原始视频路径
    uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    clips_dir = os.path.join(os.path.dirname(__file__), 'clips')
    test_dir = os.path.join(clips_dir, 'test')
    
    # 创建测试目录
    os.makedirs(test_dir, exist_ok=True)
    
    # 查找上传的视频文件
    video_files = []
    if os.path.exists(uploads_dir):
        for f in os.listdir(uploads_dir):
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(os.path.join(uploads_dir, f))
    
    if not video_files:
        print("No video files found in uploads directory")
        return
    
    # 使用第一个视频文件进行测试
    video_path = video_files[0]
    print(f"Testing with video: {video_path}")
    
    # 获取视频信息
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS")
    
    # 测试提取一些片段（模拟之前失败的时间段）
    test_segments = [
        (25828, 860.9),  # shot_15 - 应该成功
        (29540, 984.7),  # shot_16 - 之前失败
        (35952, 1198.4), # shot_17 - 之前失败
        (42088, 1402.9), # shot_20 - 之前失败
    ]
    
    for i, (frame_idx, timestamp) in enumerate(test_segments, 1):
        print(f"\n--- Testing segment {i}: frame {frame_idx}, time {timestamp}s ---")
        
        # 计算提取范围（前后各3秒）
        before_frames = int(3 * fps)
        after_frames = int(3 * fps)
        start_frame = max(0, frame_idx - before_frames)
        end_frame = min(total_frames, frame_idx + after_frames)
        
        # 输出文件
        output_filename = f"test_shot_{i}_frame{frame_idx}_t{timestamp:.1f}s.mp4"
        output_path = os.path.join(test_dir, output_filename)
        
        # 提取片段
        success = extract_video_segment(video_path, start_frame, end_frame, output_path)
        
        if success:
            print(f"✅ Successfully extracted: {output_filename}")
        else:
            print(f"❌ Failed to extract: {output_filename}")

if __name__ == "__main__":
    test_video_extraction() 