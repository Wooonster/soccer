import os
import unicodedata
import atexit
import gc

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
# from eca_process import get_metadata, process_video
from tsm_process import get_metadata, process_video

app = Flask(__name__)
CORS(app)  # 启用CORS以允许前端与后端通信

# 配置上传文件的目录
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv', 'png'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 4 * 500 * 1024 * 1024  # 限制上传文件大小为 4 GB

# 全局变量 存储 clip_info
current_clip_info = None

def normalize_filename(filename):
    # 规范化 Unicode 字符
    filename = unicodedata.normalize('NFKC', filename)
    name, ext = os.path.splitext(filename)
    safe_name = secure_filename(name)

    if not safe_name:
        safe_name = 'video'
    return safe_name + ext

def allowed_file(filename):
    """
    文件类型检查 确保上传的文件是允许的视频格式
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    # 检查请求中是否有文件
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # 如果用户没有选择文件，浏览器会提交一个没有文件名的空文件部分
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # 处理文件名，支持 UTF-8
        filename = normalize_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # 如果文件已存在，添加数字后缀
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(filepath):
            filename = f"{base}_{counter}{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            counter += 1
            
        file.save(filepath)
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'filename': filename,
            'filepath': filepath
        }), 200
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/videos', methods=['GET'])
def get_videos():
    videos = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            videos.append(filename)
    return jsonify(videos)

@app.route('/api/download/<path:filename>')
def download_file(filename):
    # 尝试从上传文件夹获取
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(upload_path):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    
    # 尝试从clips文件夹获取（支持嵌套路径）
    clips_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clips')
    
    # 如果filename包含路径分隔符，直接使用相对路径
    if '/' in filename or '\\' in filename:
        clip_path = os.path.join(clips_folder, filename)
        if os.path.exists(clip_path):
            # 分离目录和文件名
            dir_path, file_name = os.path.split(filename)
            full_dir_path = os.path.join(clips_folder, dir_path)
            return send_from_directory(full_dir_path, file_name, as_attachment=True)
    else:
        # 如果filename只是文件名，搜索所有子目录
        clip_path = os.path.join(clips_folder, filename)
        if os.path.exists(clip_path):
            return send_from_directory(clips_folder, filename, as_attachment=True)
        
        # 在子目录中搜索
        for root, dirs, files in os.walk(clips_folder):
            if filename in files:
                return send_from_directory(root, filename, as_attachment=True)
    
    # 文件不存在
    return jsonify({'error': f'File not found: {filename}'}), 404

@app.route('/api/clip-mode', methods=['POST'])
def clip_metadata():
    """设置剪辑参数，但不执行处理"""
    global current_clip_info
    data = request.json
    mode = data.get('mode')
    before = data.get('before')
    after = data.get('after')
    print(f'Received clip mode: {mode}, before: {before}, after: {after}')
    
    # 存储 clip_info 到全局变量
    current_clip_info = get_metadata(mode, before, after)
    print(f'Clip info set: {current_clip_info}')
    
    return jsonify({
        'success': True, 
        'message': 'Clip parameters saved successfully',
        'clip_info': {
            'mode': mode,
            'before': before,
            'after': after
        }
    })

@app.route('/api/process-video', methods=['POST'])
def process_videos():
    """执行视频处理"""
    global current_clip_info
    
    # 检查是否已设置剪辑参数
    if current_clip_info is None:
        return jsonify({
            'success': False,
            'message': 'No clip parameters set. Please set clip mode first.'
        }), 400
    
    # 可选：从请求中获取文件名，如果指定了特定文件
    data = request.json
    specific_filename = data.get('filename', None)
    
    # 处理视频
    if specific_filename:
        # 处理特定视频
        print(f"process specific video: {specific_filename}")
        if allowed_file(specific_filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], specific_filename)
            if os.path.exists(filepath):
                print(f'Processing specific video: {filepath}')
                shot_result = process_video(current_clip_info, filepath)
                
                # 合并所有片段到一个视频
                merged_video_path = None
                if shot_result and len(shot_result.clip_path) > 0:
                    merged_video_path = merge_clips(shot_result.clip_path)
                
                if shot_result:
                    return jsonify({
                        'success': True,
                        'message': f'Video {specific_filename} processed successfully',
                        'shot_result': {
                            'video_path': shot_result.video_path,
                            'frame_count': len(shot_result.frame_idx),
                            'timestamps': shot_result.timestamp,
                            'confidences': shot_result.confidence,
                            'clip_paths': shot_result.clip_path
                        },
                        'merged_video': merged_video_path
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': f'No shot frames detected in {specific_filename}'
                    })
            else:
                return jsonify({
                    'success': False,
                    'message': f'File {specific_filename} not found'
                }), 404
    else:
        print(f"process all videos")
        # 处理所有视频
        results = []
        all_clips = []
        
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if allowed_file(filename):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                print(f'Processing video: {filepath}')
                shot_result = process_video(current_clip_info, filepath)
                if shot_result:
                    results.append({
                        'filename': filename,
                        'frame_count': len(shot_result.frame_idx),
                        'timestamps': shot_result.timestamp,
                        'confidence': shot_result.confidence,
                        'clip_paths': shot_result.clip_path,
                        'frame_idx': shot_result.frame_idx
                    })
                    all_clips.extend(shot_result.clip_path)
        
        # 合并所有视频的所有片段
        merged_video_path = None
        if all_clips:
            merged_video_path = merge_clips(all_clips)
        
        if results:
            return jsonify({
                'success': True,
                'message': 'All videos processed successfully',
                'results': results,
                'merged_video': merged_video_path
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No shot frames detected in any video'
            })
        

def merge_clips(clip_paths):
    """合并多个视频片段为一个视频"""
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips
        import tempfile
        import atexit
        
        # 确保clips文件夹存在
        clips_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clips')
        
        # 加载所有片段
        clips = []
        temp_files = []
        
        try:
            for clip_path in clip_paths:
                # 处理不同的路径格式
                full_path = None
                
                if os.path.isabs(clip_path):
                    # 绝对路径，直接使用
                    full_path = clip_path
                elif clip_path.startswith('clips/'):
                    # 相对于项目根目录的路径
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    full_path = os.path.join(project_root, clip_path)
                elif '/' in clip_path:
                    # 包含路径分隔符，相对于clips文件夹
                    full_path = os.path.join(clips_folder, clip_path)
                else:
                    # 只是文件名，在clips文件夹及其子目录中搜索
                    full_path = os.path.join(clips_folder, clip_path)
                    if not os.path.exists(full_path):
                        # 在子目录中搜索
                        for root, dirs, files in os.walk(clips_folder):
                            if os.path.basename(clip_path) in files:
                                full_path = os.path.join(root, os.path.basename(clip_path))
                                break
                
                print(f"Attempting to load clip: {clip_path} -> {full_path}")
                
                if full_path and os.path.exists(full_path):
                    clip = VideoFileClip(full_path)
                    clips.append(clip)
                    print(f"Successfully loaded clip: {full_path}")
                else:
                    print(f"Warning: Clip file not found: {clip_path} (resolved to: {full_path})")
            
            if not clips:
                print("No valid clips to merge")
                return None
            
            print(f"Merging {len(clips)} clips...")
            
            # 合并片段
            final_clip = concatenate_videoclips(clips, method='compose')
            
            # 保存合并后的视频
            output_path = os.path.join(clips_folder, "merged_video.mp4")
            
            # 使用临时文件避免资源冲突
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_path = temp_file.name
                temp_files.append(temp_path)
            
            # 写入临时文件
            final_clip.write_videofile(
                temp_path, 
                codec='libx264', 
                audio_codec='aac',
                verbose=False,
                logger=None
            )
            
            # 移动到最终位置
            import shutil
            shutil.move(temp_path, output_path)
            
        finally:
            # 确保所有资源都被释放
            for clip in clips:
                try:
                    if hasattr(clip, 'close'):
                        clip.close()
                    if hasattr(clip, 'reader') and clip.reader:
                        clip.reader.close()
                except:
                    pass
            
            try:
                if 'final_clip' in locals():
                    final_clip.close()
                    if hasattr(final_clip, 'reader') and final_clip.reader:
                        final_clip.reader.close()
            except:
                pass
            
            # 清理临时文件
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass
        
        print(f"Merged video saved to {output_path}")
        return "merged_video.mp4"
        
    except Exception as e:
        print(f"Error merging clips: {e}")
        return None


@app.route('/api/clear-all', methods=['POST'])
def clear_all_data():
    """清除所有上传的文件和处理结果"""
    try:
        # 清空上传文件夹
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
                print(f"Deleted uploaded file: {filepath}")
        
        # 清空clips文件夹（包括子目录）
        clips_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clips')
        if os.path.exists(clips_folder):
            import shutil
            # 删除整个clips目录及其内容
            shutil.rmtree(clips_folder)
            print(f"Deleted clips folder: {clips_folder}")
            
            # 重新创建空的clips目录
            os.makedirs(clips_folder, exist_ok=True)
            print(f"Recreated empty clips folder: {clips_folder}")
        
        # 重置全局变量
        global current_clip_info
        current_clip_info = None
        
        return jsonify({
            'success': True,
            'message': 'All data has been cleared successfully'
        })
    except Exception as e:
        print(f"Error clearing data: {e}")
        return jsonify({
            'success': False,
            'message': f'Error clearing data: {str(e)}'
        }), 500

def cleanup():
    """应用退出时的资源清理函数"""
    try:
        # 强制垃圾回收
        gc.collect()
        print("Application cleanup completed")
    except Exception as e:
        print(f"Error during cleanup: {e}")

if __name__ == '__main__':
    # 注册退出时的清理函数
    atexit.register(cleanup)
    app.run(debug=True, host='0.0.0.0', port=50001) 