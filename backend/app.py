import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import unicodedata

from model_process import get_metadata, process_video

app = Flask(__name__)
CORS(app)  # 启用CORS以允许前端与后端通信

# 配置上传文件的目录
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 4 * 500 * 1024 * 1024  # 限制上传文件大小为 4 GB

def normalize_filename(filename):
    # 规范化 Unicode 字符
    filename = unicodedata.normalize('NFKC', filename)
    # 保留原始文件扩展名
    name, ext = os.path.splitext(filename)
    # 使用 secure_filename 处理基本名称
    safe_name = secure_filename(name)
    # 如果文件名完全被过滤掉了，使用默认名称
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

@app.route('/api/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/api/clip-mode', methods=['POST'])
def clip_metadata():
    data = request.json
    mode = data.get('mode')
    before = data.get('before')
    after = data.get('after')
    print(f'Received clip mode: {mode}, before: {before}, after: {after}')
    clip_info = get_metadata(mode, before, after)
    print(f'Clip info: {clip_info}')
    # process video
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(f'Processing video: {filepath}')
            shot_results = process_video(clip_info, filepath)
            print(f'Shot results: {shot_results}')
            if shot_results:
                return jsonify({
                    'success': True, 
                    'message': 'Video processed successfully',
                    'shot_results': shot_results
                })
    
    return jsonify({
        'success': False, 
        'message': 'No valid videos found in upload folder'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=50001) 