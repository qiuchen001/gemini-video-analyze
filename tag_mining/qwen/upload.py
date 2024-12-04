import os
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from utils.minio_uploader import MinioFileUploader
from utils.common import *


upload_bp = Blueprint('upload', __name__)

def upload_thumbnail_to_oss(object_name, file_path):
    uploader = MinioFileUploader()
    return uploader.upload_file(object_name, file_path)

@upload_bp.route('/vision-analyze/video/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    filename = secure_filename(video_file.filename)
    video_file_path = os.path.join('/tmp', filename)
    video_file.save(video_file_path)

    try:
        video_oss_url = upload_thumbnail_to_oss(filename, video_file_path)
        print(video_oss_url)

        response = {
            "msg": "success",
            "code": 0,
            "data": {
                "file_name": video_oss_url,
                "video_url": video_oss_url,
            }
        }

        return jsonify(response), 200
    finally:
        os.remove(video_file_path)
        current_app.logger.debug(f"Deleted temporary file: {video_file_path}")