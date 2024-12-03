import shortuuid
import requests
import os
import cv2
import base64
import glob
import shutil
from flask import Flask, request, jsonify
import logging
from .minio_uploader import MinioFileUploader

app = Flask(__name__)

# 配置日志
logging.basicConfig(level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)

# 添加文件日志处理器
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)
app.logger.addHandler(file_handler)

# 添加控制台日志处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
app.logger.addHandler(console_handler)


def get_uuid():
    # 生成一个随机的短 UUID
    unique_id = shortuuid.uuid()
    return str.lower(unique_id)


def download_video_to_local(video_minio_url):
    try:
        # 使用requests库下载视频
        response = requests.get(video_minio_url, stream=True)
        response.raise_for_status()

        # 生成一个随机的文件名
        file_name = get_uuid() + ".mp4"
        file_path = os.path.join('/tmp', file_name)

        # 将视频保存到本地
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        app.logger.debug(f"视频下载成功: {file_path}")
        return file_path
    except Exception as e:
        app.logger.error(f"视频下载失败: {str(e)}")
        raise


def extract_frames_from_video(video_url, output_dir, frame_interval=1):
    """
    从视频中每秒抽取一帧并保存到本地。

    :param video_path: 视频文件路径
    :param output_dir: 保存帧的目录
    :param frame_interval: 帧间隔（默认为1秒）
    """
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_url)

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算抽帧间隔
    frame_interval_frames = int(fps * frame_interval)

    # 计算图片名称补全位数
    total_frames_count = total_frames / frame_interval_frames
    num_digits = len(str(int(total_frames_count)))

    # 初始化帧计数器
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval_frames == 0:
            # 构建保存路径
            frame_path = os.path.join(output_dir, f'frame_{frame_count // frame_interval_frames:0{num_digits}d}.jpg')
            # 保存帧
            cv2.imwrite(frame_path, frame)
            # print(f'Saved frame: {frame_path}')

        frame_count += 1

    # 释放视频捕获对象
    cap.release()
    print('Done extracting frames.')


#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        return f"data:image/jpeg;base64,{base64_image}"


def video_frames_and_convert_to_base64(image_folder):
    # 获取文件夹中的所有图片文件
    image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
    result =  [encode_image(image_file) for image_file in image_files]

    # 删除图片文件夹
    shutil.rmtree(image_folder)
    
    return result


def upload_thumbnail_to_oss(object_name, file_path):
    # 创建 MinioFileUploader 实例
    uploader = MinioFileUploader()
    return uploader.upload_file(object_name, file_path)


def extract_frames_and_convert_to_base64(video_url):
    frames_image_folder = os.path.join("/tmp", get_uuid())
    extract_frames_from_video(video_url, frames_image_folder)
    base64_images = video_frames_and_convert_to_base64(frames_image_folder)
    return base64_images