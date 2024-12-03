import os
import json
import ffmpeg
from openai import OpenAI
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from minio_uploader import MinioFileUploader

from prompts import mining
from utils.common import *

from dotenv import load_dotenv
load_dotenv()


model_name = "qwen-vl-max-latest"

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def upload_thumbnail_to_oss(object_name, file_path):
    # 创建 MinioFileUploader 实例
    uploader = MinioFileUploader()
    return uploader.upload_file(object_name, file_path)


# 辅助函数：将秒转换为 '1:23:45' 格式
def seconds_to_time_format(total_seconds):
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02}:{seconds:02}"



def time_to_standard_format(time_range_str):
    """将 '0:13-0:14' 或 '1:23:45-1:23:46' 格式的时间范围统一转换为 '1:23:45-1:23:46' 格式"""

    # 分割时间范围字符串
    start_time_str, end_time_str = time_range_str.split('-')

    # 将开始时间和结束时间分别转换为秒
    start_seconds = time_to_seconds(start_time_str)
    end_seconds = time_to_seconds(end_time_str)

    # 将秒转换为 '1:23:45' 格式
    start_time_formatted = seconds_to_time_format(start_seconds)
    end_time_formatted = seconds_to_time_format(end_seconds)

    # 返回格式化后的时间范围
    return start_time_formatted, end_time_formatted


def time_to_seconds(time_str):
    """将 '0:13' 或 '1:23:45' 格式的时间转换为秒"""
    parts = list(map(int, time_str.split(':')))

    if len(parts) == 2:
        # 格式为 '0:13'
        minutes, seconds = parts
        return minutes * 60 + seconds
    elif len(parts) == 3:
        # 格式为 '1:23:45'
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError("时间格式不正确，应为 '0:13' 或 '1:23:45'")


def generate_thumbnail_from_video(video_url, thumbnail_path, time_seconds):
    if not video_url:
            raise ValueError("视频URL不能为空")
    (
        ffmpeg
        .input(video_url, ss=time_seconds)  # ss参数指定时间点
        .output(thumbnail_path, vframes=1)  # 只输出一帧
        .overwrite_output()  # 使用overwrite_output方法来覆盖输出文件
        .run()
    )


def format_mining_result(mining_result, video_url):
    mining_result_new = []
    for item in mining_result:
        if item['behaviour']['behaviourId'] is None or item['behaviour']['behaviourName'] is None or item['behaviour']['timeRange'] is None:
            continue

        # 格式化时间范围
        start_time_formatted, end_time_formatted = time_to_standard_format(item['behaviour']['timeRange'])
        time_range_str = f"{start_time_formatted}-{end_time_formatted}"
        item['behaviour']['timeRange'] = time_range_str

        # 获得片段的缩略图
        start_time = time_to_seconds(start_time_formatted)
        thumbnail_file_name =  os.path.basename(video_url) + "_t_" + str(start_time) + ".jpg"
        thumbnail_local_path = os.path.join('/tmp', thumbnail_file_name)
        generate_thumbnail_from_video(video_url, thumbnail_local_path, start_time)
        item['thumbnail_url'] = upload_thumbnail_to_oss(thumbnail_file_name, thumbnail_local_path)

        mining_result_new.append(item)

        # 删除临时本地图片文件
        os.remove(thumbnail_local_path)
    return mining_result_new


def get_filename_without_extension(video_local_path):
    """
    从文件路径中提取文件名（不包括扩展名）。

    :param file_path: 文件路径
    :return: 文件名（不包括扩展名）
    """
    # 获取文件名（包括扩展名）
    filename_with_extension = os.path.basename(video_local_path)
    
    # 获取文件名（不包括扩展名）
    filename_without_extension = os.path.splitext(filename_with_extension)[0]
    
    return filename_without_extension


def extract_frames_and_convert_to_base64(video_url):
    # 从视频中进行抽帧
    frames_image_folder = os.path.join("/tmp", get_uuid())
    extract_frames_from_video(video_url, frames_image_folder)

    # 将抽取的视频帧转化为base64
    base64_images = video_frames_and_convert_to_base64(frames_image_folder)

    return base64_images


def mining_video_handler(video_url):
    base64_images = extract_frames_and_convert_to_base64(video_url)

    # 构建消息结构
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": base64_images
                },
                {
                    "type": "text",
                    "text": mining.system_instruction + "\n" + mining.prompt 
                }
            ]
        }
    ]

    # 调用API
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        # temperature=0.1,
        # top_k=40, top_p=0.95
        response_format={"type": "json_object"}
    )

    return response.model_dump_json()


def parse_json_string(json_str):
    # 去除字符串中的转义字符和多余的换行符
    cleaned_str = json_str.replace('\\n', '').replace('\\"', '"')
    
    # 去除字符串开头和结尾的多余符号
    cleaned_str = cleaned_str.strip('```json')
    
    # 解析 JSON 字符串
    parsed_data = json.loads(cleaned_str)
    
    return parsed_data



@app.route('/vision-analyze/video/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    filename = video_file.filename
    video_file_path = os.path.join('/tmp', filename)
    video_file.save(video_file_path)

    try:
        video_oss_url = upload_thumbnail_to_oss(filename, video_file_path)
        print(video_oss_url)

        response = {
            "msg": "success",
            "code": 0,
            "data": {
                "file_name": video_oss_url, # TODO 临时兼容gemini，持久化后优化
                "video_url": video_oss_url
            }
        }

        return jsonify(response), 200
    finally:
        os.remove(video_file_path)
        app.logger.debug(f"Deleted temporary file: {video_file_path}")


@app.route('/vision-analyze/video/mining', methods=['POST'])
def mining_video():
    video_url = request.form.get('file_name')

    try:
        mining_result = mining_video_handler(video_url)
        js = json.loads(mining_result)
        content = js['choices'][0]['message']['content']
        mining_json = parse_json_string(content)

        mining_result_new = format_mining_result(mining_json, video_url)

        response = {
            "msg": "success",
            "code": 0,
            "data": mining_result_new
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=30500,
        debug=True)
