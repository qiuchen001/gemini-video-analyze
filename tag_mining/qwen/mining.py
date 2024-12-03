import os
import json
import ffmpeg
from openai import OpenAI
from dotenv import load_dotenv
from flask import Blueprint, request, jsonify, current_app
from utils.common import *
from ..prompts import mining

load_dotenv()

mining_bp = Blueprint('mining', __name__)

model_name = "qwen-vl-max-latest"

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def seconds_to_time_format(total_seconds):
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02}:{seconds:02}"

def time_to_standard_format(time_range_str):
    start_time_str, end_time_str = time_range_str.split('-')
    start_seconds = time_to_seconds(start_time_str)
    end_seconds = time_to_seconds(end_time_str)
    start_time_formatted = seconds_to_time_format(start_seconds)
    end_time_formatted = seconds_to_time_format(end_seconds)
    return start_time_formatted, end_time_formatted

def time_to_seconds(time_str):
    parts = list(map(int, time_str.split(':')))
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError("时间格式不正确，应为 '0:13' 或 '1:23:45'")

def generate_thumbnail_from_video(video_url, thumbnail_path, time_seconds):
    if not video_url:
        raise ValueError("视频URL不能为空")
    (
        ffmpeg
        .input(video_url, ss=time_seconds)
        .output(thumbnail_path, vframes=1)
        .overwrite_output()
        .run()
    )

def format_mining_result(mining_result, video_url):
    mining_result_new = []
    for item in mining_result:
        if item['behaviour']['behaviourId'] is None or item['behaviour']['behaviourName'] is None or item['behaviour']['timeRange'] is None:
            continue
        start_time_formatted, end_time_formatted = time_to_standard_format(item['behaviour']['timeRange'])
        time_range_str = f"{start_time_formatted}-{end_time_formatted}"
        item['behaviour']['timeRange'] = time_range_str
        start_time = time_to_seconds(start_time_formatted)
        thumbnail_file_name =  os.path.basename(video_url) + "_t_" + str(start_time) + ".jpg"
        thumbnail_local_path = os.path.join('/tmp', thumbnail_file_name)
        generate_thumbnail_from_video(video_url, thumbnail_local_path, start_time)
        item['thumbnail_url'] = upload_thumbnail_to_oss(thumbnail_file_name, thumbnail_local_path)
        mining_result_new.append(item)
        os.remove(thumbnail_local_path)
    return mining_result_new

# def extract_frames_and_convert_to_base64(video_url):
#     frames_image_folder = os.path.join("/tmp", get_uuid())
#     extract_frames_from_video(video_url, frames_image_folder)
#     base64_images = video_frames_and_convert_to_base64(frames_image_folder)
#     return base64_images

def mining_video_handler(video_url):
    base64_images = extract_frames_and_convert_to_base64(video_url)
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
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        response_format={"type": "json_object"}
    )
    return response.model_dump_json()

def parse_json_string(json_str):
    cleaned_str = json_str.replace('\\n', '').replace('\\"', '"')
    cleaned_str = cleaned_str.strip('```json')
    parsed_data = json.loads(cleaned_str)
    return parsed_data

@mining_bp.route('/vision-analyze/video/mining', methods=['POST'])
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
        current_app.logger.error(f"Error in mining video: {e}")
        return jsonify({"error": str(e)}), 500