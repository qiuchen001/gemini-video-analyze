import os
import json
import ffmpeg
from openai import OpenAI
from dotenv import load_dotenv
from flask import Blueprint, request, jsonify, current_app
from utils.common import *
from prompts import summary

load_dotenv()

summary_bp = Blueprint('summary', __name__)


model_name = "qwen-vl-max-latest"

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def summary_video_handler(video_url):
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
                    "text": summary.system_instruction + "\n" + summary.prompt 
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


    # video_file = genai.get_file(video_file_name)

    # system_instruction = summary.system_instruction
    # prompt = summary.prompt

    # generation_config = {
    #     "temperature": 0.1,
    #     "top_p": 0.95,
    #     "top_k": 40,
    #     "max_output_tokens": 8192,
    #     "response_mime_type": "application/json",
    # }
    # generation_config = genai.GenerationConfig(**generation_config)

    # model = genai.GenerativeModel(model_name=os.getenv('VISION_MODEL'), generation_config=generation_config,
    #                               system_instruction=system_instruction)

    # app.logger.debug("Making LLM inference request...")
    # response = model.generate_content([video_file, prompt],
    #                                   request_options={"timeout": 600})

    # app.logger.debug(response.text)

    # result = json.loads(response.text)

    # return result

def parse_json_string(json_str):
    cleaned_str = json_str.replace('\\n', '').replace('\\"', '"')
    cleaned_str = cleaned_str.strip('```json')
    parsed_data = json.loads(cleaned_str)
    return parsed_data


@summary_bp.route('/vision-analyze/video/summary', methods=['POST'])
def summary_video():
    file_name = request.form.get('file_name')

    try:
        summary_result = summary_video_handler(file_name)

        js = json.loads(summary_result)
        content = js['choices'][0]['message']['content']
        mining_content_json = parse_json_string(content)
        # mining_result_new = format_mining_result(mining_json, video_url)
        response = {
            "msg": "success",
            "code": 0,
            "data": mining_content_json
        }
        return jsonify(response), 200
    except Exception as e:
        current_app.logger.error(f"Error in mining video: {e}")
        return jsonify({"error": str(e)}), 500