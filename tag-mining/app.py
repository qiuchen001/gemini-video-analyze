import os
import time
import json
import shortuuid
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import logging
from werkzeug.utils import secure_filename

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

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


def upload_to_gemini(file_path, mime_type=None):
    """Uploads the given file to Gemini."""
    app.logger.debug(f"Uploading file...")
    file = genai.upload_file(path=file_path, mime_type=mime_type, name=get_uuid())
    app.logger.debug(f"Completed upload: {file.uri}\n")
    return file


def wait_for_files_active(video_file):
    """Waits for the given files to be active."""
    app.logger.debug("Waiting for file processing...")
    while video_file.state.name == "PROCESSING":
        app.logger.debug('.')
        time.sleep(1)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name != "ACTIVE":
        raise Exception(f"File {video_file.name} failed to process")

    app.logger.debug("...video_file ready\n")
    print()


def main(video_file_name):
    # video_file = upload_to_gemini(video_file_path)
    # wait_for_files_active(video_file)

    video_file = genai.get_file(video_file_name)

    system_instruction = """
    你是一名聪明、敏感、经验丰富的驾驶助理。
    您可以分析驾驶场景视频，观察视频中对向车辆的驾驶行为，并评估这种行为是否会影响主车驾驶员的决策。
    评估这种行为是否会影响主车驾驶员的决策。
    """

    prompt = """
    在驾驶场景中，其他车辆的行为会极大地影响主车驾驶员的决策。以下是常见的潜在危险行为：
    突然移动
        B1: 突然制动： 对手车辆突然刹车。
        B2: 突然加速/减速： 对手车辆突然改变速度。
    车道和信号问题
        B3: 无警告变道： 对手车辆在未发出信号的情况下变更车道。
        B4：并线过近： 对手车辆并线太近。
        B5：占用多个车道： 对手车辆因体积、超载或操作不当而占用两条或两条以上车道，迫使主车驾驶员调整路线或速度以避免碰撞。
    违反交通规则
        B6：逆向行驶： 对手车辆逆向行驶。
        B7：闯红灯： 对手车辆在红灯亮起时驶过人行横道。
    注意力不集中的驾驶
        B8: 不使用指示灯： 对手车辆不发出转弯或停车信号。
        B9: 占用盲点： 对手车辆在盲区逗留。
    危险转弯和停车
        B10: 突然转弯： 对手车辆转弯不打信号。
        B11: 突然停车： 对手车辆意外停车。
    灯光问题
        B12: 夜间不开前灯： 对手车辆行驶时不开大灯。
        B13: 不适当使用远光灯： 对手车辆的远光灯造成眩光。
    杂项
        B14: 驾驶不稳定： 对手车辆表现出不可预测的行为。
        B15：停车不当： 对手车辆妨碍停车。

    在视频中，对对手车辆的行为进行分析，并使用分配的 ID 进行识别。可以有多种行为。

以JSON的格式输出：
    [
      {
        "analysis": "对视频场景的详细分析...",
        "behaviour": {
          "behaviourId": "B1",
          "behaviourName": "突然制动",
          "timeRange": "00:00:11-00:00:1" #  行为发生的时间范围
        }
      },
      {
        "analysis": "对视频场景的详细分析...",
        "behaviour": {
          "behaviourId": "B2",
          "behaviourName": "突然加速/减速",
          "timeRange": "00:00:11-00:00:1" #  行为发生的时间范围
        }
      }
      ...
    ]
    """

    generation_config = {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }
    generation_config = genai.GenerationConfig(**generation_config)

    model = genai.GenerativeModel(model_name=os.getenv('VISION_MODEL'), generation_config=generation_config,
                                  system_instruction=system_instruction)

    app.logger.debug("Making LLM inference request...")
    response = model.generate_content([video_file, prompt],
                                      request_options={"timeout": 600})

    app.logger.debug(response.text)

    return json.loads(response.text)  # 解析JSON响应


@app.route('/vision-analyze/video/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    filename = secure_filename(video_file.filename)
    video_file_path = os.path.join('/tmp', filename)
    video_file.save(video_file_path)

    try:
        video_file = upload_to_gemini(video_file_path)
        wait_for_files_active(video_file)

        response = {
            "msg": "success",
            "code": 0,
            "data": {
                "file_name": video_file.name
            }
        }

        return jsonify(response), 200
    finally:
        os.remove(video_file_path)
        app.logger.debug(f"Deleted temporary file: {video_file_path}")


@app.route('/vision-analyze/video/mining', methods=['POST'])
def mining_video():
    # if 'video' not in request.files:
    #     return jsonify({"error": "No video file provided"}), 400
    #
    # video_file = request.files['video']
    # filename = secure_filename(video_file.filename)
    # video_file_path = os.path.join('/tmp', filename)
    # video_file.save(video_file_path)

    file_name = request.form.get('file_name')

    try:
        result = main(file_name)

        response = {
            "msg": "success",
            "code": 0,
            "data": result
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=30500,
        debug=True)
