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
    You are an intelligent, sensitive and experienced driving assistant. 
    You analyse videos of driving scenarios, observe the driving behaviour of oncoming vehicles in the video and 
    assess whether this behaviour influences the decision-making of the driver of the main vehicle.
    """

    prompt = """
    In driving scenarios, the behavior of other vehicles can significantly influence the primary vehicle's driver's decisions. Below are common potentially dangerous behaviors:

    Sudden Movements
        B1: Sudden Braking: Opposing vehicle brakes unexpectedly.
        B2: Sudden Acceleration/Deceleration: Opposing vehicle changes speed abruptly.
    Lane and Signal Issues
        B3: Lane Changing Without Warning: Opposing vehicle changes lanes without signaling.
        B4: Merging Closely: Opposing vehicle merges too closely.
        B5: Occupying Multiple Lanes: Opposing vehicle occupies two or more lanes due to its size, overloading, or improper operation, forcing the main vehicle driver to adjust their route or speed to avoid collision.
    Traffic Violations
        B6: Driving Against Traffic: Opposing vehicle drives in the wrong direction.
        B7: Jumping the Light: Opposing vehicle runs a red light.
    Inattentive Driving
        B8: Not Using Indicators: Opposing vehicle doesn't signal turns or stops.
        B9: Occupying Blind Spots: Opposing vehicle lingers in blind spots.
    Dangerous Turns and Stops
        B10: Unannounced Turning: Opposing vehicle turns without signaling.
        B11: Sudden Stopping: Opposing vehicle stops unexpectedly.
    Lighting Issues
        B12: Not Using Headlights at Night: Opposing vehicle drives without headlights.
        B13: Using High Beams Inappropriately: Opposing vehicle's high beams cause glare.
    Miscellaneous
        B14: Erratic Driving: Opposing vehicle exhibits unpredictable behavior.
        B15: Improper Parking: Opposing vehicle parks obstructively.

    In the video, the other vehicle is analysed for these behaviours and identified using the assigned ID. There can be multiple behaviours.

    Output JSON Format:
    [
      {
        "analysis": "Detailed analysis of the video scenario...",
        "behaviour": {
          "behaviourId": "B1",
          "behaviourName": "Sudden Braking",
          "timeRange": "Time range during which the behavior occurred" # e.g. "00:00:11-00:00:16"
        }
      },
      {
        "analysis": "Detailed analysis of the video scenario...",
        "behaviour": {
          "behaviourId": "B2",
          "behaviourName": "Sudden Acceleration/Deceleration",
          "timeRange": "Time range during which the behavior occurred" # e.g. "00:00:11-00:00:16"
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
