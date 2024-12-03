import os
import glob
import base64
from openai import OpenAI
import cv2
import json
from dotenv import load_dotenv
load_dotenv()


model_name = "qwen-vl-max-latest"

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


prompt = """
你是一名专业的驾驶行为分析助手，负责精准分析驾驶场景视频。你的任务是仔细观察视频中对手车辆的驾驶行为，并准确评估这些行为是否会影响主车驾驶员的决策。在驾驶过程中，其他车辆的行为会极大地影响主车驾驶员的决策。以下是常见的潜在危险行为：

- **突然移动**
  - B1: 突然制动：对手车辆突然刹车。
  - B2: 突然加速/减速：对手车辆突然改变速度。
- **车道和信号问题**
  - B3: 无警告变道：对手车辆在未发出信号的情况下变更车道。
  - B4: 并线过近：对手车辆并线太近。
  - B5: 占用多个车道：对手车辆因体积、超载或操作不当而占用两条或两条以上车道（如车身跨越车道分界线），迫使主车驾驶员调整路线或速度以避免碰撞。
- **违反交通规则**
  - B6: 逆向行驶：对手车辆逆向行驶。
  - B7: 闯红灯：对手车辆在红灯亮起时驶过人行横道。
- **注意力不集中的驾驶**
  - B8: 不使用指示灯：对手车辆不发出转弯或停车信号。
  - B9: 占用盲点：对手车辆在盲区逗留。
- **危险转弯和停车**
  - B10: 突然转弯：对手车辆转弯不打信号。
  - B11: 突然停车：对手车辆意外停车。
- **灯光问题**
  - B12: 夜间不开前灯：对手车辆行驶时不开大灯。
  - B13: 不适当使用远光灯：对手车辆的远光灯造成眩光。
- **杂项**
  - B14: 驾驶不稳定：对手车辆表现出不可预测的行为。
  - B15：停车不当：对手车辆妨碍停车。

对视频中的对手车辆行为进行详细分析，并使用上述分配的 ID 进行识别；
如果匹配了行为，由于提供的视频是一秒一帧，请务必严格按照视频开始的相对时间来确定对手车辆行为发生的时间范围，格式为固定的“时:分：秒”格式。
视频的开始时间为：00:00:00，如果一个行为发生在第3帧（或第2秒）开始，并在第5帧（或第4秒）开始时结束，则timeRange为：00:00:02-00:00:04。

最终，请严格按照以下 JSON 格式输出结果，你应该以json字符串的形式输出，方面用户进行解析：
{
"videoTime": 16,  # 视频的总时长
"list": [
  {
    "analysis": "对视频场景的详细分析...",
    "behaviour": {
      "behaviourId": "B1",
      "behaviourName": "突然制动",
      "timeRange": "实际发生的准确时间范围"
    }
  },
  {
    "analysis": "对视频场景的详细分析...",
    "behaviour": {
      "behaviourId": "B2",
      "behaviourName": "突然加速/减速",
      "timeRange": "实际发生的准确时间范围"
    }
  },
  {
    "analysis": "对视频场景的详细分析...",
    "behaviour": {
      "behaviourId": "B3",
      "behaviourName": "无警告变道",
      "timeRange": "实际发生的准确时间范围"
    }
  },
  // 其他行为以此类推
]

}

请确保每条记录都包含详细的分析描述、行为 ID、行为名称以及严格按照视频实际情况确定的行为发生的时间范围。            

"""


#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        return f"data:image/jpeg;base64,{base64_image}"


def extract_frames_from_video(video_path, output_dir, frame_interval=1):
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
    cap = cv2.VideoCapture(video_path)

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 计算每一秒的帧数
    frame_interval_frames = int(fps * frame_interval)

    # 初始化帧计数器
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 每秒抽取一帧
        if frame_count % frame_interval_frames == 0:
            # 构建保存路径
            frame_path = os.path.join(output_dir, f'frame_{frame_count // frame_interval_frames:04d}.jpg')
            # 保存帧
            cv2.imwrite(frame_path, frame)
            # print(f'Saved frame: {frame_path}')

        frame_count += 1

    # 释放视频捕获对象
    cap.release()
    print('Done extracting frames.')


def describe_video_process(image_folder):
    """
    描述视频的具体过程。

    :param image_folder: 包含图片的文件夹路径
    :param api_key: 百炼API Key
    """
    # 获取文件夹中的所有图片文件
    image_files = glob.glob(os.path.join(image_folder, "*.jpg"))

    # 按照文件名中的数字部分排序
    image_files = sorted(image_files, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))


    # 添加固定前缀
    prefixed_image_files = [rf"E:\playground\ai\projects\gemini-vision-perception\{file}" for file in image_files]
    encoded_images = [encode_image(image_path) for image_path in prefixed_image_files]

    # print("视频图片：", prefixed_image_files)

    # 构建消息结构
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": encoded_images
                },
                {
                    "type": "text",
                    "text": prompt
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


def get_filename_without_extension(file_path):
    """
    从文件路径中提取文件名（不包括扩展名）。

    :param file_path: 文件路径
    :return: 文件名（不包括扩展名）
    """
    # 获取文件名（包括扩展名）
    filename_with_extension = os.path.basename(file_path)
    
    # 获取文件名（不包括扩展名）
    filename_without_extension = os.path.splitext(filename_with_extension)[0]
    
    return filename_without_extension


def append_to_file(file_path, content):
    """
    向文件中追加内容。

    :param file_path: 文件路径
    :param content: 要追加的内容
    """
    # 以追加模式打开文件
    with open(file_path, 'a', encoding='utf-8') as file:
        # 写入内容
        file.write(content)
        # 写入换行符（可选）
        file.write('\n')


def parse_json_string(json_str):
    # 去除字符串中的转义字符和多余的换行符
    cleaned_str = json_str.replace('\\n', '').replace('\\"', '"')
    
    # 去除字符串开头和结尾的多余符号
    cleaned_str = cleaned_str.strip('```json')
    
    # 解析 JSON 字符串
    # parsed_data = json.loads(cleaned_str)
    
    return cleaned_str


def extract_frames_and_convert_to_base64(video_url):
    """
    从在线视频地址中提取每一秒的帧，并将图片转化为base64编码。

    :param video_url: 在线视频地址
    :return: 包含base64编码图片的字典
    """
    # 创建一个临时文件夹来存储提取的帧
    temp_folder = "temp_frames"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    # 下载视频并提取帧
    video_path = os.path.join(temp_folder, "temp_video.mp4")
    os.system(f"wget -O {video_path} {video_url}")

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    base64_images = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每秒提取一帧
        if frame_count % fps == 0:
            frame_filename = f"{temp_folder}/frame_{frame_count}.jpg"
            cv2.imwrite(frame_filename, frame)

            # 将图片转化为base64
            with open(frame_filename, "rb") as image_file:
                base64_images[frame_count] = base64.b64encode(image_file.read()).decode('utf-8')

        frame_count += 1

    cap.release()

    # 删除临时文件夹
    os.system(f"rm -rf {temp_folder}")

    return base64_images


video_path_list = [
    # r"E:\workspace\ai-ground\videos-new\af9dc016b25ebc8a6d4a84cdd4df62ee.mp4",

    "http://10.66.12.37:30946/perception-mining/b7ec1001240181ceb5ec3e448c7f9b78.mp4"
    
    # "file:///home/me/2TSSD/workspace/ai-ground/dataset/videos/videos-new/af9dc016b25ebc8a6d4a84cdd4df62ee.mp4",
    # "file:///home/me/2TSSD/workspace/ai-ground/dataset/videos/videos-new/5994428260466ae3f48c816ca8f8d68b.mp4",
    # "file:///home/me/2TSSD/workspace/ai-ground/dataset/videos/videos-new/54415cb81748d4f86b34b5b0cdf435db.mp4",
    # "file:///home/me/2TSSD/workspace/ai-ground/dataset/videos/videos-new/58fa05cf51e916ad2b824867b92060c7.mp4",
    # "file:///home/me/2TSSD/workspace/ai-ground/dataset/videos/videos-new/8b1de0729c40bedd7c28936f894b6625.mp4",
]

for video_path in video_path_list:
    image_folder = get_filename_without_extension(video_path)
    extract_frames_from_video(video_path, image_folder)

    res = describe_video_process(image_folder)
    print(res)

    js = json.loads(res)
    content = js['choices'][0]['message']['content']
    # exit()

    # text = res.choices[0].message.content

    text = parse_json_string(content)

    json_obj = {
        "video": image_folder,
        "text": text,
        "model": model_name
    }

    # 将 JSON 对象序列化为 JSON 字符串
    json_string = json.dumps(json_obj, ensure_ascii=False)

    # 示例调用
    file_path = image_folder + '.txt'
    append_to_file(file_path, json_string)

    


    # exit()