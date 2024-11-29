# 1.先调用上传视频接口
# 2.先调用提取视频摘要的接口
# 3.将摘要信息进行embedding
# 4.embedding入库

import requests,os,json, uuid
from update_vector_V2 import embed_fn, update_image_vector
from milvus_operator import summary_video_vector
from dotenv import load_dotenv
load_dotenv()
server_host = os.getenv("SERVER_HOST")


def upload_video(video_path):
    url = "http://10.66.12.37:30500/vision-analyze/video/upload"
    
    # 从路径中获取文件名
    video_filename = video_path.split('/')[-1]
    
    # 打开文件并准备上传
    with open(video_path, 'rb') as file:
        files = {'video': (video_filename, file, 'video/mp4')}
        response = requests.post(url, files=files)
    
    # 检查响应状态
    if response.status_code == 200:
        result = response.json()
        if result['code'] == 0:
            return result['data']
        else:
            raise Exception(f"上传视频失败: {result['msg']}")
    else:
        raise Exception(f"请求失败,状态码: {response.status_code}")


def summary_video(video_file_name):
    api_url = f"http://{server_host}:30500/vision-analyze/video/summary"
    
    # 构建请求参数
    data = {
        'file_name': video_file_name
    }
    
    # 发送POST请求
    response = requests.post(api_url, data=data)
    
    # 检查响应状态
    if response.status_code == 200:
        result = response.json()
        if result['code'] == 0:
            return result['data']
        else:
            raise Exception(f"提取摘要失败: {result['msg']}")
    else:
        raise Exception(f"请求失败,状态码: {response.status_code}")


if __name__ == "__main__":
    video_dir = r"E:\workspace\ai-ground\videos-new"
    # 获取视频目录下的所有视频文件
    video_path_list = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            # 检查文件扩展名是否为视频格式
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(root, file)
                video_path_list.append(video_path)
    
    print(f"找到 {len(video_path_list)} 个视频文件")




    # exit()
    for video_path in video_path_list:
        # # 上传视频
        # video_info = upload_video(video_path)
        # video_file_name = video_info.get("file_name")
        # print(f"视频名称为：{video_file_name}")
        # # 获得视频摘要
        # video_file_name = "files/he9kqwgyjqlcn2t9k5bd86"
        # summary_video_info = summary_video(video_file_name)
        # 
        # print(f"视频摘要：{summary_video_info}")

        # summary_txt = summary_video_info.get("summary")

        summary_txt = "主车行驶在乡村道路上，道路两侧是山坡。视频开始时，主车以90km/h左右的速度行驶，前方道路弯曲，有连续弯道。在行驶过程中，主车前方道路中央出现了行人乱窜的情况，主车采取了车辆急刹的操作，避免了安全事故的发生。"
        video_path = "http://{server_host}:30946/perception-mining/4cf55cb9b1a4313ebac796d83482f9ef.mp4"
        
        # 将摘要进行向量化
        embeding = embed_fn(summary_txt)

        data_list = []
        data_info = {
            "m_id": str(uuid.uuid4()),
            "embeding": embeding,
            "path": video_path,
            "thumbnail_path": "",
            "summary_txt": summary_txt,
        }
        data_list.append(data_info)
        
        # 向量化
        # update_image_vector(embeding, summary_video_vector)
        update_image_vector(data_list)
        exit()
