import requests
import os
from dotenv import load_dotenv
load_dotenv()

# 本地文件路径
file_path = r'E:\workspace\ai-ground\videos-new\b7ec1001240181ceb5ec3e448c7f9b78.mp4'

# 接口URL
SERVER_HOST = os.getenv("SERVER_HOST")
url = f'http://{SERVER_HOST}:30500/vision-analyze/video/upload'


file_name = "b7ec1001240181ceb5ec3e448c7f9b78.mp4"
# 打开文件并准备上传
with open(file_path, 'rb') as file:
    files = {'video': (file_name, file, 'video/mp4')}
    response = requests.post(url, files=files)

# 打印响应
print(response)
#print(response.status_code)
#print(response.json())
