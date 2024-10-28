import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()


# 设置 API 密钥
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# 设置请求的 URL
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}"

# 设置请求头
headers = {
    'Content-Type': 'application/json'
}

# 设置请求体
data = {
    "contents": [{
        "parts": [{"text": "Write a story about a magic backpack."}]
    }]
}

# 发送 POST 请求
proxies = {
    "https": os.getenv('http_proxy'),
    "http": os.getenv('https_proxy'),
}
response = requests.post(url, headers=headers, data=json.dumps(data), proxies=proxies)

# 检查请求是否成功
if response.status_code == 200:
    print("Request was successful!")
    print("Response JSON:")
    print(response.json())
else:
    print(f"Request failed with status code {response.status_code}")
    print("Response text:")
    print(response.text)
