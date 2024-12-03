import os
import json
import ffmpeg
import requests,os,json, uuid
from openai import OpenAI
from dotenv import load_dotenv
from flask import Blueprint, request, jsonify, current_app
from utils.common import *
from sentence_transformers import SentenceTransformer
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from embedding.update_vector_V2 import embed_fn, update_image_vector
from embedding.milvus_operator import summary_video_vector



load_dotenv()

embedding_summary_retrieval_bp = Blueprint('embedding_summary_retrieval', __name__)


model_name = "qwen-vl-max-latest"

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def embed_fn(text):
    model = SentenceTransformer('../rag_app/bge-small-zh-v1.5')
    return model.encode(text, normalize_embeddings=True)


@embedding_summary_retrieval_bp.route('/vision-analyze/video/embedding-summary-retrieval', methods=['POST'])
def embedding_summary_video():
    txt = request.form.get("txt")
    # video_url = request.form.get("video_url")

    embeding = embed_fn(txt)
    # print(embeding)
    # print(embeding.shape)


    results = summary_video_vector.search_data(embeding)
    print("results:", results)
    for item in results:
        path = item.get("path")
        print(f"视频地址：{path}")

        # 向量化
    # update_image_vector(embeding, summary_video_vector)


    # data_list = []
    # m_id = str(uuid.uuid4())
    # data_info = {
    #     "m_id": m_id,
    #     "embeding": embeding,
    #     "path": video_url,
    #     "thumbnail_path": "",
    #     "summary_txt": summary_txt,
    # }
    # data_list.append(data_info)
    
    # # 向量化
    # # update_image_vector(embeding, summary_video_vector)
    # update_image_vector(data_list)

    response = {
        "msg": "success",
        "code": 0,
        "data": {
            "list": results
        }
    }
    return jsonify(response), 200
    
