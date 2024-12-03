import os
from pymilvus import MilvusClient, Collection, db, connections
from dotenv import load_dotenv

load_dotenv()

SERVER_HOST = os.getenv("SERVER_HOST")
coll_name = 'summary_video_vector'

client = MilvusClient(
    uri=f"http://{SERVER_HOST}:19530",
    token="root:Milvus"
)
client.using_database("summary_video_db")

results = client.query(
    collection_name=coll_name,
    filter='summary_txt != ""',
    output_fields=["m_id"]
)

print(f"results:{results}")
