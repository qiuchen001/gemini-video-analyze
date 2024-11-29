from pymilvus import Collection, utility, connections, db
import os
from dotenv import load_dotenv

load_dotenv()

conn = connections.connect(host=os.getenv("SERVER_HOST"), port=19530)
db.using_database("summary_video_db")

index_params = {
  "metric_type": "IP",
  "index_type": "IVF_FLAT",
  "params": {"nlist": 512}
}

collection = Collection("summary_video_vector")
collection.create_index(
  field_name="embeding",
  index_params=index_params
)

utility.index_building_progress("summary_video_vector")