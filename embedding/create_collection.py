import os
from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import Collection, db, connections
from dotenv import load_dotenv
load_dotenv()

conn = connections.connect(host=os.getenv("SERVER_HOST"), port=19530)
db.using_database("summary_video_db")

m_id = FieldSchema(name="m_id", dtype=DataType.INT64, is_primary=True,)
embeding = FieldSchema(name="embeding", dtype=DataType.FLOAT_VECTOR, dim=768,)
path = FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=256,)
summary_txt = FieldSchema(name="summary_txt", dtype=DataType.VARCHAR, max_length=3072,)
schema = CollectionSchema(
  fields=[m_id, embeding, path, summary_txt],
  description="text to video summary embeding search",
  enable_dynamic_field=True
)

collection_name = "summary_video_vector_v2"
collection = Collection(name=collection_name, schema=schema, using='default', shards_num=2)