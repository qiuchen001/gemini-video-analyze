import os
from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import Collection, db, connections
from dotenv import load_dotenv
load_dotenv()

conn = connections.connect(host=os.getenv("SERVER_HOST"), port=19530)
db.using_database("summary_video_db")

m_id = FieldSchema(name="m_id", dtype=DataType.VARCHAR, is_primary=True, max_length=256)
embeding = FieldSchema(name="embeding", dtype=DataType.FLOAT_VECTOR, dim=512,)
path = FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=256,)
thumbnail_path = FieldSchema(name="thumbnail_path", dtype=DataType.VARCHAR, max_length=256,)
summary_txt = FieldSchema(name="summary_txt", dtype=DataType.VARCHAR, max_length=3072,)
tags = FieldSchema(name="tags", dtype=DataType.ARRAY, max_length=256,)
schema = CollectionSchema(
  fields=[m_id, embeding, path, thumbnail_path, summary_txt, tags],
  description="text to video summary embeding search",
  enable_dynamic_field=True
)

collection_name = "summary_video_vector"
collection = Collection(name=collection_name, schema=schema, using='default', shards_num=2)