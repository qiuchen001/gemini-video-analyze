from pymilvus import Collection, db, connections
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
 
conn = connections.connect(host=os.getenv("SERVER_HOST"), port=19530)
db.using_database("summary_video_db")
coll_name = 'summary_video_vector'
 
search_params = {
    "metric_type": 'COSINE',
    "offset": 0,
    "ignore_growing": False,
    "params": {"nprobe": 16}
}
 
collection = Collection(coll_name)
collection.load()
 
results = collection.search(
    data=[np.random.normal(0, 0.1, 512).tolist()],
    anns_field="embeding",
    param=search_params,
    limit=16,
    expr=None,
    # output_fields=['m_id', 'embeding', 'desc', 'count'],
    output_fields=['m_id', 'path'],
    consistency_level="Strong"
)
collection.release()
print(results[0].ids)
print(results[0].distances)
hit = results[0][0]
print(hit.entity.get('path'))
print(results)
 