from pymilvus import connections, db
import os
from dotenv import load_dotenv

load_dotenv()


conn = connections.connect(host=os.getenv("SERVER_HOST"), port=19530)
database = db.create_database("summary_video_db")

db.using_database("summary_video_db")
print(db.list_database())
