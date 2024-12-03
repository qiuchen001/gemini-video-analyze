from pymilvus import connections, db, Collection
import os
from dotenv import load_dotenv

load_dotenv()


class MilvusOperator:
    def __init__(self, database, collection, metric_type='COSINE'):
        self.database = database
        self.coll_name = collection
        self.metric_type = metric_type
        self.connect = connections.connect(alias="default", host=os.getenv("SERVER_HOST"), port='19530')
        db.using_database(database)

    def insert_data(self, data):
        collection = Collection(self.coll_name)
        mr = collection.insert(data)

    def search_data(self, embeding, top_k=6):
        collection = Collection(self.coll_name)
        collection.load()

        search_params = {
            "metric_type": self.metric_type,
            "offset": 0,
            "ignore_growing": False,
            "params": {"nprobe": 16}
        }

        results = collection.search(
            data=[embeding],
            anns_field="embeding",
            param=search_params,
            limit=top_k,
            expr=None,
            output_fields=['m_id', 'path', 'summary_txt'],
            consistency_level="Strong"
        )
        entity_list = []
        if results[0] is not None:
            for idx in range(len(results[0])):
                hit = results[0][idx]

                entity_list.append({'m_id': results[0].ids[idx],
                                    'distance': results[0].distances[idx],
                                    'path': hit.entity.get('path')})

        return entity_list

    def query_by_ids(self, ids: list):
        collection = Collection(self.coll_name)
        collection.load()

        str_list = [str(id) for id in ids]
        temp_str = ', '.join(str_list)
        query_expr = f'M_id in [{temp_str}]'

        res = collection.query(
            expr=query_expr,
            offset=0,
            limit=16384,
            output_fields=["m_id", "embeding", "path"],
        )

        return res

    def delete_by_ids(self, ids: list):
        collection = Collection(self.coll_name)
        collection.load()

        str_list = [str(id) for id in ids]
        temp_str = ', '.join(str_list)
        query_expr = f'm_id in [{temp_str}]'
        collection.delete(query_expr)
        return


summary_video_vector = MilvusOperator('summary_video_db', 'summary_video_vector', 'IP')