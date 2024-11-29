from sentence_transformers import SentenceTransformer


import sys,os
sys.path.append(os.path.abspath('..'))

from milvus_operator import MilvusOperator, summary_video_vector


def embed_fn(text):
    model = SentenceTransformer('../rag_app/bge-small-zh-v1.5')
    return model.encode(text, normalize_embeddings=True)


# def update_image_vector(embedding, operator: MilvusOperator):
def update_image_vector(data_list):
    # import uuid
    # uuid = str(uuid.uuid4())

    # idxs = [uuid]
    # embedings = [embedding]
    # paths = ["http://10.66.12.37:30946/perception-mining/af9dc016b25ebc8a6d4a84cdd4df62ee.mp4"]
    # summary_txt_list = ["主车行驶在城市道路上，道路两侧有绿化和建筑物。视频开始时，主车前方有一辆白色轿车，随后一辆黄色出租车从左侧超车并占据主车前方车道。主车在接近路口时，黄色出租车未打转向灯突然左转，主车驾驶员对此行为感到不满。"]

    # data = [idxs, embedings, paths, summary_txt_list]
    summary_video_vector.insert_data(data_list)


if __name__ == '__main__':
    text = '飞机'
    embeding = embed_fn(text)
    print(embeding)
    print(embeding.shape)

    # 向量化
    # update_image_vector(embeding, summary_video_vector)

    # 检索
    results = summary_video_vector.search_data(embeding)
    print("results:", results)
    for item in results:
        path = item.get("path")
        print(f"视频地址：{path}")