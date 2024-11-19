from sentence_transformers import SentenceTransformer
from milvus_operator import text_image_vector, MilvusOperator


def embed_fn(text):
    model = SentenceTransformer('../rag_app/bge-small-zh-v1.5')
    return model.encode(text, normalize_embeddings=True)


def update_image_vector(data_path, operator: MilvusOperator):
    idxs = [0]
    embedings = []
    paths = []
    summary_txt_list = []

    data = [idxs, embedings, paths, summary_txt_list]
    operator.insert_data(data)


if __name__ == '__main__':
    data_dir = r'E:\workspace\ai-ground\dataset\traffic'
    update_image_vector(data_dir, text_image_vector)
