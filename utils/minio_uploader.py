import os

from minio import Minio
from minio.error import S3Error
from urllib.parse import urljoin
import mimetypes
from dotenv import load_dotenv
load_dotenv()


class MinioFileUploader:
    def __init__(self):
        """
        初始化 MinIO 客户端
        """
        self.minio_client = Minio(
            os.getenv('OSS_ENDPOINT'),  # MinIO 服务端点
            access_key=os.getenv('OSS_ACCESS_KEY'),  # 访问密钥
            secret_key=os.getenv('OSS_SECRET_KEY'),  # 秘密密钥
            secure=False  # 如果你的 Minio 实例没有启用 SSL，请将 secure 参数设置为 False
        )

    def upload_file(self, object_name, file_path):
        """
        上传文件到 MinIO
        :param object_name: 对象名（包含路径）
        :param file_path: 本地文件路径
        """
        bucket_name = os.getenv('OSS_BUCKET_NAME')
        # 检查桶是否存在，如果不存在则创建
        found = self.minio_client.bucket_exists(bucket_name)
        if not found:
            self.minio_client.make_bucket(bucket_name)
            print(f"桶 {bucket_name} 已创建")
        else:
            print(f"桶 {bucket_name} 已存在")

        try:
            # 获取文件的 MIME 类型
            content_type, _ = mimetypes.guess_type(file_path)
            if content_type is None:
                content_type = "application/octet-stream"  # 默认 MIME 类型

            # 上传文件
            self.minio_client.fput_object(bucket_name, object_name, file_path, content_type=content_type)
            print(f"文件 {file_path} 已上传到 {bucket_name}/{object_name}")
        except S3Error as e:
            print(f"上传文件时发生错误: {e}")

        url_prefix = urljoin("http://" + os.getenv('OSS_ENDPOINT'), bucket_name)
        return url_prefix + "/" + object_name


# 示例用法
if __name__ == "__main__":
    # 创建 MinioFileUploader 实例
    uploader = MinioFileUploader()

    # 上传文件
    # bucket_name = os.getenv('OSS_BUCKET_NAME')
    # object_name = "path/to/your/object.txt"
    # file_path = "local/path/to/your/file.txt"

    object_name = "b7ec1001240181ceb5ec3e448c7f9b78.mp4"
    file_path = r"E:\workspace\ai-ground\videos\mining-well\b7ec1001240181ceb5ec3e448c7f9b78.mp4"

    uploader.upload_file(object_name, file_path)