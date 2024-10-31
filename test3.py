import ffmpeg


def get_thumbnail(video_path, thumbnail_path, time_seconds):
    (
        ffmpeg
        .input(video_path, ss=time_seconds)  # ss参数指定时间点
        .output(thumbnail_path, vframes=1)  # 只输出一帧
        .overwrite_output()  # 使用overwrite_output方法来覆盖输出文件
        .run()
    )


# 示例用法
video_path = r'E:\workspace\ai-ground\videos\mining-well\b7ec1001240181ceb5ec3e448c7f9b78.mp4'
# thumbnail_path = 'b7ec1001240181ceb5ec3e448c7f9b78.mp4-output_thumbnail.jpg'
thumbnail_path = 'b7ec1001240181ceb5ec3e448c7f9b78.mp4_t_10.jpg'
time_seconds = 10  # 获取第10秒的缩略图

get_thumbnail(video_path, thumbnail_path, time_seconds)