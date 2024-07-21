import cv2
import os

def video_to_frames(video_path, output_folder):
    # 读取视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件： {video_path}")
        return

    # 获取视频的帧率和总帧数
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 逐帧读取并保存图片
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 保存图片
        frame_name = f"{output_folder}/frame_{frame_count}.jpg"
        cv2.imwrite(frame_name, frame)
        frame_count += 1

    # 释放资源
    cap.release()
    print(f"视频 {video_path} 已处理完成，共提取了 {frame_count} 张图片。")

def process_videos_in_folder(input_folder, output_folder):
    # 遍历文件夹中的所有文件
    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)

        # 检查是否为视频文件
        if file.endswith(('.mp4', '.avi', '.mkv', '.flv', '.mov')):
            # 将视频逐帧转换为图片
            video_to_frames(file_path, output_folder)

if __name__ == "__main__":
    input_folder = "video"
    output_folder = "dataset_0"
    process_videos_in_folder(input_folder, output_folder)
