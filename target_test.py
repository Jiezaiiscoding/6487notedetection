import cv2
import torch
from ultralytics import YOLO
import numpy as np

# 加载预训练的目标检测模型
model = YOLO("note_0719.pt")

# 打开视频文件
video_path = "video/6487.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频的基本信息
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

# 初始化轨迹列表
tracked_points = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用模型进行目标检测
    results = model(frame)[0]

    # 绘制检测到的目标中心轨迹
    for result in results.boxes.data:
        x1, y1, x2, y2 = result[:4].int().tolist()
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        class_id = int(result[5])#get id

        if class_id not in tracked_points:
            tracked_points[class_id] = []


        conf = result[4]
        if conf > 0.6:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 128), 2)
            cv2.putText(frame,f"note {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 128), 2)

        tracked_points[class_id].append((cx, cy))

        # 只保留最近40帧的轨迹点
        if len(tracked_points[class_id]) > 40:
            tracked_points[class_id].pop(0)

        # 从近到远绘制轨迹点，颜色透明度递减
        for i, point in enumerate(reversed(tracked_points[class_id])):
            color = (255, 100, 100, 255 - i * 6)  # BGRA格式，最后一个值表示透明度
            cv2.circle(frame, point, 5, color, -1)

    # for class_id in tracked_points:
    #     if len(tracked_points[class_id]) > 1:
    #         points = np.array(tracked_points[class_id], np.int32)
    #         cv2.polylines(frame, [points], False, (172, 98, 0), thickness=4)
    # 将带有绘制路径的视频帧写入输出文件
    out.write(frame)

    # 显示处理后的视频帧（可选）
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()