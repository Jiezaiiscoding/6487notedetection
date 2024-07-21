
import tkinter as tk
from datetime import datetime
from tkinter import filedialog,messagebox
import cv2
from ultralytics import YOLO
from threading import Thread
import os

# 加载预训练的目标检测模型
model = YOLO("note_0719.pt")


def detect_processing(input_path,output_path):
    cap = cv2.VideoCapture(input_path)
    # 获取视频的基本信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if input_path==0:
        out = cv2.VideoWriter('output/' + datetime.now().strftime('%m%d%H%M') + '.mp4', fourcc, fps, (width, height))
    else:
        file_name = input_path.split('/')[-1]
        out = cv2.VideoWriter('output/'+file_name+'.mp4', fourcc, fps, (width, height))

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
            class_id = int(result[5])  # get id
            if class_id not in tracked_points:
                tracked_points[class_id] = []
            conf = result[4]
            if conf > 0.6:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 128), 2)
                cv2.putText(frame, f"note {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 128), 2)

            tracked_points[class_id].append((cx, cy))
            # 只保留最近40帧的轨迹点
            if len(tracked_points[class_id]) > 40:
                tracked_points[class_id].pop(0)
            # 从近到远绘制轨迹点，颜色透明度递减
            for i, point in enumerate(reversed(tracked_points[class_id])):
                color = (255, 100, 100, 255 - i * 6)  # BGRA格式，最后一个值表示透明度
                cv2.circle(frame, point, 5, color, -1)
        out.write(frame)
        # 显示处理后的视频帧（可选）
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def open_camera():
    # 在这里添加打开摄像头并实时处理画面的代码
    notice.config(text="结果保存在output文件夹,按Q结束")
    file_name = datetime.now().strftime('%m%d%H%M')

    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, file_name)
    # 使用线程来处理视频，避免阻塞GUI
    t = Thread(target=detect_processing, args=(0, output_path))
    t.start()

    messagebox.showinfo("Processing", "Processed")

def open_video():
    # 在这里添加打开视频文件并处理的代码
    file_path = filedialog.askopenfilename(filetypes=[("Video and Image Files", "*.mp4;*.avi;*.mov;*.jpg;*.png;*.jpeg;*.bmp;*.tiff;*.gif"), ("All Files", "*.*")])
    notice.config(text="结果保存在output文件夹")
    file_name = os.path.basename(file_path)
    if file_path:
        output_folder = "output"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, file_name)
        # 使用线程来处理视频，避免阻塞GUI
        t = Thread(target=detect_processing, args=(file_path, output_path))
        t.start()

        messagebox.showinfo("Processing", "Processed")


def start_recording():
    # 在这里添加开始录制处理后的画面的代码
    pass

def stop_recording():
    # 在这里添加停止录制并保存视频的代码
    pass

root = tk.Tk()
root.title("6487 NOTE DETECT")

frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

left_frame = tk.Frame(frame,width=400, height=400)
left_frame.pack(side=tk.LEFT, fill=tk.Y)

open_camera_button = tk.Button(left_frame, text="Open Camera", command=open_camera)
open_camera_button.pack(pady=10)

open_video_button = tk.Button(left_frame, text="Choose MV", command=open_video)
open_video_button.pack(pady=10)

bg_img = tk.PhotoImage(file="6487logo.png")
bg_label = tk.Label(root, image=bg_img)
bg_label.place(x=0, y=150, relwidth=1, relheight=1)

# 在这里添加画面展示区域的代码，例如使用OpenCV的imshow函数
notice = tk.Label(root,font='Heiti 30 bold')
notice.pack(pady=10)


# 运行GUI
root.mainloop()

