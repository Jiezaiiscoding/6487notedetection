# 6487notedetection
A note detection which can be used for 2024 crescendo robotic testing
![image](https://github.com/user-attachments/assets/bfa17895-6fe7-4cc2-9835-bdfed99dba79)

# Environment 
YoloV8 https://docs.ultralytics.com/tasks/detect/
![image](https://github.com/user-attachments/assets/c7749f2f-8b3d-45e2-837a-d72c44700d01)
YOLOv8n已经足够用了哦~
第一次用8X结果参数比较大，直接训练崩掉了

CUDA 11.8
如果没有显卡也可以，但是要去找一些在线的平台

# Quick Start
安装python，``pip install ultralytics``安装yolo的ultralytics库

直接运行final_6487_notedetect.py

这里帮助大家设置了两个按钮，可以选择视频进行分析，也可以选择摄像头进行实时画面捕捉，
结果会以文件的名字/拍摄时间保存在output文件夹中

![image](https://github.com/user-attachments/assets/ebc860a3-0c12-4df3-beb9-21e8a0c29c05)

这里做了一个轨迹追踪，能够标记出note发射在空中运行的路径，帮助训练过程的分析

同样你也可以做定点拍摄，根据摄像头在场地的位置和note在画面中的位置来计算和矫正对发射的测试。

# 制作数据集 
接下来给大家展示下制作过程，方便修改和训练其他项目
## 1-图像来源
直接从网络上下载了几支队伍的视频
使用视频编辑软件剪掉视频中没有note的镜头，把视频放慢
对于比较好的片段进行画面的颠倒、翻转等
注意视频的画面可以多选一些场景，比如不同队伍的训练场、比赛的图像等
![image](https://github.com/user-attachments/assets/4d9162f4-5439-414e-8e9a-1c09a334e534)
![image](https://github.com/user-attachments/assets/e870927c-95c7-4022-aee8-49c5d1b782f6)
![image](https://github.com/user-attachments/assets/8cd253ef-36c3-4f5a-820a-755701e3b389)

1_video_data.py
这个程序就是对完整的视频素材进行切分，逐帧保存为图片，放在一个叫做dataset_0的文件夹中

## 2-图像的标记
我用了一个比较快捷的图像标注网站makesense: https://www.makesense.ai/
![image](https://github.com/user-attachments/assets/253a7fe7-d88c-45f5-b921-b423154977b5)
![image](https://github.com/user-attachments/assets/867e6c40-216c-4043-bf6a-52646c97ab2a)

因为是目标检测项目直接选择创建一个新的项目，添加一个唯一类别—— note
![image](https://github.com/user-attachments/assets/3be22edf-906a-4e5d-9c18-8751555146f5)

然后就可以开始标记工作了
标记的过程和注意事项
从视频中我一共获取了近1.3W图片，但是一个人无法完成大量标注，因此最后只标注了600张

但是最后效果仍然不错，主要是因为在标注的时候着重标记了很多飞行中的note
比如这种可能有点糊，甚至可能已经隐藏到Speaker的角落中的，但是不要紧，只管去标记！
![image](https://github.com/user-attachments/assets/736cffaf-7788-4b35-bea3-2d3cec97f413)
有一个细节是，比如下图中这位同学手中有这么多note是否要标注呢？
![image](https://github.com/user-attachments/assets/0501fcb7-a580-4523-ba81-0141835eaaaa)
主要根据最后识别的需求，如果是让机器去找note，那么完全没有问题
但如果像本次是追踪单个note运动轨迹，那么只要标注一个note就可以，标注一堆反而会影响效果

最后点击左上角导出标记结果即可
![image](https://github.com/user-attachments/assets/faa6ab2d-5a6d-47a8-b71a-9fadee86103f)

导出的结果就是TXT文件，每个里面有5个数据，分别对应了检测框的
标注类别（这里只有一种，所以都是0），检测框左上角的点（x1,y1）右下角的点（x2,y2）

## 3-数据集处理
![image](https://github.com/user-attachments/assets/d14b0726-cfea-4edc-9941-fcee894dd222)
面对做好的labels标注文件，很显然是和原有图片数据集不能一一对应的
因此我又写了一段代码，把所有被标记过的图片复制到单独的文件夹中（2_yolo_datasetmaker.py）里面的第一段

![image](https://github.com/user-attachments/assets/4b5e38a7-f14a-4271-8c7e-4f9c89db8e79)

保证里面的文件是一一对应的
![image](https://github.com/user-attachments/assets/f447b390-de50-4b39-9c00-4bc6f5128272)

接着就可以用代码对数据集进行切分，在我提供的代码中，可以直接运行。也可以对训练集、测试机的比例进行调整

![image](https://github.com/user-attachments/assets/65342228-5ae0-4d43-b87d-2aa83de87c9c)

最后会形成这样的数据集
![image](https://github.com/user-attachments/assets/2fd34ea8-6aec-4c35-b3d3-c94eccaf8fd5)

根据数据集的位置完成yaml文件，这个文件会告诉我们要训练的模型如何

注意，yaml里面的文件路径要是绝对路径会比较方便哦

![image](https://github.com/user-attachments/assets/d674a801-bab1-49ce-acb0-4db6d8b52e90)
同时，yaml文件本身请放置在 note_dataset同一级文件夹中

# 训练
训练根据Ultralytics的指示也非常的简单，比如“3_target_train.py”只有短短几行代码

``if __name__ == '__main__':<br>
    import multiprocessing<br>
    multiprocessing.freeze_support()<br>
    from ultralytics import YOLO<br>
    model = YOLO('yolov8n.pt')<br>
    model.train(data='D:/python project/target_detect/note_data.yaml', epochs=200, batch=2, imgsz=640, device=0)<br>
``
其中需要注意①device选择0是因为我用了GPU，如果没有可以不写或者修改为'cpu'

同时训练时每一轮的batch数量不要太多，否则GPU内存会爆满而终止训练

训练好了以后就可以得到最好的一轮结果，存放在最后一个runs/detect/tains...中，找到best.pt文件，拿出来修改一下名字就可以


