if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    from ultralytics import YOLO
    model = YOLO('note_0719.pt')#
    model.train(data='D:/python project/target_detect/note_data.yaml', epochs=200, batch=2, imgsz=640, device=0)
