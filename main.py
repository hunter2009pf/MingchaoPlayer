from ultralytics import YOLO

if __name__=="__main__":
    model = YOLO("D:/digital_human/MingchaoPlayer/models/yolo_models/yolov8n-cls.pt")  # load a pretrained model (recommended for training)
    results = model.train(data="D:/digital_human/yoga_poses", epochs=100, imgsz=640)
    print(results)
    