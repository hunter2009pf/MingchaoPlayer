from ultralytics import YOLO


if __name__=="__main__":
    model = YOLO("D:/digital_human/MingchaoPlayer/runs/classify/train/weights/best.pt")
    results = model("D:/digital_human/MingchaoPlayer/images/148.png", conf=0.7)
    print(results[0].probs.data)
