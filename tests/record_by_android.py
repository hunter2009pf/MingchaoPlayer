import cv2
import numpy as np
import torch

from ultralytics import YOLO

# 替换为你手机IP摄像头的URL地址
# video_url = "http://admin:admin@192.168.2.11:8080"
video_url = "http://192.168.2.11:8080/video"


if __name__ == "__main__":
    # 创建一个窗口
    cv2.namedWindow("camera", 1)

    # 使用VideoCapture打开视频流
    cap = cv2.VideoCapture(video_url)

    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    # 假设 'YOLO' 是你用来加载模型的类
    model = YOLO("../models/yolo_models/bestm_cls_v1.pt")
    while True:
        # 从摄像头读取一帧
        ret, image_np = cap.read()

        # 如果正确读取帧，ret为True
        if not ret:
            print("Failed to grab frame")
            break

        # 显示图像
        cv2.imshow("camera", image_np)

        # 按'q'退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            # 如果需要，将BGR转换为RGB
            # 加载NumPy数组代表的图片
            # image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

            # 保存帧为PNG图片
            cv2.imwrite("output.png", image_np)

            # 增加一个维度，因为模型需要批量维度
            image_batch = np.expand_dims(image_np, axis=0)

            # 转换为torch.Tensor
            # image_tensor = torch.from_numpy(image_batch).float()

            # 预测
            with torch.no_grad():
                results = model.predict(image_batch)
                numpy_array = results[0].probs.data.numpy()
                print(numpy_array)
                max_prob = numpy_array[0]
                max_idx = 0
                for i in range(1, 14):
                    if numpy_array[i] > max_prob:
                        max_prob = numpy_array[i]
                        max_idx = i
                print(f"{results[0].names[max_idx]}，置信度是{max_prob}")
            break

    # 释放VideoCapture对象并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()
