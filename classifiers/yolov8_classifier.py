import datetime
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def classify_pose(queue):
    is_available = torch.cuda.is_available()
    print(f"CUDA is {'' if is_available else 'not'} available.")
    if is_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # 加载模型
    model = YOLO("../models/yolo_models/bestm_cls_v2.pt")
    model.to(device)

    # print("Start classifying pose:", date.strftime("%Y-%m-%d %H:%M:%S"))

    while True:
        # 打印起始时间
        start_date = datetime.datetime.now()
        # 子进程从队列中取出数据
        image_np = queue.get()
        if image_np is None:
            break
        print("Classifying pose...")

        # 将图像从 BGR 转换为 RGB
        frame = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # 将 uint8 类型转换为 float32
        frame = frame.astype(np.float32)

        # 调整图像大小（如果需要）
        # frame = cv2.resize(frame, (640, 640))

        # 添加批处理维度 [1, 3, H, W]
        frame = np.expand_dims(frame, axis=0)

        # 转换为 PyTorch Tensor
        image_tsr = torch.from_numpy(frame)

        image_tsr = image_tsr.permute(0, 3, 1, 2)

        # 根据 PyTorch 模型的需要，可能还需要除以 255 进行归一化
        image_tsr /= 255.0

        results = model(image_tsr, conf=0.7)
        # Now convert to NumPy array
        numpy_array = results[0].probs.data.cpu().numpy()
        max_prob = numpy_array[0]
        max_idx = 0
        for i in range(1, 14):
            if numpy_array[i] > max_prob:
                max_prob = numpy_array[i]
                max_idx = i
        print(f"动作类型是：{results[0].names[max_idx]}，置信度是{max_prob}")

        # 打印结束时间
        end_date = datetime.datetime.now()
        print("Cope with one frame, time consumption: ", end_date - start_date)
