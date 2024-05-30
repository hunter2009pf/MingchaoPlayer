import cv2
from ultralytics import YOLO
from config.custom_config import *


def connect_camera(queue):
    # 创建一个窗口
    cv2.namedWindow("camera", 1)

    # 使用VideoCapture打开视频流
    cap = cv2.VideoCapture(IP_WEBCAM_URL)

    if cap.isOpened():
        cnt = 0
        while True:
            cnt += 1
            # 从摄像头读取一帧
            is_ok, image_np = cap.read()

            # 如果正确读取帧，ret为True
            if not is_ok:
                print("Failed to grab frame")
                break

            # 将图片信息放入队列
            if cnt % POSE_CLASSIFICATION_FRAME_INTERVAL == 0:
                queue.put(image_np)

            # 显示图像
            cv2.imshow("camera", image_np)

            # 按'q'退出循环
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        print("Failed to connect camera")

    queue.put(None)
    # 释放VideoCapture对象并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()
