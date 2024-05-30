import multiprocessing
from ultralytics import YOLO

from camera.ip_webcam import connect_camera
from classifiers.yolov8_classifier import classify_pose


if __name__ == "__main__":
    queue = multiprocessing.Queue()
    # 创建子进程
    process = multiprocessing.Process(target=classify_pose, args=(queue,))
    process.start()
    connect_camera(queue)
    # 等待子进程执行完毕
    process.join()
    queue.close()
