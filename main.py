import multiprocessing

from camera.ip_webcam import connect_camera
from classifiers.mediapipe_classifier import classify_pose_v2

# def classify_humnan_pose_by_yolov8():
#     queue = multiprocessing.Queue()
#     # 创建子进程
#     process = multiprocessing.Process(target=classify_pose, args=(queue,))
#     process.start()
#     connect_camera(queue)
#     # 等待子进程执行完毕
#     process.join()
#     queue.close()


def classify_humnan_pose_by_mediapipe():
    print("arrive here")
    queue = multiprocessing.Queue()
    # 创建子进程
    print("start new process")
    process = multiprocessing.Process(target=classify_pose_v2, args=(queue,))
    process.start()
    connect_camera(queue)
    # 等待子进程执行完毕
    process.join()
    queue.close()


if __name__ == "__main__":
    # init_websocket_v2()
    classify_humnan_pose_by_mediapipe()
