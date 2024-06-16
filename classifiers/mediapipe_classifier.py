import multiprocessing
import threading
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import websocket
import rel

from config.custom_config import HUMAN_POSE_STANDARD_EMBEDDINGS
from constants.constants import *


PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult

ws = None


def on_message(ws, message):
    print(message)


def on_error(ws, error):
    print("websocket error: {}".format(error))


def on_close(ws, close_status_code, close_msg):
    print("### closed ###")


def on_open(ws):
    print("Opened connection")
    ws.send("Hello, server!")


def init_websocket():
    global ws
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(
        "ws://127.0.0.1:8888/ws/",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    ws.run_forever(
        reconnect=5,
    )  # Set dispatcher to automatic reconnection, 5 second reconnect delay if connection closed unexpectedly


# Create a pose landmarker instance with the live stream mode:
def print_result(
    result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    global ws
    # print("pose landmarker result: {}".format(result))
    if len(result.pose_world_landmarks) == 0:
        return

    flatten_coords0 = []
    # 归一化worldLandmarks
    worldLandmarks = result.pose_world_landmarks[0]
    max_x = 0
    min_x = 0
    max_y = 0
    min_y = 0
    for landmark in worldLandmarks:
        if landmark.x > max_x:
            max_x = landmark.x
        if landmark.x < min_x:
            min_x = landmark.x
        if landmark.y > max_y:
            max_y = landmark.y
        if landmark.y < min_y:
            min_y = landmark.y
    for landmark in worldLandmarks:
        if landmark.x > 0:
            temp_x = landmark.x / max_x
        elif landmark.x < 0:
            temp_x = landmark.x / min_x
        else:
            temp_x = 0
        if landmark.y > 0:
            temp_y = landmark.y / max_y
        elif landmark.y < 0:
            temp_y = landmark.y / min_y
        else:
            temp_y = 0
        # print(f"{temp_x}   {temp_y}")
        flatten_coords0.append(temp_x)
        flatten_coords0.append(temp_y)

    # 计算相似度
    min_L2_distance = 100000
    current_action = ""
    for key, value in HUMAN_POSE_STANDARD_EMBEDDINGS.items():
        L2_distance_temp = 0
        for i in range(len(flatten_coords0)):
            L2_distance_temp += (flatten_coords0[i] - value[i]) ** 2
        if L2_distance_temp < min_L2_distance:
            min_L2_distance = L2_distance_temp
            current_action = key

    print(f"当前动作: {current_action}, 欧几里得距离平方数: {min_L2_distance}")
    if min_L2_distance < L2_DISTANCE_THRESHOLD:
        print("尝试发送动作")
        # 动作正确，发送正确信号
        ws.send(current_action)
    elif min_L2_distance < L2_DISTANCE_CHANGE_DIRECTION and (
        current_action == "turn_on_left" or current_action == "turn_on_right"
    ):
        print("向左或向右转换视角")
        ws.send(current_action)


def classify_pose_v2(queue):
    # global ws
    # ws = websocket.create_connection("ws://127.0.0.1:8888/ws/", timeout=360000)
    print("start websocket")

    # 创建一个线程
    thread = threading.Thread(target=init_websocket)

    # 启动线程
    thread.start()

    # 主线程继续执行
    print("主线程继续执行其他任务")

    model_path = "E:/python_projects/MingchaoPlayer/models/mediapipe_models/pose_landmarker_heavy.task"
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result,
    )
    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            # 子进程从队列中取出数据
            temp_list = queue.get()
            image_np, frame_timestamp_ms = temp_list[0], temp_list[1]

            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

            # Send live image data to perform pose landmarking.
            # The results are accessible via the `result_callback` provided in
            # the `PoseLandmarkerOptions` object.
            # The pose landmarker must be created with the live stream mode.
            print(f"timestamp ms: {frame_timestamp_ms}")
            landmarker.detect_async(mp_image, frame_timestamp_ms)

    # 等待子线程完成
    thread.join()
    print("子线程已结束")
