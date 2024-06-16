import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image


def test_similarity():
    # image_path = "E:/mingchao_data/images_for_eval/move_left.png"
    image_near_path = "E:/mingchao_data/human_pose_test/near1.jpg"
    # image_faraway_path = "E:/mingchao_data/human_pose_test/faraway2.PNG"
    # image_faraway_path = "E:/mingchao_data/human_pose_test/near2.jpg"
    # image_faraway_path = "E:/mingchao_data/human_pose_test/near3.jpg"
    # image_faraway_path = "E:/mingchao_data/human_pose_test/near4.jpg"
    image_faraway_path = "E:/mingchao_data/images_for_eval/move_left.png"

    model_path = "E:/python_projects/MingchaoPlayer/models/mediapipe_models/pose_landmarker_heavy.task"

    flatten_coords0 = []
    flatten_coords1 = []

    # STEP 2: Create an PoseLandmarker object.
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options, output_segmentation_masks=True
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    image = mp.Image.create_from_file(image_near_path)

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(image)

    # print(f"pose result: {detection_result}")

    # 归一化worldLandmarks
    worldLandmarks = detection_result.pose_world_landmarks[0]
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

    # STEP 3: Load the input image.
    image1 = mp.Image.create_from_file(image_faraway_path)

    # STEP 4: Detect pose landmarks from the input image.
    detection_result1 = detector.detect(image1)

    # print(f"pose result: {detection_result}")

    # 归一化worldLandmarks
    worldLandmarks1 = detection_result1.pose_world_landmarks[0]
    max_x1 = 0
    min_x1 = 0
    max_y1 = 0
    min_y1 = 0
    for landmark in worldLandmarks1:
        if landmark.x > max_x1:
            max_x1 = landmark.x
        if landmark.x < min_x1:
            min_x1 = landmark.x
        if landmark.y > max_y1:
            max_y1 = landmark.y
        if landmark.y < min_y1:
            min_y1 = landmark.y
    for landmark in worldLandmarks1:
        if landmark.x > 0:
            temp_x = landmark.x / max_x1
        elif landmark.x < 0:
            temp_x = landmark.x / min_x1
        else:
            temp_x = 0
        if landmark.y > 0:
            temp_y = landmark.y / max_y1
        elif landmark.y < 0:
            temp_y = landmark.y / min_y1
        else:
            temp_y = 0
        # print(f"{temp_x}   {temp_y}")
        flatten_coords1.append(temp_x)
        flatten_coords1.append(temp_y)

    # 计算相似度
    similarity = 0
    for i in range(0, len(flatten_coords0)):
        similarity += (flatten_coords0[i] - flatten_coords1[i]) ** 2
    similarity = similarity**0.5
    print(f"similarity: {similarity}")
    # STEP 5: Process the detection result. In this case, visualize it.
    # annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    # cv2.imshow(winname="image", mat=cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)


if __name__ == "__main__":
    model_path = "E:/python_projects/MingchaoPlayer/models/mediapipe_models/pose_landmarker_heavy.task"
    # 获取当前工作目录
    # dir_path = "E:/mingchao_data/images_for_eval"
    dir_path = "E:/mingchao_data/forward_left_right"

    # 列出dir_path下的所有文件和文件夹
    for filename in os.listdir(dir_path):
        # 获取文件的完整路径
        file_path = os.path.join(dir_path, filename)
        # 检查这个路径是否是文件
        if os.path.isfile(file_path):
            print(filename)

            flatten_coords0 = []

            # STEP 2: Create an PoseLandmarker object.
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options, output_segmentation_masks=True
            )
            detector = vision.PoseLandmarker.create_from_options(options)

            # STEP 3: Load the input image.
            image = mp.Image.create_from_file(file_path)

            # STEP 4: Detect pose landmarks from the input image.
            detection_result = detector.detect(image)

            # print(f"pose result: {detection_result}")

            # 归一化worldLandmarks
            worldLandmarks = detection_result.pose_world_landmarks[0]
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
            print(flatten_coords0)
