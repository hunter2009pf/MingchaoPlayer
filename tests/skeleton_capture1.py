import cv2
import mediapipe as mp


if __name__=="__main__":
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    custom_style = mp_drawing_styles.get_default_pose_landmarks_style()
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    image = cv2.imread('./output.png')
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=custom_style,
        )
    cv2.imwrite('skeleton_image.jpg', image)
    # 或者
    cv2.imshow('Skeleton Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()