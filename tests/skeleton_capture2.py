## Holistic Solution using Video in Python

import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


if __name__=="__main__":
    file = 'D:/digital_human/yolo_test_20240528/eval.mp4'
    video = cv2.VideoCapture(file)
    frame_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Frame size ", (frame_width, frame_height))

    output = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), video.get(cv2.CAP_PROP_FPS), (int(frame_width), int(frame_height)))

    print("Video FPS: ", video.get(cv2.CAP_PROP_FPS))
    print("Video Frame count:", video.get(cv2.CAP_PROP_FRAME_COUNT))
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=True) as holistic:

        while video.isOpened(): 
            # Capture frame-by-frame
            success, frame = video.read()
        
            if not success:
                print("Ignoring empty frame")
                # if loading a video, use break or else use continue for live stream
                break
            

            # Convert the BGR image to RGB before processing.
            
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Create a white frame of same size as video
            # dtype np.uint8 is important since that is the default value of cv2 image R/W
            # check this https://scikit-image.org/docs/stable/user_guide/data_types.html#image-data-types-and-what-they-mean
            annotated_frame = np.full((int(frame_height), int(frame_width), 3), 255, dtype=np.uint8)

            # uncomment this if you need to draw pose lines over the actual video itself
            # annotated_frame = frame.copy()

            # Draw pose, left and right hands, and face landmarks on the image.
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.
                    get_default_pose_landmarks_style())

            # Uncomment following if you want to write to a new video file
            output.write(annotated_frame)
        
            # Plot pose world landmarks.
            # mp_drawing.plot_landmarks(
            #     results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)
        

        # release the video capture object
        video.release()

        # release the video output object
        output.release()