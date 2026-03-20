import mediapipe as mp
import cv2
import os

OBJECT_MODEL_PATH = 'models/efficientdet_lite0.tflite'
GESTURE_MODEL_PATH = 'models/gesture_recognizer.task'

def detect_objects():

    if not os.path.exists(OBJECT_MODEL_PATH):
        print(f"Model file not found at {OBJECT_MODEL_PATH}. Please ensure the model is downloaded and placed in the correct directory.")
        return
    
    BaseOptions = mp.tasks.BaseOptions
    ObjectDetector = mp.tasks.vision.ObjectDetector
    ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=OBJECT_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        score_threshold=0.5
    )

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with ObjectDetector.create_from_options(options) as detector:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

            detection_result = detector.detect_for_video(mp_image, timestamp_ms)

            for detection in detection_result.detections:
                
                bbox = detection.bounding_box
                start_point = int(bbox.origin_x), int(bbox.origin_y)
                end_point = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)

                cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

                category = detection.categories[0]
                category_name = category.category_name
                score = round(category.score, 2)

                label = f"{category_name}: {score}"

                cv2.putText(frame, label, (start_point[0], max(0, start_point[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('MediaPipe Object Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def detect_gestures():

    if not os.path.exists(GESTURE_MODEL_PATH):
        print(f"Model file not found at {GESTURE_MODEL_PATH}. Please ensure the model is downloaded and placed in the correct directory.")
        return
    
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode


    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=GESTURE_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO
    )

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with GestureRecognizer.create_from_options(options) as recognizer:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

            recognizer_result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            for index, gestures in enumerate(recognizer_result.gestures):
                if not gestures:
                    continue

                top_gesture = gestures[0]
                gesture_name = top_gesture.category_name
                score = round(top_gesture.score, 2)

                label = f"Hand {index + 1}: {gesture_name} ({score})"
                y = 30 + index * 30
                cv2.putText(frame, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow('MediaPipe Gesture Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_objects()


            



