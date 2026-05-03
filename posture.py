import cv2
import mediapipe as mp
import time
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

latest_result = None
baseline_angle = None  # This will store your 'good' posture
is_slouching = False

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

def get_angle(p1, p2):
    # Using the vertical angle (angle from the vertical Y-axis)
    # This is often more stable for neck tilt
    return math.degrees(math.atan2(p2.x - p1.x, p2.y - p1.y))

base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=result_callback
)

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    
    print("Press 's' to Calibrate while sitting straight!")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        if latest_result and latest_result.pose_landmarks:
            landmarks = latest_result.pose_landmarks[0]
            
            # Use Ear (7) and Shoulder (11)
            ear = landmarks[7]
            shoulder = landmarks[11]
            current_angle = get_angle(shoulder, ear)

            # Display logic
            if baseline_angle is None:
                text = "Press 's' to Calibrate"
                color = (255, 255, 255)
            else:
                # If current angle deviates more than 10 degrees from baseline
                # You can tweak '10' to be more or less sensitive
                if abs(current_angle - baseline_angle) > 10:
                    text = "ALARM: SLOUCHING!"
                    color = (0, 0, 255)
                else:
                    text = "Posture: Good"
                    color = (0, 255, 0)
            
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Angle: {int(current_angle)}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Refined Posture Checker', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'): # Press 's' to set baseline
            baseline_angle = current_angle
            print(f"Calibrated! Baseline set to: {baseline_angle}")
        elif key == 27: # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()