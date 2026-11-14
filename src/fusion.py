import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO
import xgboost as xgb
import math
from pathlib import Path
import time
from collections import deque

# --- Paths ---
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
LANDMARKER_PATH = MODEL_DIR / "face_landmarker.task"
IMAGE_MODEL_PATH = MODEL_DIR / "image_model.pt"
TABULAR_MODEL_PATH = MODEL_DIR / "tabular_model.json"

# --- Load Models ---
print("Loading models...")
base_options = mp_python.BaseOptions(model_asset_path=str(LANDMARKER_PATH))
options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    running_mode=mp_vision.RunningMode.VIDEO,
)
landmarker = mp_vision.FaceLandmarker.create_from_options(options)
image_model = YOLO(IMAGE_MODEL_PATH)
tabular_model = xgb.XGBClassifier()
tabular_model.load_model(TABULAR_MODEL_PATH)

# --- Variables ---
ALPHA = 0.15  
p_focused_smoothed = 0.5  
last_face_time = time.time()
t0 = time.time()  

# --- Calibration ---
calibration_frames = []
baseline_pitch = 0.0
baseline_yaw = 0.0
is_calibrated = False

# --- Temporal Logic Variables ---
score_history = deque()
WINDOW_SECONDS = 10.0 
PERSISTENCE_SECONDS = 3.0 

# Start in FOCUSED instead of NO FACE so it doesn't deadlock
current_state = "FOCUSED" 
pending_state = None
state_change_timer = time.time()

def get_euler_angles(R):
    sy = math.hypot(R[0, 0], R[1, 0])
    if sy > 1e-6:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw   = math.atan2(-R[2, 0], sy)
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw   = math.atan2(-R[2, 0], sy)
    return math.degrees(pitch), math.degrees(yaw)

def main():
    global p_focused_smoothed, last_face_time, calibration_frames, baseline_pitch, baseline_yaw, is_calibrated
    global current_state, pending_state, state_change_timer
    
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        
        # --- 1. Image Model ---
        results_img = image_model(frame, verbose=False)
        classes = results_img[0].names
        probs = results_img[0].probs.data.cpu().numpy()
        focused_idx = [k for k, v in classes.items() if v == 'focused'][0]
        p_image = probs[focused_idx]
        
        # --- 2. MediaPipe & Tabular Model ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        t_ms = int((now - t0) * 1000)
        result = landmarker.detect_for_video(mp_img, t_ms)
        
        p_tabular = 0.5
        target_state = current_state # Default to current state
        
        if result.face_landmarks:
            last_face_time = now
            
            M = np.asarray(result.facial_transformation_matrixes[0])
            pitch, yaw = get_euler_angles(M[:3, :3])
            
            if not is_calibrated:
                calibration_frames.append((pitch, yaw))
                cv2.putText(frame, "CALIBRATING... LOOK STRAIGHT AT SCREEN", 
                            (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Frames: {len(calibration_frames)}/50", 
                            (20, h // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if len(calibration_frames) >= 50:
                    baseline_pitch = sum(f[0] for f in calibration_frames) / len(calibration_frames)
                    baseline_yaw = sum(f[1] for f in calibration_frames) / len(calibration_frames)
                    is_calibrated = True
                    print(f"✅ Calibrated! Baseline Pitch: {baseline_pitch:.1f}, Yaw: {baseline_yaw:.1f}")
                cv2.imshow("FocusCompanion", frame)
                cv2.waitKey(1)
                continue
            
            pitch_cal = pitch - baseline_pitch
            yaw_cal = yaw - baseline_yaw
            
            p_tabular = tabular_model.predict_proba(np.array([[pitch_cal, yaw_cal]]))[0][1]
            
            # Fusion (20% Image, 80% Tabular)
            p_fused = 0.2 * p_image + 0.8 * p_tabular
            p_focused_smoothed = (ALPHA * p_fused) + ((1 - ALPHA) * p_focused_smoothed)
            
            # Rolling Window
            score_history.append((now, p_focused_smoothed))
            while score_history and now - score_history[0][0] > WINDOW_SECONDS:
                score_history.popleft()
            
            window_avg = sum(s for t, s in score_history) / len(score_history)
            
            # Determine Target State
            if window_avg >= 0.6:
                target_state = "FOCUSED"
            elif window_avg <= 0.4:
                target_state = "DISTRACTED"
            else:
                target_state = "FOCUSED" # Neutral
                
            # Display
            cv2.putText(frame, f"Img: {p_image:.2f} | Tab: {p_tabular:.2f} | Smoothed: {p_focused_smoothed:.2f}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"10s Window Avg: {window_avg:.2f}", 
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Pitch (cal): {pitch_cal:+.1f} | Yaw (cal): {yaw_cal:+.1f}", 
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
        else:
            # Away Logic
            time_away = now - last_face_time
            if time_away > 5.0:
                target_state = "AWAY"
            else:
                target_state = "NO FACE"
            
            score_history.clear()
            p_focused_smoothed = 0.5
            
        # --- FIXED STATE MACHINE ---
        if target_state != current_state:
            if target_state != pending_state:
                # New target detected, start the timer
                pending_state = target_state
                state_change_timer = now
            elif now - state_change_timer >= PERSISTENCE_SECONDS:
                # Timer reached! Commit the change.
                current_state = target_state
                pending_state = None
        else:
            # Target matches current state, cancel any pending change
            pending_state = None
            
        # Colors
        if current_state == "FOCUSED":
            color = (0, 255, 0)
        elif current_state == "DISTRACTED":
            color = (0, 0, 255)
        elif current_state == "AWAY":
            color = (0, 165, 255)
        else:
            color = (128, 128, 128)
            
        # Show pending state text
        if pending_state:
            time_left = PERSISTENCE_SECONDS - (now - state_change_timer)
            if time_left > 0:
                cv2.putText(frame, f"Changing to {pending_state} in {time_left:.1f}s...", 
                            (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2)

        # Draw big state text
        text_size = cv2.getTextSize(current_state, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, current_state, (text_x, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        cv2.imshow("FocusCompanion", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()