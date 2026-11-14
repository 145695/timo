import sys
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO
import xgboost as xgb
import math
import time
import random
import signal
from pathlib import Path
from collections import deque

from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                               QMenu, QPushButton, QMainWindow)
from PySide6.QtCore import QThread, Signal, Qt, QTimer
from PySide6.QtGui import QFont, QMovie, QAction, QGuiApplication

# --- Paths ---
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
LANDMARKER_PATH = MODEL_DIR / "face_landmarker.task"
IMAGE_MODEL_PATH = MODEL_DIR / "image_model.pt"
TABULAR_MODEL_PATH = MODEL_DIR / "tabular_model.json"

# --- Dialogue Lines ---
DIALOGUES = {
    "focused": ["Keep it up!", "You're in the zone!", "Nice focus.", "Great job!", "Let's go!"],
    "distracted_1": ["Eyes on the screen!", "Lost in thought?", "Let's get back to it."],
    "distracted_2": ["That's the second time...", "Focus! You were doing so well.", "I see you looking away."],
    "away": ["Where did you go?", "I'll be here when you get back.", "Taking a break?"]
}

# --- Background Vision Thread ---
class VisionWorker(QThread):
    state_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        
        base_options = mp_python.BaseOptions(model_asset_path=str(LANDMARKER_PATH))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options, output_facial_transformation_matrixes=True,
            num_faces=1, running_mode=mp_vision.RunningMode.VIDEO
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self.image_model = YOLO(IMAGE_MODEL_PATH)
        self.tabular_model = xgb.XGBClassifier()
        self.tabular_model.load_model(TABULAR_MODEL_PATH)
        
        self.cap = cv2.VideoCapture(0)
        self.t0 = time.time()
        self.last_face_time = time.time()
        
        self.calibration_frames = []
        self.baseline_pitch = 0.0
        self.baseline_yaw = 0.0
        self.is_calibrated = False
        
        self.ALPHA = 0.15
        self.p_smoothed = 0.5
        self.score_history = deque()
        self.current_state = "FOCUSED"
        self.pending_state = None
        self.state_change_timer = time.time()

    def get_euler_angles(self, R):
        sy = math.hypot(R[0, 0], R[1, 0])
        if sy > 1e-6:
            pitch = math.atan2(R[2, 1], R[2, 2])
            yaw   = math.atan2(-R[2, 0], sy)
        else:
            pitch = math.atan2(-R[1, 2], R[1, 1])
            yaw   = math.atan2(-R[2, 0], sy)
        return math.degrees(pitch), math.degrees(yaw)

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            now = time.time()

            results_img = self.image_model(frame, verbose=False)
            classes = results_img[0].names
            probs = results_img[0].probs.data.cpu().numpy()
            focused_idx = [k for k, v in classes.items() if v == 'focused'][0]
            p_image = probs[focused_idx]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            t_ms = int((now - self.t0) * 1000)
            result = self.landmarker.detect_for_video(mp_img, t_ms)

            p_tabular = 0.5
            target_state = self.current_state

            if result.face_landmarks:
                self.last_face_time = now
                M = np.asarray(result.facial_transformation_matrixes[0])
                pitch, yaw = self.get_euler_angles(M[:3, :3])

                if not self.is_calibrated:
                    self.calibration_frames.append((pitch, yaw))
                    if len(self.calibration_frames) >= 50:
                        self.baseline_pitch = sum(f[0] for f in self.calibration_frames) / len(self.calibration_frames)
                        self.baseline_yaw = sum(f[1] for f in self.calibration_frames) / len(self.calibration_frames)
                        self.is_calibrated = True
                    self.msleep(30)
                    continue

                pitch_cal = pitch - self.baseline_pitch
                yaw_cal = yaw - self.baseline_yaw
                p_tabular = self.tabular_model.predict_proba(np.array([[pitch_cal, yaw_cal]]))[0][1]

                p_fused = 0.2 * p_image + 0.8 * p_tabular
                self.p_smoothed = (self.ALPHA * p_fused) + ((1 - self.ALPHA) * self.p_smoothed)

                self.score_history.append((now, self.p_smoothed))
                while self.score_history and now - self.score_history[0][0] > 10.0:
                    self.score_history.popleft()

                window_avg = sum(s for t, s in self.score_history) / len(self.score_history)

                if window_avg >= 0.6: target_state = "FOCUSED"
                elif window_avg <= 0.4: target_state = "DISTRACTED"
                else: target_state = "FOCUSED"

            else:
                time_away = now - self.last_face_time
                target_state = "AWAY" if time_away > 5.0 else "NO FACE"
                self.score_history.clear()
                self.p_smoothed = 0.5

            if target_state != self.current_state:
                if target_state != self.pending_state:
                    self.pending_state = target_state
                    self.state_change_timer = now
                elif now - self.state_change_timer >= 3.0:
                    self.current_state = target_state
                    self.pending_state = None
                    self.state_changed.emit(self.current_state)
            else:
                self.pending_state = None

            self.msleep(50)

    def stop(self):
        self.running = False
        self.cap.release()
        self.wait()

# --- Floating Character Overlay ---
class CompanionOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.resize(255, 180) 
        self.move(10, 10) 

        # 1. Speech Bubble 
        self.speech = QLabel("Ready...", self)
        self.speech.setGeometry(85, 10, 160, 130) 
        self.speech.setAlignment(Qt.AlignCenter)
        self.speech.setWordWrap(True)
        self.speech.setStyleSheet("""
            border: 2px solid #6b8eb5; 
            background-color: white; 
            color: #6b8eb5; 
            font-family: Arial; 
            font-weight: bold; 
            font-size: 16px;
            padding-left: 45px; 
            padding-right: 10px;
        """)

        # 2. Character (Overlapping the bubble)
        self.sprite_label = QLabel(self)
        self.sprite_label.setGeometry(0, 20, 110, 110) 
        self.sprite_label.setAlignment(Qt.AlignCenter)

        # 3. Timer (Full Width)
        self.timer_label = QLabel("25:00", self)
        self.timer_label.setGeometry(10, 145, 235, 25) 
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("""
            border: 2px solid #6b8eb5; 
            background-color: white; 
            color: #6b8eb5; 
            font-family: Arial; 
            font-weight: bold; 
            font-size: 14px;
        """)
        
        self.sprite_label.raise_()
        self.timer_label.raise_()
        
        self.old_pos = None
        self.last_reaction_time = time.time()
        self.COOLDOWN = 15.0 
        self.distraction_count = 0
        
        self.set_animation("away")

    def set_animation(self, state_name):
        path = ASSETS_DIR / f"{state_name}.gif"
        if path.exists():
            self.movie = QMovie(str(path))
            self.movie.setScaledSize(self.sprite_label.size())
            self.sprite_label.setMovie(self.movie)
            self.movie.start()
        else:
            self.sprite_label.setText(f"🎬\n{state_name}.gif")
            self.sprite_label.setStyleSheet("border: 1px dashed gray; color: gray; font-size: 10px;")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self.old_pos:
            delta = event.globalPosition().toPoint() - self.old_pos
            self.move(self.pos() + delta)
            self.old_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        self.old_pos = None

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setStyleSheet("QMenu { background-color: white; color: black; padding: 5px; }")
        show_dash = QAction("⚙️ Show Dashboard / Pause", self)
        show_dash.triggered.connect(self.show_dashboard)
        quit_action = QAction("❌ Quit", self)
        quit_action.triggered.connect(QApplication.instance().quit)
        menu.addAction(show_dash)
        menu.addAction(quit_action)
        menu.exec(event.globalPos())

    def show_dashboard(self):
        self.hide()
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, MainWindow):
                widget.show()
                widget.pomodoro_timer.stop()
                widget.status_label.setText("Status: Paused")
                break

# --- Main Dashboard Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FocusCompanion Dashboard")
        self.resize(400, 300)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        title = QLabel("FocusCompanion")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(QFont("Arial", 14))
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        layout.addSpacing(20)
        
        self.btn_start = QPushButton("▶ Start 25m Focus Session")
        self.btn_start.setFont(QFont("Arial", 12, QFont.Bold))
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;")
        self.btn_start.clicked.connect(self.start_focus)
        layout.addWidget(self.btn_start)
        
        self.btn_break = QPushButton("☕ Start 5m Break")
        self.btn_break.setFont(QFont("Arial", 12))
        self.btn_break.clicked.connect(self.start_break)
        layout.addWidget(self.btn_break)

        self.overlay = CompanionOverlay()
        
        self.vision = VisionWorker()
        self.vision.state_changed.connect(self.on_state_changed)
        self.vision.start()
        
        self.timer_state = "IDLE"
        self.time_left = 25 * 60
        self.pomodoro_timer = QTimer()
        self.pomodoro_timer.timeout.connect(self.update_timer)

    def start_focus(self):
        self.timer_state = "FOCUS"
        self.time_left = 25 * 60
        self.pomodoro_timer.start(1000)
        
        self.hide()
        self.overlay.show()
        self.overlay.set_animation("focused")
        self.overlay.timer_label.setText("25:00")
        self.overlay.speech.setText("Let's get to work!")
        
    def start_break(self):
        self.timer_state = "BREAK"
        self.time_left = 5 * 60
        self.pomodoro_timer.start(1000)
        
        self.hide()
        self.overlay.show()
        self.overlay.set_animation("break")
        self.overlay.timer_label.setText("05:00")
        self.overlay.speech.setText("Time for a break!")

    def update_timer(self):
        self.time_left -= 1
        mins, secs = divmod(self.time_left, 60)
        self.overlay.timer_label.setText(f"{mins:02d}:{secs:02d}")
        
        if self.time_left <= 0:
            self.pomodoro_timer.stop()
            if self.timer_state == "FOCUS":
                self.timer_state = "COMPLETE"
                self.overlay.set_animation("complete")
                self.overlay.speech.setText("Session complete! Great work!")
                QTimer.singleShot(5000, lambda: self.start_break())
            elif self.timer_state == "BREAK":
                self.timer_state = "IDLE"
                self.overlay.hide()
                self.show()
                self.status_label.setText("Status: Ready (Session Complete)")

    def on_state_changed(self, state):
        now = time.time()
        time_since_reaction = now - self.overlay.last_reaction_time

        if self.timer_state == "BREAK":
            return

        if state == "FOCUSED":
            if self.timer_state != "COMPLETE":
                self.overlay.set_animation("focused")
            # Change to a new encouraging message instead of clearing it
            if time_since_reaction > 3.0:
                self.speak(random.choice(DIALOGUES["focused"]))
            self.overlay.distraction_count = 0
            
        elif state == "AWAY":
            self.overlay.set_animation("away")
            if time_since_reaction > self.overlay.COOLDOWN:
                self.speak(random.choice(DIALOGUES["away"]))
                
        elif state == "DISTRACTED":
            self.overlay.set_animation("distracted")
            self.overlay.distraction_count += 1
            if time_since_reaction > self.overlay.COOLDOWN:
                if self.overlay.distraction_count == 1:
                    self.speak(random.choice(DIALOGUES["distracted_1"]))
                else:
                    self.speak(random.choice(DIALOGUES["distracted_2"]))

    def speak(self, text):
        self.overlay.speech.setText(text)
        self.overlay.last_reaction_time = time.time()
        # Notice: Removed the QTimer.singleShot that cleared the text after 5 seconds!
        # Now the text stays on screen until the state changes.

    def closeEvent(self, event):
        self.vision.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    def sigint_handler(*args):
        print("\n\n⛔ Caught Ctrl+C. Shutting down gracefully...")
        QApplication.quit()
        
    signal.signal(signal.SIGINT, sigint_handler)
    
    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None) 

    window = MainWindow()
    app.aboutToQuit.connect(window.close)
    
    window.show()
    sys.exit(app.exec())