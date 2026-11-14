import sys
import numpy as np
import threading
import math
import time
import random
import signal
from pathlib import Path
from collections import deque

from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                               QMenu, QPushButton, QMainWindow, QFrame, QLineEdit,
                               QScrollArea, QButtonGroup, QSizePolicy, QStackedWidget)
from PySide6.QtCore import QThread, Signal, Qt, QTimer, QSize, QRectF, QPointF, QByteArray, QObject
from PySide6.QtGui import (QFont, QMovie, QAction, QGuiApplication, QPixmap, QPainter, QColor,
                           QPen, QBrush, QRadialGradient, QFontDatabase, QIcon, QFontMetrics)
from PySide6.QtSvg import QSvgRenderer
import json

# --- Paths ---
# PyInstaller uses sys._MEIPASS to unpack bundled data (assets, models)
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
TIMO_DIR = ASSETS_DIR / "timo"        
FONTS_DIR = ASSETS_DIR / "fonts"      
LANDMARKER_PATH = MODEL_DIR / "face_landmarker.task"
IMAGE_MODEL_PATH = MODEL_DIR / "image_model.pt"
TABULAR_MODEL_PATH = MODEL_DIR / "tabular_model.json"

def complete_gif_path():
    for name in ("complete.gif", "compleat.gif"):      
        p = ASSETS_DIR / name
        if p.exists():
            return p
    return None

_complete_ms = None

def complete_gif_ms():
    global _complete_ms
    if _complete_ms is None:
        total = 0
        path = complete_gif_path()
        if path is not None:
            m = QMovie(str(path))
            for i in range(max(0, m.frameCount())):
                m.jumpToFrame(i)
                total += max(10, m.nextFrameDelay())
        _complete_ms = int(min(3500, max(1000, total))) if total else 1500
    return _complete_ms

DIALOGUES = {
    "focused": ["Keep it up!", "You're in the zone!", "Nice focus.", "Great job!", "Let's go!"],
    "distracted_1": ["Eyes on the screen!", "Lost in thought?", "Let's get back to it."],
    "distracted_2": ["That's the second time...", "Focus! You were doing so well.", "I see you looking away."],
    "away": ["Where did you go?", "I'll be here when you get back.", "Taking a break?"]
}

cv2 = mp = mp_python = mp_vision = YOLO = xgb = None

class ML:
    ready = False
    error = None
    image_model = None
    tabular_model = None
    landmarker = None
    t0 = time.time()

def _import_ml_libraries():
    global cv2, mp, mp_python, mp_vision, YOLO, xgb
    import cv2 as _cv2
    import mediapipe as _mp
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision
    from ultralytics import YOLO as _YOLO
    import xgboost as _xgb
    cv2, mp, mp_python, mp_vision, YOLO, xgb = _cv2, _mp, _mp_python, _mp_vision, _YOLO, _xgb

class ModelLoader(QObject):
    loaded = Signal(bool)

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        t = time.time()
        try:
            _import_ml_libraries()
            ML.image_model = YOLO(IMAGE_MODEL_PATH)
            model = xgb.XGBClassifier()
            model.load_model(TABULAR_MODEL_PATH)
            ML.tabular_model = model
            base_options = mp_python.BaseOptions(model_asset_path=str(LANDMARKER_PATH))
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options, output_facial_transformation_matrixes=True,
                num_faces=1, running_mode=mp_vision.RunningMode.VIDEO
            )
            ML.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            ML.image_model(np.zeros((480, 640, 3), np.uint8), verbose=False)
            ML.tabular_model.predict_proba(np.array([[0.0, 0.0]]))
            ML.t0 = time.time()
            ML.ready = True
            print(f"Timo: focus models ready in {time.time() - t:.1f}s")
        except Exception as exc:
            ML.error = str(exc)
            print(f"Timo: could not load the focus models: {exc}")
        self.loaded.emit(ML.ready)

def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    return cap

class VisionWorker(QThread):
    state_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.reset_state()

    def reset_state(self, keep_calibration=False):
        now = time.time()
        keep = (keep_calibration and getattr(self, "is_calibrated", False)
                and now - getattr(self, "calibrated_at", 0.0) < 1800)
        self.last_face_time = now
        if not keep:
            self.calibration_frames = []
            self.baseline_pitch = 0.0
            self.baseline_yaw = 0.0
            self.is_calibrated = False
        self.ALPHA = 0.15
        self.p_smoothed = 0.5
        self.score_history = deque()
        self.current_state = "FOCUSED"
        self.pending_state = None
        self.state_change_timer = now

    def get_euler_angles(self, R):
        sy = math.hypot(R[0, 0], R[1, 0])
        if sy > 1e-6:
            pitch = math.atan2(R[2, 1], R[2, 2])
            yaw   = math.atan2(-R[2, 0], sy)
        else:
            pitch = math.atan2(-R[1, 2], R[1, 1])
            yaw   = math.atan2(-R[2, 0], sy)
        return math.degrees(pitch), math.degrees(yaw)

    def start_camera(self):
        if self.isRunning():
            return
        self.running = True
        self.start()

    def stop_camera(self):
        self.running = False
        if self.isRunning():
            self.wait(4000)

    stop = stop_camera

    def run(self):
        while self.running and not ML.ready and ML.error is None:
            self.msleep(100)
        if not self.running:
            return
        if not ML.ready:
            print("Timo: focus models are not available, so the camera stays off.")
            return

        cap = open_camera()
        if not cap.isOpened():
            print("Timo: could not open the camera.")
            cap.release()
            return

        self.reset_state(keep_calibration=True)
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.msleep(30)
                    continue
                frame = cv2.flip(frame, 1)
                now = time.time()

                results_img = ML.image_model(frame, verbose=False)
                classes = results_img[0].names
                probs = results_img[0].probs.data.cpu().numpy()
                focused_idx = [k for k, v in classes.items() if v == 'focused'][0]
                p_image = probs[focused_idx]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                t_ms = int((now - ML.t0) * 1000)
                result = ML.landmarker.detect_for_video(mp_img, t_ms)

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
                            self.calibrated_at = now
                        self.msleep(30)
                        continue

                    pitch_cal = pitch - self.baseline_pitch
                    yaw_cal = yaw - self.baseline_yaw
                    p_tabular = ML.tabular_model.predict_proba(np.array([[pitch_cal, yaw_cal]]))[0][1]

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
        finally:
            cap.release()

C = {
    "bg": "#eaf3fb", "card": "#ffffff", "panel": "#f3f8fd", "ink": "#17324b", "muted": "#5d7891",
    "line": "#d3e3f0", "accent": "#2f67a0", "soft": "#e2eef9", "btn_top": "#3a76b0", "btn_bottom": "#2a5b8d",
}
DISPLAY_FAMILIES = ["Fredoka", "Fredoka SemiBold", "Arial Rounded MT Bold", "Segoe UI", "Helvetica Neue", "Arial"]
BODY_FAMILIES = ["Nunito", "Segoe UI", "Helvetica Neue", "Arial"]
FOCUS_OPTIONS = [15, 25, 45, 60]
BREAK_OPTIONS = [5, 10, 15]
SAMPLE_TASKS = ["Outline the introduction", "Reply to two emails", "Tidy the reading list"]
STATE_FILE = Path.home() / ".timo_state.json"

def load_app_fonts():
    if FONTS_DIR.exists():
        for f in list(FONTS_DIR.glob("*.ttf")) + list(FONTS_DIR.glob("*.otf")):
            QFontDatabase.addApplicationFont(str(f))

def make_font(families, px, weight=QFont.Weight.Normal, spacing=0.0):
    f = QFont()
    f.setFamilies(families)
    f.setPixelSize(px)
    f.setWeight(weight)
    if spacing:
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    return f

def display_font(px):
    return make_font(DISPLAY_FAMILIES, px, QFont.Weight.DemiBold)

def body_font(px, weight=QFont.Weight.DemiBold, spacing=0.0):
    return make_font(BODY_FAMILIES, px, weight, spacing)

def eyebrow(text):
    lab = QLabel(text.upper())
    lab.setFont(body_font(11, QFont.Weight.ExtraBold, 1.8))
    lab.setStyleSheet(f"color: {C['muted']};")
    return lab

SVG_LOGO = '<svg viewBox="0 0 32 32" fill="none"><circle cx="16" cy="16" r="13" stroke="currentColor" stroke-width="3"/><path d="M16 8v8" stroke="currentColor" stroke-width="3" stroke-linecap="round"/><path d="M16 16l6 3" stroke="currentColor" stroke-width="3" stroke-linecap="round"/></svg>'
SVG_RESET = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12a8 8 0 1 0 3-6.2"/><path d="M4 4v5h5"/></svg>'
SVG_SOUND_ON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9v6h4l5 4V5L8 9H4z"/><path d="M16.5 8.5a5 5 0 0 1 0 7"/></svg>'
SVG_SOUND_OFF = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9v6h4l5 4V5L8 9H4z"/><path d="M17 9l4 6M21 9l-4 6"/></svg>'
SVG_PLUS = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>'
SVG_CLOSE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>'
SVG_CLOCK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>'

def svg_pixmap(svg, size, color):
    if "xmlns" not in svg:
        svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    svg = svg.replace("currentColor", color)
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pm = QPixmap(size * 2, size * 2)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    renderer.render(painter, QRectF(0, 0, size * 2, size * 2))
    painter.end()
    pm.setDevicePixelRatio(2)
    return pm

def svg_icon(svg, size, color):
    return QIcon(svg_pixmap(svg, size, color))

class Stage(QWidget):
    RATIOS = {"idle": 1.0, "pause": 1.0, "focus": 0.86, "break": 0.78, "cheer": 0.86}
    STEP_SECONDS = {"idle": 2.6, "pause": 1.8, "focus": 0.7, "break": 0.65, "cheer": 1.0}
    COUNTS = (("idle", 2), ("pause", 2), ("focus", 5), ("break", 9), ("cheer", 1))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(180, 120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.movie = None         
        self.boost = 1.0          
        self.hop = 22             
        self.frames = {}
        self.kind = "idle"
        self.idx = 0
        self.direction = 1
        self.phase = 0.0
        self.last_step = time.monotonic()
        for kind, count in self.COUNTS:
            loaded = []
            for i in range(count):
                path = TIMO_DIR / f"{kind}_{i}.png"
                if path.exists():
                    loaded.append(QPixmap(str(path)))
            self.frames[kind] = loaded
        self.ticker = QTimer(self)
        self.ticker.timeout.connect(self._tick)
        self.ticker.start(33)

    def set_kind(self, kind):
        if kind == self.kind:
            return
        self.kind = kind
        self.idx = 0
        self.direction = 1
        self.last_step = time.monotonic()
        self._sync_movie()
        self.update()

    def _sync_movie(self):
        if self.kind == "cheer":
            path = complete_gif_path()
            if path is not None:
                if self.movie is None:
                    self.movie = QMovie(str(path))
                    self.movie.frameChanged.connect(lambda _=0: self.update())
                self.movie.jumpToFrame(0)
                self.movie.start()
        elif self.movie is not None:
            self.movie.stop()

    def _tick(self):
        if not self.isVisible():
            return
        self.phase += 0.033
        frames = self.frames.get(self.kind) or []
        now = time.monotonic()
        if len(frames) > 1 and now - self.last_step >= self.STEP_SECONDS[self.kind]:
            self.last_step = now
            if self.kind in ("idle", "pause"):
                self.idx = (self.idx + 1) % len(frames)
            else:
                self.idx += self.direction
                if self.idx >= len(frames) - 1:
                    self.idx = len(frames) - 1
                    self.direction = -1
                elif self.idx <= 0:
                    self.idx = 0
                    self.direction = 1
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        w, h = self.width(), self.height()
        k = h / 310.0                      

        p.save()
        p.translate(w / 2, h - 12 * k)
        p.scale(1, 0.16)
        radius = min(w * 0.30, 130 * k)
        grad = QRadialGradient(QPointF(0, 0), radius)
        grad.setColorAt(0.0, QColor(31, 76, 121, 70))
        grad.setColorAt(1.0, QColor(31, 76, 121, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(grad))
        p.drawEllipse(QPointF(0, 0), radius, radius)
        p.restore()

        if self.kind == "cheer" and self.movie is not None and self.movie.isValid():
            gif = self.movie.currentPixmap()
            if not gif.isNull():
                gh = min((h - 24 * k) * 0.86, gif.height() * 2.0)      
                gw = gif.width() * gh / gif.height()
                if gw > w - 8:
                    gw = w - 8
                    gh = gif.height() * gw / gif.width()
                p.drawPixmap(QRectF((w - gw) / 2, h - 16 * k - gh, gw, gh), gif, QRectF(gif.rect()))
                return

        frames = self.frames.get(self.kind) or []
        if not frames:
            return
        pm = frames[min(self.idx, len(frames) - 1)]
        target_h = max(40.0, (h - 24 * k) * self.RATIOS[self.kind] * self.boost)
        if self.kind == "cheer":           
            target_h = min(target_h, h - 16 * k - self.hop * k - 4)
        target_w = pm.width() * target_h / pm.height()
        if target_w > w - 8:               
            target_w = w - 8
            target_h = pm.height() * target_w / pm.width()
        if self.kind == "cheer":
            offset = -abs(math.sin(self.phase * 4.5)) * self.hop * k
        else:
            offset = (-math.sin(self.phase * 1.75) * 4 - 4) * k
        target = QRectF((w - target_w) / 2, h - 16 * k - target_h + offset, target_w, target_h)
        p.drawPixmap(target, pm, QRectF(pm.rect()))

class Dots(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.filled = 0
        self.setFixedSize(4 * 10 + 3 * 8, 12)

    def set_filled(self, n):
        self.filled = n
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        accent = QColor(C["accent"])
        for i in range(4):
            x = i * 18 + 1.75
            p.setPen(QPen(accent, 1.5))
            p.setBrush(QBrush(accent) if i < self.filled else Qt.NoBrush)
            p.drawEllipse(QRectF(x, 1.75, 8.5, 8.5))

class CheckBox(QWidget):
    toggled = Signal(bool)

    def __init__(self, checked=False, parent=None):
        super().__init__(parent)
        self.checked = checked
        self.setFixedSize(22, 22)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        accent = QColor(C["accent"])
        p.setPen(QPen(accent, 2))
        p.setBrush(QBrush(accent) if self.checked else Qt.NoBrush)
        p.drawRoundedRect(QRectF(1.5, 1.5, 19, 19), 7, 7)
        if self.checked:
            pen = QPen(QColor("#ffffff"), 2.6)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            p.drawPolyline([QPointF(6.4, 11.4), QPointF(9.8, 14.8), QPointF(15.8, 7.4)])
        if self.hasFocus():
            p.setPen(QPen(accent, 1, Qt.DotLine))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(QRectF(0.5, 0.5, 21, 21), 8, 8)

    def _toggle(self):
        self.checked = not self.checked
        self.update()
        self.toggled.emit(self.checked)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.rect().contains(event.position().toPoint()):
            self._toggle()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            self._toggle()
        else:
            super().keyPressEvent(event)

class TodoRow(QWidget):
    toggled = Signal(int, bool)
    deleted = Signal(int)

    def __init__(self, todo, compact=False, parent=None):
        super().__init__(parent)
        self.todo_id = todo["id"]
        self.setObjectName("row")
        self.setAttribute(Qt.WA_StyledBackground, True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(*((6, 3, 4, 3) if compact else (8, 6, 6, 6)))
        lay.setSpacing(10)

        self.box = CheckBox(todo["done"])
        self.box.toggled.connect(lambda state: self.toggled.emit(self.todo_id, state))
        lay.addWidget(self.box, 0, Qt.AlignVCenter)

        self.label = QLabel(todo["text"])
        self.label.setWordWrap(True)
        font = body_font(13 if compact else 14)
        font.setStrikeOut(todo["done"])
        self.label.setFont(font)
        self.label.setStyleSheet(f"color: {C['muted'] if todo['done'] else C['ink']};")
        lay.addWidget(self.label, 1)

        self.remove = QPushButton()
        self.remove.setObjectName("remove")
        self.remove.setFixedSize(24, 24)
        self.remove.setCursor(Qt.PointingHandCursor)
        self.remove.setIcon(svg_icon(SVG_CLOSE, 14, C["muted"]))
        self.remove.setIconSize(QSize(14, 14))
        self.remove.setToolTip("Delete task")
        self.remove.clicked.connect(lambda: self.deleted.emit(self.todo_id))
        lay.addWidget(self.remove, 0, Qt.AlignVCenter)

STYLE = f"""
QMainWindow, QWidget#root {{ background: {C['bg']}; }}
QLabel {{ background: transparent; }}
QFrame#card {{ background: {C['card']}; border: 1px solid {C['line']}; border-radius: 28px; }}
QFrame#side {{ background: {C['panel']}; border: none; border-right: 1px solid {C['line']};
               border-top-left-radius: 27px; border-bottom-left-radius: 27px; }}
QFrame#main {{ background: transparent; border: none; }}
QFrame#div {{ background: {C['line']}; border: none; }}
QLineEdit {{ background: transparent; border: none; border-bottom: 2px solid {C['accent']};
            padding: 6px 2px; color: {C['ink']}; selection-background-color: {C['soft']}; }}
QLineEdit#todoInput {{ border-bottom: 1px solid {C['accent']}; }}
QPushButton#chip {{ background: {C['card']}; border: 1px solid {C['line']}; border-radius: 16px;
                   padding: 5px 11px; color: {C['muted']}; }}
QPushButton#chip:hover {{ border-color: {C['accent']}; color: {C['accent']}; }}
QPushButton#chip:checked {{ background: {C['soft']}; border-color: {C['accent']}; color: {C['accent']}; }}
QPushButton#lock {{ background: transparent; border: 2px solid {C['accent']}; border-radius: 22px;
                   padding: 10px 16px; color: {C['accent']}; }}
QPushButton#lock:hover {{ background: {C['accent']}; color: #ffffff; }}
QPushButton#plus {{ background: {C['accent']}; border: none; border-radius: 18px; }}
QPushButton#plus:hover {{ background: {C['btn_bottom']}; }}
QPushButton#remove {{ background: transparent; border: none; border-radius: 8px; }}
QPushButton#remove:hover {{ background: {C['card']}; }}
QWidget#row {{ background: transparent; border-radius: 12px; }}
QWidget#row:hover {{ background: {C['soft']}; }}
QFrame#tabs {{ background: {C['soft']}; border: none; border-radius: 20px; }}
QPushButton#tab {{ background: transparent; border: none; border-radius: 16px; padding: 8px 22px; color: {C['muted']}; }}
QPushButton#tab:checked {{ background: {C['card']}; color: {C['accent']}; }}
QPushButton#tool {{ background: {C['card']}; border: 1px solid {C['line']}; border-radius: 17px; }}
QPushButton#tool:hover {{ border-color: {C['accent']}; }}
QPushButton#go {{ border: none; border-radius: 27px; color: #ffffff;
                  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {C['btn_top']}, stop:1 {C['btn_bottom']}); }}
QPushButton#go:hover {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4584be, stop:1 #316699); }}
QPushButton#go:pressed {{ background: {C['btn_bottom']}; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ width: 8px; background: transparent; margin: 0; }}
QScrollBar::handle:vertical {{ background: {C['line']}; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
"""

def fmt_time(seconds):
    mins, secs = divmod(max(0, seconds), 60)
    return f"{mins:02d}:{secs:02d}"

SVG_PAUSE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"><path d="M9 6v12M15 6v12"/></svg>'
SVG_CHEVRON_UP = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M6 15l6-6 6 6"/></svg>'
SVG_CHEVRON_DOWN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>'

OVERLAY_STYLE = f"""
QFrame#ocard {{ background: {C['card']}; border: 1px solid {C['line']}; border-radius: 22px; }}
QLabel#bubble {{ background: {C['soft']}; border-radius: 10px; padding: 4px 8px; color: {C['ink']}; }}
QPushButton#mini {{ background: {C['soft']}; border: none; border-radius: 13px; }}
QPushButton#mini:hover {{ background: {C['line']}; }}
QPushButton#miniPlus {{ background: {C['accent']}; border: none; border-radius: 14px; }}
QPushButton#miniPlus:hover {{ background: {C['btn_bottom']}; }}
"""

class CompanionOverlay(QWidget):
    WIDTH = 256

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedWidth(self.WIDTH)

        self.dashboard = None            
        self.movie = None
        self.base_anim = "away"       
        self._cheer_until = 0.0
        self.expanded = True
        self._placed = False
        self.old_pos = None
        self.last_reaction_time = time.time()
        self.COOLDOWN = 15.0
        self.distraction_count = 0

        self._build()
        self.setStyleSheet(STYLE + OVERLAY_STYLE)
        self.set_animation("away")

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        card = QFrame()
        card.setObjectName("ocard")
        outer.addWidget(card)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(8)

        self.sprite_label = QLabel()
        self.sprite_label.setFixedSize(116, 116)      
        self.sprite_label.setAlignment(Qt.AlignCenter)
        self.sprite = Stage()
        self.sprite.setMinimumSize(0, 0)
        self.sprite.setFixedSize(116, 116)
        self.sprite.boost = 1.15
        self.sprite.hop = 38
        self.sprite_stack = QStackedWidget()
        self.sprite_stack.setFixedSize(116, 116)
        self.sprite_stack.addWidget(self.sprite)        
        self.sprite_stack.addWidget(self.sprite_label)  
        header.addWidget(self.sprite_stack)

        right = QVBoxLayout()
        right.setSpacing(0)
        top = QHBoxLayout()
        top.setSpacing(2)
        self.mode_label = eyebrow("Focus")
        top.addWidget(self.mode_label)
        top.addStretch(1)
        self.collapse_btn = QPushButton()
        self.pause_btn = QPushButton()
        for btn in (self.collapse_btn, self.pause_btn):
            btn.setObjectName("mini")
            btn.setFixedSize(26, 26)
            btn.setIconSize(QSize(14, 14))
            btn.setCursor(Qt.PointingHandCursor)
            top.addWidget(btn)
        self.pause_btn.setIcon(svg_icon(SVG_PAUSE, 14, C["accent"]))
        self.pause_btn.setToolTip("Pause and open the dashboard")
        self.pause_btn.clicked.connect(self.show_dashboard)
        self.collapse_btn.setIcon(svg_icon(SVG_CHEVRON_UP, 14, C["accent"]))
        self.collapse_btn.setToolTip("Hide the to-do list")
        self.collapse_btn.clicked.connect(self.toggle_todos)
        right.addLayout(top)
        right.addStretch(1)

        self.timer_label = QLabel("25:00")
        self.timer_label.setFont(display_font(32))
        self.timer_label.setStyleSheet(f"color: {C['ink']};")
        right.addWidget(self.timer_label)

        self.task_label = QLabel("")
        self.task_label.setFont(body_font(11, QFont.Weight.Bold))
        self.task_label.setStyleSheet(f"color: {C['muted']};")
        right.addWidget(self.task_label)
        right.addStretch(1)
        header.addLayout(right, 1)
        lay.addLayout(header)

        self.speech = QLabel("Ready...")
        self.speech.setObjectName("bubble")
        self.speech.setWordWrap(True)
        self.speech.setMinimumHeight(30)
        self.speech.setFont(body_font(11, QFont.Weight.Bold))
        lay.addWidget(self.speech)

        self.todo_box = QWidget()
        box = QVBoxLayout(self.todo_box)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(5)

        head = QHBoxLayout()
        head.addWidget(eyebrow("To-do"))
        head.addStretch(1)
        self.count_label = QLabel("")
        self.count_label.setFont(body_font(12, QFont.Weight.ExtraBold))
        self.count_label.setStyleSheet(f"color: {C['accent']};")
        head.addWidget(self.count_label)
        box.addLayout(head)

        self.todo_scroll = QScrollArea()
        self.todo_scroll.setWidgetResizable(True)
        self.todo_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.todo_scroll.setFixedHeight(68)       
        inner = QWidget()
        self.todo_layout = QVBoxLayout(inner)
        self.todo_layout.setContentsMargins(0, 0, 4, 0)
        self.todo_layout.setSpacing(1)
        self.todo_scroll.setWidget(inner)
        box.addWidget(self.todo_scroll)

        add_row = QHBoxLayout()
        add_row.setSpacing(8)
        self.todo_input = QLineEdit()
        self.todo_input.setObjectName("todoInput")
        self.todo_input.setPlaceholderText("Add a task")
        self.todo_input.setMaxLength(80)
        self.todo_input.setFont(body_font(12))
        self.todo_input.returnPressed.connect(self._add_todo)
        plus = QPushButton()
        plus.setObjectName("miniPlus")
        plus.setFixedSize(24, 24)
        plus.setIcon(svg_icon(SVG_PLUS, 12, "#ffffff"))
        plus.setIconSize(QSize(12, 12))
        plus.setCursor(Qt.PointingHandCursor)
        plus.clicked.connect(self._add_todo)
        add_row.addWidget(self.todo_input, 1)
        add_row.addWidget(plus)
        box.addLayout(add_row)
        lay.addWidget(self.todo_box)

    PNG_KINDS = {"focused": "focus", "break": "break"}
    FALLBACK_KINDS = {"distracted": "pause", "away": "idle", "complete": "cheer"}

    def _show_frames(self, kind):
        if not self.sprite.frames.get(kind):
            return False
        if self.movie is not None:
            self.movie.stop()
        self.sprite.set_kind(kind)
        self.sprite_stack.setCurrentIndex(0)
        return True

    def set_animation(self, state_name):
        self.base_anim = state_name
        if time.time() < self._cheer_until:
            return                      
        self._apply_animation(state_name)

    def cheer_once(self, ms=1300):
        self._cheer_until = time.time() + ms / 1000.0
        self._apply_animation("complete")
        QTimer.singleShot(ms + 60, self._end_cheer)

    def _end_cheer(self):
        if time.time() >= self._cheer_until:
            self._apply_animation(self.base_anim)

    def _apply_animation(self, state_name):
        kind = self.PNG_KINDS.get(state_name)
        if kind and self._show_frames(kind):
            return
        path = complete_gif_path() if state_name == "complete" else ASSETS_DIR / f"{state_name}.gif"
        if path is not None and path.exists():
            self.sprite_label.setStyleSheet("")
            self.sprite_label.setText("")
            self.movie = QMovie(str(path))
            self.movie.jumpToFrame(0)
            size = self.movie.currentImage().size()
            if size.width() > 0 and size.height() > 0:
                size.scale(self.sprite_label.size(), Qt.KeepAspectRatio)   
                self.movie.setScaledSize(size)
            self.sprite_label.setMovie(self.movie)
            self.movie.start()
            self.sprite_stack.setCurrentIndex(1)
            return
        if self._show_frames(self.FALLBACK_KINDS.get(state_name, "idle")):
            return
        self.sprite_label.setText(f"🎬\n{state_name}.gif")
        self.sprite_label.setStyleSheet("border: 1px dashed gray; color: gray; font-size: 10px;")
        self.sprite_stack.setCurrentIndex(1)

    def sync(self):
        d = self.dashboard
        if d is None:
            return
        on_break = d.timer_state == "BREAK"
        self.mode_label.setText("BREAK" if on_break else "FOCUS")
        task = "Rest your eyes" if on_break else (d.task or "No focus set")
        metrics = QFontMetrics(self.task_label.font())
        self.task_label.setText(metrics.elidedText(task, Qt.ElideRight, 112))

        bar = self.todo_scroll.verticalScrollBar()
        keep = bar.value()
        while self.todo_layout.count():
            item = self.todo_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not d.todos:
            empty = QLabel("Nothing here yet.")
            empty.setFont(body_font(12))
            empty.setStyleSheet(f"color: {C['muted']}; padding: 4px;")
            self.todo_layout.addWidget(empty)
        for todo in d.todos:
            row = TodoRow(todo, compact=True)
            row.toggled.connect(d.on_todo_toggled)
            row.deleted.connect(d.on_todo_deleted)
            self.todo_layout.addWidget(row)
        self.todo_layout.addStretch(1)
        done = sum(1 for t in d.todos if t["done"])
        self.count_label.setText(f"{done} of {len(d.todos)} done" if d.todos else "")
        QTimer.singleShot(0, lambda: bar.setValue(keep))

    def _add_todo(self):
        text = self.todo_input.text().strip()
        if text and self.dashboard is not None:
            self.todo_input.clear()
            self.dashboard.add_todo_text(text)
            bar = self.todo_scroll.verticalScrollBar()
            QTimer.singleShot(40, lambda: bar.setValue(bar.maximum()))   

    def toggle_todos(self):
        self.expanded = not self.expanded
        self.todo_box.setVisible(self.expanded)
        icon = SVG_CHEVRON_UP if self.expanded else SVG_CHEVRON_DOWN
        self.collapse_btn.setIcon(svg_icon(icon, 14, C["accent"]))
        self.collapse_btn.setToolTip("Hide the to-do list" if self.expanded else "Show the to-do list")
        self.adjustSize()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._placed:                      
            self._placed = True
            self.adjustSize()
            geo = QGuiApplication.primaryScreen().availableGeometry()
            self.move(geo.right() - self.width() - 20, geo.bottom() - self.height() - 20)

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
        if self.dashboard is not None:
            self.dashboard.pause_from_overlay()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Timo")
        self.BASE_W, self.BASE_H = 820, 580
        self.setMinimumSize(680, 520)
        geo = QGuiApplication.primaryScreen().availableGeometry()
        w = max(680, min(self.BASE_W, int(geo.width() * 0.85)))
        h = max(520, min(self.BASE_H, int(geo.height() * 0.85)))
        self.resize(w, h)
        self.move(geo.center().x() - w // 2, geo.center().y() - h // 2)
        self._last_scale = None

        self.mode = "focus"            
        self.focus_min = 25
        self.break_min = 5
        self.task = ""
        self.todos = [{"id": i + 1, "text": t, "done": False} for i, t in enumerate(SAMPLE_TASKS)]
        self.next_id = 10
        self.harvest_min = 0
        self.harvest_tasks = 0
        self.completed = 0
        self.sound = True
        self.sample = True
        self.note = ""
        self.celebrate_until = 0.0
        self.task_cheer_until = 0.0
        self._load_state()

        self.timer_state = "IDLE"
        self.time_left = self.focus_min * 60
        self.pomodoro_timer = QTimer()
        self.pomodoro_timer.timeout.connect(self.update_timer)

        self._build_ui()
        self.setStyleSheet(STYLE)

        self.overlay = CompanionOverlay()
        self.overlay.dashboard = self

        self.vision = VisionWorker()
        self.vision.state_changed.connect(self.on_state_changed)   

        self.models_loading = True
        self.loader = ModelLoader()
        self.loader.loaded.connect(self._on_models_loaded)
        QTimer.singleShot(250, self.loader.start)

        self.refresh()
        self._apply_scale()

    def _load_state(self):
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return
        for key in ("focus_min", "break_min", "task", "todos", "next_id", "harvest_min",
                    "harvest_tasks", "completed", "sound", "sample"):
            if key in data:
                setattr(self, key, data[key])

    def _save_state(self):
        try:
            data = {k: getattr(self, k) for k in ("focus_min", "break_min", "task", "todos", "next_id",
                                                  "harvest_min", "harvest_tasks", "completed", "sound", "sample")}
            STATE_FILE.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            pass

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        self.outer = outer
        outer.setContentsMargins(20, 20, 20, 20)

        card = QFrame()
        card.setObjectName("card")
        outer.addWidget(card)
        row = QHBoxLayout(card)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self._build_side())
        row.addWidget(self._build_main(), 1)

    def _build_side(self):
        side = QFrame()
        side.setObjectName("side")
        self.side = side
        side.setFixedWidth(290)
        lay = QVBoxLayout(side)
        self.side_layout = lay
        lay.setContentsMargins(24, 22, 24, 18)
        lay.setSpacing(10)

        brand = QHBoxLayout()
        brand.setSpacing(10)
        logo = QLabel()
        logo.setPixmap(svg_pixmap(SVG_LOGO, 30, C["accent"]))
        logo.setFixedSize(30, 30)
        name = QLabel("Timo")
        name.setFont(display_font(26))
        name.setStyleSheet(f"color: {C['accent']};")
        brand.addWidget(logo)
        brand.addWidget(name)
        brand.addStretch(1)
        lay.addLayout(brand)

        head = QVBoxLayout()
        head.setSpacing(4)
        head.addWidget(eyebrow("Your focus"))
        ask = QLabel("What are we working on?")
        ask.setWordWrap(True)
        ask.setFont(display_font(26))
        ask.setStyleSheet(f"color: {C['ink']};")
        self.ask_label = ask
        head.addWidget(ask)
        lay.addLayout(head)

        self.task_input = QLineEdit(self.task)
        self.task_input.setPlaceholderText("Add a focus, e.g. Chapter 4 notes")
        self.task_input.setMaxLength(60)
        self.task_input.setFont(body_font(15, QFont.Weight.Bold))
        self.task_input.returnPressed.connect(self.lock_in)
        lay.addWidget(self.task_input)

        lengths = QVBoxLayout()
        lengths.setSpacing(8)
        self.len_label = eyebrow("Session length")
        lengths.addWidget(self.len_label)
        self.chip_row = QHBoxLayout()
        self.chip_row.setSpacing(8)
        lengths.addLayout(self.chip_row)
        lay.addLayout(lengths)

        self.lock_btn = QPushButton("LOCK THIS IN")
        self.lock_btn.setObjectName("lock")
        self.lock_btn.setFont(body_font(12, QFont.Weight.ExtraBold, 2.0))
        self.lock_btn.setCursor(Qt.PointingHandCursor)
        self.lock_btn.clicked.connect(self.lock_in)
        lay.addWidget(self.lock_btn)

        lay.addWidget(self._divider())

        todo_head = QHBoxLayout()
        todo_head.addWidget(eyebrow("To-do for this session"))
        todo_head.addStretch(1)
        self.count_label = QLabel("")
        self.count_label.setFont(body_font(12, QFont.Weight.ExtraBold))
        self.count_label.setStyleSheet(f"color: {C['accent']};")
        todo_head.addWidget(self.count_label)
        lay.addLayout(todo_head)

        add_row = QHBoxLayout()
        add_row.setSpacing(8)
        self.todo_input = QLineEdit()
        self.todo_input.setObjectName("todoInput")
        self.todo_input.setPlaceholderText("Add a task")
        self.todo_input.setMaxLength(80)
        self.todo_input.setFont(body_font(14))
        self.todo_input.returnPressed.connect(self.add_todo)
        plus = QPushButton()
        plus.setObjectName("plus")
        plus.setFixedSize(36, 36)
        plus.setCursor(Qt.PointingHandCursor)
        plus.setIcon(svg_icon(SVG_PLUS, 18, "#ffffff"))
        plus.setIconSize(QSize(18, 18))
        plus.setToolTip("Add task")
        plus.clicked.connect(self.add_todo)
        add_row.addWidget(self.todo_input, 1)
        add_row.addWidget(plus)
        lay.addLayout(add_row)

        self.todo_scroll = QScrollArea()
        self.todo_scroll.setWidgetResizable(True)
        self.todo_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.todo_scroll.setMinimumHeight(48)
        inner = QWidget()
        self.todo_layout = QVBoxLayout(inner)
        self.todo_layout.setContentsMargins(0, 0, 4, 0)
        self.todo_layout.setSpacing(2)
        self.todo_scroll.setWidget(inner)
        lay.addWidget(self.todo_scroll, 1)

        self.sample_label = QLabel("Sample tasks. Delete them or add your own.")
        self.sample_label.setFont(body_font(12))
        self.sample_label.setStyleSheet(f"color: {C['muted']};")
        lay.addWidget(self.sample_label)

        lay.addWidget(self._divider())

        harvest = QHBoxLayout()
        harvest.setSpacing(12)
        clock = QLabel()
        clock.setPixmap(svg_pixmap(SVG_CLOCK, 22, C["accent"]))
        clock.setFixedSize(22, 22)
        harvest.addWidget(clock, 0, Qt.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(1)
        col.addWidget(eyebrow("Total harvest"))
        self.harvest_big = QLabel("")
        self.harvest_big.setFont(body_font(14, QFont.Weight.ExtraBold))
        self.harvest_big.setStyleSheet(f"color: {C['ink']};")
        self.harvest_sub = QLabel("")
        self.harvest_sub.setFont(body_font(12))
        self.harvest_sub.setStyleSheet(f"color: {C['muted']};")
        col.addWidget(self.harvest_big)
        col.addWidget(self.harvest_sub)
        harvest.addLayout(col, 1)
        lay.addLayout(harvest)
        return side

    def _divider(self):
        d = QFrame()
        d.setObjectName("div")
        d.setFixedHeight(1)
        return d

    def _build_main(self):
        main = QFrame()
        main.setObjectName("main")
        lay = QVBoxLayout(main)
        self.main_layout = lay
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(6)

        top = QHBoxLayout()
        left_pad = QWidget()
        left_pad.setFixedWidth(76)
        top.addWidget(left_pad)
        top.addStretch(1)

        tabs = QFrame()
        tabs.setObjectName("tabs")
        tabs_lay = QHBoxLayout(tabs)
        tabs_lay.setContentsMargins(4, 4, 4, 4)
        tabs_lay.setSpacing(2)
        self.tab_focus = QPushButton("FOCUS")
        self.tab_break = QPushButton("BREAK")
        group = QButtonGroup(self)
        group.setExclusive(True)
        for btn, mode in ((self.tab_focus, "focus"), (self.tab_break, "break")):
            btn.setObjectName("tab")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFont(body_font(11, QFont.Weight.ExtraBold, 1.8))
            btn.clicked.connect(lambda _=False, m=mode: self.set_mode(m))
            group.addButton(btn)
            tabs_lay.addWidget(btn)
        top.addWidget(tabs)
        top.addStretch(1)

        tools = QWidget()
        tools.setFixedWidth(76)
        tools_lay = QHBoxLayout(tools)
        tools_lay.setContentsMargins(0, 0, 0, 0)
        tools_lay.setSpacing(8)
        tools_lay.addStretch(1)
        self.reset_btn = QPushButton()
        self.sound_btn = QPushButton()
        for btn in (self.reset_btn, self.sound_btn):
            btn.setObjectName("tool")
            btn.setFixedSize(34, 34)
            btn.setIconSize(QSize(17, 17))
            btn.setCursor(Qt.PointingHandCursor)
            tools_lay.addWidget(btn)
        self.reset_btn.setIcon(svg_icon(SVG_RESET, 17, C["muted"]))
        self.reset_btn.setToolTip("Reset timer")
        self.reset_btn.clicked.connect(self.reset_timer)
        self.sound_btn.clicked.connect(self.toggle_sound)
        top.addWidget(tools)
        lay.addLayout(top)

        self.time_label = QLabel("25:00")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setFont(display_font(88))
        self.time_label.setStyleSheet(f"color: {C['ink']};")
        lay.addWidget(self.time_label)

        self.stage = Stage()
        lay.addWidget(self.stage, 1)          

        self.task_label = QLabel("")
        self.task_label.setAlignment(Qt.AlignCenter)
        self.task_label.setWordWrap(True)
        self.task_label.setFont(body_font(16, QFont.Weight.ExtraBold))
        self.task_label.setStyleSheet(f"color: {C['ink']};")
        lay.addWidget(self.task_label)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(22)
        self.status_label.setFont(body_font(13))
        self.status_label.setStyleSheet(f"color: {C['muted']};")
        lay.addWidget(self.status_label)

        go_row = QHBoxLayout()
        go_row.addStretch(1)
        self.go_btn = QPushButton("START FOCUS")
        self.go_btn.setObjectName("go")
        self.go_btn.setFixedSize(360, 50)
        self.go_btn.setFont(body_font(13, QFont.Weight.ExtraBold, 2.6))
        self.go_btn.setCursor(Qt.PointingHandCursor)
        self.go_btn.clicked.connect(self.on_go)
        go_row.addWidget(self.go_btn)
        go_row.addStretch(1)
        lay.addLayout(go_row)

        dots_row = QHBoxLayout()
        dots_row.setSpacing(12)
        dots_row.addStretch(1)
        self.dots = Dots()
        self.session_label = eyebrow("Session 1 of 4")
        dots_row.addWidget(self.dots)
        dots_row.addWidget(self.session_label)
        dots_row.addStretch(1)
        lay.addSpacing(4)
        lay.addLayout(dots_row)
        return main

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "time_label"):
            self._apply_scale()

    def _apply_scale(self):
        w, h = self.width(), self.height()
        s = max(0.7, min(w / self.BASE_W, h / self.BASE_H, 1.6))
        compact = h < 600 or w < 760
        key = (round(s, 2), compact, w // 20)
        if key == self._last_scale:
            return
        self._last_scale = key

        side_w = int(max(270, min(360, w * 0.34)))
        self.side.setFixedWidth(side_w)
        self.time_label.setFont(display_font(int(max(52, min(130, 88 * s)))))
        self.ask_label.setFont(display_font(int(max(20, min(32, 26 * s)))))
        self.go_btn.setFixedSize(int(max(220, min(440, (w - side_w) * 0.72))), int(max(42, min(60, 50 * s))))

        self.outer.setContentsMargins(*(12, 12, 12, 12) if compact else (20, 20, 20, 20))
        self.side_layout.setSpacing(8 if compact else 12)
        self.side_layout.setContentsMargins(*(20, 16, 20, 14) if compact else (24, 22, 24, 18))
        self.main_layout.setContentsMargins(*(14, 12, 14, 12) if compact else (20, 18, 20, 18))
        self.sample_label.setVisible(bool(self.sample and self.todos) and h >= 600)

    def _stage_kind(self):
        if time.time() < max(self.celebrate_until, self.task_cheer_until):
            return "cheer"
        if self.mode == "break":
            return "break"
        if self.pomodoro_timer.isActive():
            return "focus"
        if self.timer_state == "FOCUS":
            return "pause"
        return "idle"

    def celebrate_task(self):
        ms = complete_gif_ms()
        self.task_cheer_until = time.time() + ms / 1000.0
        self.stage.set_kind("cheer")
        if self.overlay.isVisible():
            self.overlay.cheer_once(ms)
        QTimer.singleShot(ms + 100, self._end_task_cheer)

    def _end_task_cheer(self):
        if time.time() >= self.task_cheer_until:
            self.stage.set_kind(self._stage_kind())

    def _on_models_loaded(self, ok):
        self.models_loading = False
        if not ok:
            self.note = "Focus camera models could not load. Timer still works. Check the terminal."
        self.refresh()

    def _stop_timer(self):
        self.pomodoro_timer.stop()
        self.vision.stop_camera()

    def total_seconds(self):
        return (self.focus_min if self.mode == "focus" else self.break_min) * 60

    def is_paused(self):
        return self.timer_state in ("FOCUS", "BREAK") and not self.pomodoro_timer.isActive()

    def _paint_time(self):
        self.time_label.setText(fmt_time(self.time_left))

    def refresh(self):
        self._paint_time()
        self.tab_focus.setChecked(self.mode == "focus")
        self.tab_break.setChecked(self.mode == "break")

        if self.is_paused():
            self.go_btn.setText("RESUME")
        else:
            self.go_btn.setText("START FOCUS" if self.mode == "focus" else "START BREAK")

        filled = 4 if (self.completed > 0 and self.completed % 4 == 0 and self.mode == "break") else self.completed % 4
        if self.mode == "focus":
            number = filled + 1
        elif self.completed == 0:
            number = 1
        else:
            number = ((self.completed - 1) % 4) + 1
        self.dots.set_filled(filled)
        self.session_label.setText(f"SESSION {number} OF 4")

        self.task_label.setText("Rest your eyes" if self.mode == "break" else (self.task or "Ready when you are"))
        self.status_label.setText(self.note or ("Getting the focus camera ready in the background..." if self.models_loading else ""))

        tasks = self.harvest_tasks
        self.harvest_big.setText(f"{self.harvest_min} min focused")
        self.harvest_sub.setText(f"{tasks} {'task' if tasks == 1 else 'tasks'} finished")

        self.sound_btn.setIcon(svg_icon(SVG_SOUND_ON if self.sound else SVG_SOUND_OFF, 17, C["muted"]))
        self.sound_btn.setToolTip("Sound on" if self.sound else "Sound off")

        self._render_chips()
        self._render_todos()

        self.stage.set_kind(self._stage_kind())
        if getattr(self, "overlay", None) is not None:
            self.overlay.sync()
        self._save_state()

    def _render_chips(self):
        while self.chip_row.count():
            item = self.chip_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        options = FOCUS_OPTIONS if self.mode == "focus" else BREAK_OPTIONS
        current = self.focus_min if self.mode == "focus" else self.break_min
        self.len_label.setText("SESSION LENGTH (MIN)" if self.mode == "focus" else "BREAK LENGTH (MIN)")
        for minutes in options:
            chip = QPushButton(str(minutes))
            chip.setObjectName("chip")
            chip.setCheckable(True)
            chip.setChecked(minutes == current)
            chip.setCursor(Qt.PointingHandCursor)
            chip.setFont(body_font(13, QFont.Weight.ExtraBold))
            chip.clicked.connect(lambda _=False, m=minutes: self.set_minutes(m))
            self.chip_row.addWidget(chip)
        self.chip_row.addStretch(1)

    def _render_todos(self):
        while self.todo_layout.count():
            item = self.todo_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not self.todos:
            empty = QLabel("Nothing here yet. Add what you want done before the timer ends.")
            empty.setWordWrap(True)
            empty.setFont(body_font(13))
            empty.setStyleSheet(f"color: {C['muted']}; padding: 6px 4px;")
            self.todo_layout.addWidget(empty)
        for todo in self.todos:
            row = TodoRow(todo)
            row.toggled.connect(self.on_todo_toggled)
            row.deleted.connect(self.on_todo_deleted)
            self.todo_layout.addWidget(row)
        self.todo_layout.addStretch(1)
        done = sum(1 for t in self.todos if t["done"])
        self.count_label.setText(f"{done} of {len(self.todos)} done" if self.todos else "")
        self.sample_label.setVisible(bool(self.sample and self.todos) and self.height() >= 600)

    def add_todo(self):
        text = self.todo_input.text().strip()
        if text:
            self.todo_input.clear()
            self.add_todo_text(text)

    def add_todo_text(self, text):
        self.todos.append({"id": self.next_id, "text": text, "done": False})
        self.next_id += 1
        self.sample = False
        self.refresh()

    def on_todo_toggled(self, todo_id, state):
        for t in self.todos:
            if t["id"] == todo_id:
                t["done"] = state
        self.sample = False
        self.refresh()
        if state:                          
            self.celebrate_task()

    def on_todo_deleted(self, todo_id):
        self.todos = [t for t in self.todos if t["id"] != todo_id]
        self.sample = False
        self.refresh()

    def set_mode(self, mode):
        if mode == self.mode:
            return
        self._stop_timer()
        self.timer_state = "IDLE"
        self.mode = mode
        self.note = ""
        self.time_left = self.total_seconds()
        self.refresh()

    def set_minutes(self, minutes):
        if self.mode == "focus":
            self.focus_min = minutes
        else:
            self.break_min = minutes
        self._stop_timer()
        self.timer_state = "IDLE"
        self.time_left = self.total_seconds()
        self.refresh()

    def lock_in(self):
        self.task = self.task_input.text().strip()[:60]
        self._stop_timer()
        self.mode = "focus"
        self.timer_state = "IDLE"
        self.time_left = self.focus_min * 60
        self.note = "Locked in. Press start when you're ready." if self.task else ""
        self.refresh()

    def reset_timer(self):
        self._stop_timer()
        self.timer_state = "IDLE"
        self.celebrate_until = 0.0
        self.note = ""
        self.time_left = self.total_seconds()
        self.refresh()

    def toggle_sound(self):
        self.sound = not self.sound
        self.refresh()

    def on_go(self):
        if self.is_paused():
            if self.timer_state == "FOCUS":
                self.start_focus(resume=True)
            else:
                self.start_break(resume=True)
        elif self.mode == "focus":
            self.start_focus()
        else:
            self.start_break()

    def pause_from_overlay(self):
        self._stop_timer()
        self.note = "Paused. Press resume to continue."
        self.refresh()
        self.show()
        self.raise_()
        self.activateWindow()

    def start_focus(self, resume=False):
        self.mode = "focus"
        self.timer_state = "FOCUS"
        if not resume:
            self.time_left = self.focus_min * 60
        self.note = ""
        self.celebrate_until = 0.0
        self.pomodoro_timer.start(1000)
        self.vision.start_camera()         

        self.hide()
        self.overlay.show()
        self.overlay.set_animation("focused")
        self.overlay.timer_label.setText(fmt_time(self.time_left))
        self.overlay.speech.setText("Back to it!" if resume else "Let's get to work!")
        self.refresh()

    def start_break(self, resume=False):
        self.mode = "break"
        self.timer_state = "BREAK"
        if not resume:
            self.time_left = self.break_min * 60
        self.note = ""
        self.vision.stop_camera()          
        self.pomodoro_timer.start(1000)

        self.hide()
        self.overlay.show()
        self.overlay.set_animation("break")
        self.overlay.timer_label.setText(fmt_time(self.time_left))
        self.overlay.speech.setText("Time for a break!")
        self.refresh()

    def _finish_focus_session(self):
        done = sum(1 for t in self.todos if t["done"])
        self.completed += 1
        self.harvest_min += self.focus_min
        self.harvest_tasks += done
        self.todos = [t for t in self.todos if not t["done"]]
        self.note = (f"{done} {'task' if done == 1 else 'tasks'} finished. " if done else "Session done. ") + "Nice work."

    def update_timer(self):
        self.time_left -= 1
        self.overlay.timer_label.setText(fmt_time(self.time_left))
        self._paint_time()

        if self.time_left <= 0:
            self._stop_timer()
            if self.sound:
                QApplication.beep()
            if self.timer_state == "FOCUS":
                self.timer_state = "COMPLETE"
                self._finish_focus_session()
                self.overlay.set_animation("complete")
                self.overlay.speech.setText("Session complete! Great work!")
                QTimer.singleShot(5000, lambda: self.start_break())
            elif self.timer_state == "BREAK":
                self.timer_state = "IDLE"
                self.mode = "focus"
                self.time_left = self.focus_min * 60
                self.note = "Break over. Ready for the next one."
                self.celebrate_until = time.time() + 6.0
                QTimer.singleShot(6200, self.refresh)
                self.overlay.hide()
                self.refresh()
                self.show()
                self.raise_()

    def on_state_changed(self, state):
        now = time.time()
        time_since_reaction = now - self.overlay.last_reaction_time

        if self.timer_state != "FOCUS":
            return                          

        if state == "FOCUSED":
            if self.timer_state != "COMPLETE":
                self.overlay.set_animation("focused")
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

    def closeEvent(self, event):
        self._save_state()
        self.vision.stop()
        try:
            if ML.landmarker is not None:
                ML.landmarker.close()
        except Exception:
            pass
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    load_app_fonts()

    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("timo.focus.companion")
        except Exception:
            pass
    icon_path = ASSETS_DIR / "timo.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

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