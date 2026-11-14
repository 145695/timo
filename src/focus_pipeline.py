"""
Day 1 — FocusCompanion vision pipeline
=======================================
MediaPipe Face Landmarker -> numeric features only (frames discarded):
  EAR (eye aspect ratio), MAR (mouth aspect ratio -> yawns),
  head pose (yaw/pitch/roll), gaze offset, blink score,
  PERCLOS (rolling eye-closed fraction), yawn count, face present.

Run (from the project root, with .venv active):
  python src/focus_pipeline.py
  python src/focus_pipeline.py --camera 1 --print
  python src/focus_pipeline.py --width 960 --height 540

Press q or Esc to quit.

Setup:
  pip install opencv-python mediapipe numpy
  Place face_landmarker.task in models/ (see README / setup instructions).

Day 1 tuning checklist:
  [ ] Blink       -> EAR drops below EAR_CLOSED
  [ ] Yawn        -> MAR rises above MAR_YAWN for > 0.4 s
  [ ] Head turn   -> |yaw| rises above YAW_LIMIT_DEG
  [ ] Look down   -> |pitch| rises above PITCH_LIMIT_DEG
Adjust the constants in the "Thresholds" block on your own face.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# --- Thresholds (tune these on your own face) --------------------------------
EAR_CLOSED        = 0.20   # below -> eyes closed      (resting ~0.30, closed ~0.10)
MAR_YAWN          = 0.55   # above -> mouth open/yawn  (resting ~0.05-0.15)
YAW_LIMIT_DEG     = 25.0   # |yaw|   above -> head turned
PITCH_LIMIT_DEG   = 20.0   # |pitch| above -> head up/down
PERCLOS_WINDOW_S  = 8.0
PERCLOS_THRESHOLD = 0.30
YAWN_WINDOW_S     = 60.0
YAWN_COUNT_THRESHOLD = 2
YAWN_MIN_DURATION_S  = 0.4   # MAR must stay high this long to count
YAWN_COOLDOWN_S      = 1.5   # ignore new yawns for this long after one ends
AWAY_TIMEOUT_S    = 5.0
# -----------------------------------------------------------------------------

# Model lives at <project_root>/models/face_landmarker.task
# (this file is at <project_root>/src/focus_pipeline.py)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

# MediaPipe face-mesh indices
LEFT_EYE_EAR  = (33, 160, 158, 133, 153, 144)   # 6-point EAR
RIGHT_EYE_EAR = (362, 385, 387, 263, 373, 380)
MOUTH_TOP, MOUTH_BOTTOM = 13, 14                 # inner upper/lower lip
MOUTH_LEFT, MOUTH_RIGHT = 78, 308                # inner mouth corners


def _px(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def _ear(pts):
    p1, p2, p3, p4, p5, p6 = pts
    v = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    hz = np.linalg.norm(p1 - p4) + 1e-9
    return float(v / (2.0 * hz))


def _euler(R):
    """(pitch, yaw, roll) in degrees from a 3x3 rotation matrix (ZYX)."""
    sy = math.hypot(R[0, 0], R[1, 0])
    if sy > 1e-6:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = 0.0
    return tuple(math.degrees(a) for a in (pitch, yaw, roll))


def _fmt(v, spec=".2f"):
    return format(v, spec) if v is not None else "nan"


class Extractor:
    def __init__(self):
        self.eye_closed_history = deque()   # (t, bool)
        self.yawn_events        = deque()   # timestamps
        self.yawn_active_since  = None
        self.last_yawn_end_t    = None
        self.last_face_t        = None

    def update(self, result, w, h, t):
        f = {
            "t": t, "face": False, "away_for": 0.0, "away": False,
            "ear_l": None, "ear_r": None, "ear": None, "mar": None,
            "yaw": None, "pitch": None, "roll": None,
            "gaze_x": None, "gaze_y": None, "blink": None,
            "perclos": 0.0, "yawns": 0, "tired": False,
        }
        if not result.face_landmarks:
            if self.last_face_t is None:
                self.last_face_t = t
            f["away_for"] = t - self.last_face_t
            f["away"]     = f["away_for"] > AWAY_TIMEOUT_S
            return f

        self.last_face_t = t
        f["face"] = True
        lm = result.face_landmarks[0]

        # EAR
        left  = [_px(lm[i], w, h) for i in LEFT_EYE_EAR]
        right = [_px(lm[i], w, h) for i in RIGHT_EYE_EAR]
        f["ear_l"] = _ear(left)
        f["ear_r"] = _ear(right)
        f["ear"]   = 0.5 * (f["ear_l"] + f["ear_r"])

        # MAR
        mt = _px(lm[MOUTH_TOP],    w, h); mb = _px(lm[MOUTH_BOTTOM], w, h)
        ml = _px(lm[MOUTH_LEFT],   w, h); mr = _px(lm[MOUTH_RIGHT],  w, h)
        f["mar"] = float(np.linalg.norm(mt - mb) / (np.linalg.norm(ml - mr) + 1e-9))

        # Head pose from facial transformation matrix
        if result.facial_transformation_matrixes:
            M = np.asarray(result.facial_transformation_matrixes[0])
            p, y, r = _euler(M[:3, :3])
            f["pitch"], f["yaw"], f["roll"] = p, y, r

        # Gaze + blink from ARKit-style blendshapes
        if result.face_blendshapes:
            bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
            f["blink"] = 0.5 * (bs.get("eyeBlinkLeft", 0.0) + bs.get("eyeBlinkRight", 0.0))
            gx = (bs.get("eyeLookOutLeft", 0.0) + bs.get("eyeLookInRight", 0.0)
                - bs.get("eyeLookInLeft", 0.0)  - bs.get("eyeLookOutRight", 0.0))
            gy = (bs.get("eyeLookUpLeft", 0.0)  + bs.get("eyeLookUpRight", 0.0)
                - bs.get("eyeLookDownLeft", 0.0) - bs.get("eyeLookDownRight", 0.0))
            f["gaze_x"] = 0.5 * gx
            f["gaze_y"] = 0.5 * gy

        # PERCLOS over rolling window
        eye_closed = f["ear"] < EAR_CLOSED
        self.eye_closed_history.append((t, eye_closed))
        while self.eye_closed_history and t - self.eye_closed_history[0][0] > PERCLOS_WINDOW_S:
            self.eye_closed_history.popleft()
        if self.eye_closed_history:
            f["perclos"] = sum(1 for _, c in self.eye_closed_history if c) / len(self.eye_closed_history)

        # Yawn state machine (duration + cooldown)
        mar_high = f["mar"] > MAR_YAWN
        if mar_high and self.yawn_active_since is None:
            self.yawn_active_since = t
        elif (not mar_high) and self.yawn_active_since is not None:
            dur = t - self.yawn_active_since
            if dur >= YAWN_MIN_DURATION_S and (
                self.last_yawn_end_t is None or t - self.last_yawn_end_t > YAWN_COOLDOWN_S
            ):
                self.yawn_events.append(t)
            self.last_yawn_end_t    = t
            self.yawn_active_since  = None
        while self.yawn_events and t - self.yawn_events[0] > YAWN_WINDOW_S:
            self.yawn_events.popleft()
        f["yawns"] = len(self.yawn_events)

        f["tired"] = f["perclos"] > PERCLOS_THRESHOLD or f["yawns"] >= YAWN_COUNT_THRESHOLD
        return f


def draw_overlay(frame, f, fps, result):
    h, w = frame.shape[:2]
    lines = []

    def add(text, ok=True):
        lines.append((text, (0, 200, 0) if ok else (0, 0, 255)))

    add(f"FPS {fps:5.1f}   Face: {'YES' if f['face'] else 'NO'}", f["face"])
    if f["face"]:
        add(f"EAR  L {f['ear_l']:.2f}  R {f['ear_r']:.2f}  mean {f['ear']:.2f}"
            f"   (closed < {EAR_CLOSED:.2f})", f["ear"] >= EAR_CLOSED)
        add(f"MAR  {f['mar']:.2f}   (yawn > {MAR_YAWN:.2f})", f["mar"] <= MAR_YAWN)
        add(f"Yaw {f['yaw']:+.1f}  Pitch {f['pitch']:+.1f}  Roll {f['roll']:+.1f}   (deg)",
            abs(f["yaw"]) <= YAW_LIMIT_DEG and abs(f["pitch"]) <= PITCH_LIMIT_DEG)
        if f["gaze_x"] is not None:
            add(f"Gaze x {f['gaze_x']:+.2f}  y {f['gaze_y']:+.2f}")
            add(f"Blink {f['blink']:.2f}")
        add(f"PERCLOS {f['perclos']*100:3.0f}%   (tired > {PERCLOS_THRESHOLD*100:.0f}%)",
            f["perclos"] <= PERCLOS_THRESHOLD)
        add(f"Yawns {f['yawns']} / {YAWN_WINDOW_S:.0f}s   (tired >= {YAWN_COUNT_THRESHOLD})",
            f["yawns"] < YAWN_COUNT_THRESHOLD)
        state = "TIRED" if f["tired"] else "OK"
        add(f"Preliminary state: {state}   (fusion arrives Day 3)", not f["tired"])
    else:
        add(f"No face for {f['away_for']:.1f}s   (away > {AWAY_TIMEOUT_S:.0f}s)",
            f["away_for"] <= AWAY_TIMEOUT_S)
        add("Preliminary state: " + ("AWAY" if f["away"] else "NO FACE"), not f["away"])

    pad, lh = 8, 22
    panel_w = min(w - 2 * pad, 600)
    panel_h = pad * 2 + lh * len(lines)
    shade = frame.copy()
    cv2.rectangle(shade, (pad, pad), (pad + panel_w, pad + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(shade, 0.55, frame, 0.45, 0, frame)
    y = pad + 18
    for text, color in lines:
        cv2.putText(frame, text, (pad + 8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
        y += lh

    if result.face_landmarks:
        lm = result.face_landmarks[0]
        for idx in LEFT_EYE_EAR + RIGHT_EYE_EAR:
            p = lm[idx]
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 2, (0, 255, 255), -1)
        for idx in (MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT):
            p = lm[idx]
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, (255, 0, 255), -1)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width",  type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--print", dest="print_csv", action="store_true",
                    help="print one CSV-ish line every 0.5 s (useful for tuning)")
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        print(f"Missing model: {MODEL_PATH}", file=sys.stderr)
        print("Download it from:\n"
              "  https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task", file=sys.stderr)
        return 1

    base = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
    extractor  = Extractor()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}", file=sys.stderr)
        return 1

    t0 = time.time()
    last_t_ms  = -1
    fps_times  = deque(maxlen=30)
    last_print = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed", file=sys.stderr)
                break
            frame = cv2.flip(frame, 1)   # selfie mirror for natural feel
            h, w = frame.shape[:2]

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            t_ms   = max(int((time.time() - t0) * 1000), last_t_ms + 1)
            last_t_ms = t_ms
            result = landmarker.detect_for_video(mp_img, t_ms)

            now = time.time()
            f   = extractor.update(result, w, h, now)

            fps_times.append(now)
            fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0] + 1e-9) \
                  if len(fps_times) >= 2 else 0.0

            display = draw_overlay(frame, f, fps, result)
            cv2.imshow("FocusCompanion - Day 1", display)

            if args.print_csv and now - last_print > 0.5:
                last_print = now
                print(
                    f"t={now - t0:6.1f} face={int(f['face'])} "
                    f"ear={_fmt(f['ear'], '.3f')} mar={_fmt(f['mar'], '.3f')} "
                    f"yaw={_fmt(f['yaw'], '+.1f')} pitch={_fmt(f['pitch'], '+.1f')} "
                    f"perclos={f['perclos']:.2f} yawns={f['yawns']}"
                )

            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())