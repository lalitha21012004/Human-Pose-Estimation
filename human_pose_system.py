import math
import os
import tempfile
import time
import urllib.request
from collections import deque

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import streamlit as st

MODEL_URLS = {
    "Lite (Fastest)": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "Full (Balanced)": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "Heavy (Most Accurate)": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

JOINTS = {
    "Left Shoulder": 11,
    "Right Shoulder": 12,
    "Left Elbow": 13,
    "Right Elbow": 14,
    "Left Wrist": 15,
    "Right Wrist": 16,
    "Left Hip": 23,
    "Right Hip": 24,
    "Left Knee": 25,
    "Right Knee": 26,
    "Left Ankle": 27,
    "Right Ankle": 28,
}

ANGLE_JOINTS = {
    "Left Elbow": (11, 13, 15),
    "Right Elbow": (12, 14, 16),
    "Left Shoulder": (23, 11, 13),
    "Right Shoulder": (24, 12, 14),
    "Left Knee": (23, 25, 27),
    "Right Knee": (24, 26, 28),
}

POSE_CONNECTIONS = [
    (c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS
]


def ensure_model(variant):
    os.makedirs("models", exist_ok=True)
    url = MODEL_URLS[variant]
    path = os.path.join("models", os.path.basename(url))
    if not os.path.exists(path):
        with st.spinner(f"Downloading {variant} pose model..."):
            urllib.request.urlretrieve(url, path)
    return path


@st.cache_resource
def load_pose(variant, static_img, det_conf, track_conf):
    model_path = ensure_model(variant)
    mode = vision.RunningMode.IMAGE if static_img else vision.RunningMode.VIDEO
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mode,
        num_poses=1,
        min_pose_detection_confidence=det_conf,
        min_pose_presence_confidence=det_conf,
        min_tracking_confidence=track_conf,
    )
    return vision.PoseLandmarker.create_from_options(opts)


def next_timestamp_ms():
    # Keep timestamps strictly increasing across Streamlit reruns for cached VIDEO landmarker
    prev = st.session_state.get("last_ts_ms", 0)
    now_ms = int(time.monotonic() * 1000)
    ts = now_ms if now_ms > prev else prev + 1
    st.session_state["last_ts_ms"] = ts
    return ts


def joint_angle(a, b, c):
    rad = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    deg = abs(rad * 180.0 / math.pi)
    return 360.0 - deg if deg > 180.0 else deg


def draw_skeleton(rgb, lms, w, h, min_vis):
    for a, b in POSE_CONNECTIONS:
        if a < len(lms) and b < len(lms):
            p1, p2 = lms[a], lms[b]
            if p1.visibility >= min_vis and p2.visibility >= min_vis:
                pt1 = (int(p1.x * w), int(p1.y * h))
                pt2 = (int(p2.x * w), int(p2.y * h))
                cv2.line(rgb, pt1, pt2, (235, 235, 235), 2, cv2.LINE_AA)

    for lm in lms:
        if lm.visibility >= min_vis:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(rgb, (cx, cy), 4, (56, 189, 248), -1, cv2.LINE_AA)


def process_pose(rgb, landmarker, min_vis=0.5, want_table=True, ts_ms=None):
    h, w = rgb.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    if ts_ms is None:
        res = landmarker.detect(mp_img)
    else:
        res = landmarker.detect_for_video(mp_img, ts_ms)

    angles = {k: None for k in ANGLE_JOINTS}
    table_rows = []

    if not res.pose_landmarks:
        cv2.putText(rgb, "No human pose detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 80, 80), 2, cv2.LINE_AA)
        return rgb, angles, table_rows, "No Person Detected"

    lms = res.pose_landmarks[0]
    draw_skeleton(rgb, lms, w, h, min_vis)

    for label, (i1, i2, i3) in ANGLE_JOINTS.items():
        p1, p2, p3 = lms[i1], lms[i2], lms[i3]
        if min(p1.visibility, p2.visibility, p3.visibility) >= min_vis:
            c1 = (p1.x * w, p1.y * h)
            c2 = (p2.x * w, p2.y * h)
            c3 = (p3.x * w, p3.y * h)
            ang = joint_angle(c1, c2, c3)
            angles[label] = ang
            cv2.putText(rgb, f"{int(ang)} deg", (int(c2[0]) + 8, int(c2[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 230, 60), 2, cv2.LINE_AA)

    if want_table:
        for name, idx in JOINTS.items():
            lm = lms[idx]
            table_rows.append({
                "Joint": name,
                "X (norm)": round(lm.x, 3),
                "Y (norm)": round(lm.y, 3),
                "Z (depth)": round(lm.z, 3),
                "Pixel (x, y)": f"({int(lm.x * w)}, {int(lm.y * h)})",
                "Visibility": round(lm.visibility, 2),
            })

    knees = [angles[k] for k in ("Left Knee", "Right Knee") if angles[k] is not None]
    shoulders = [angles[k] for k in ("Left Shoulder", "Right Shoulder") if angles[k] is not None]

    if knees and (sum(knees) / len(knees)) < 125:
        posture = "Squatting / Seated"
    elif shoulders and max(shoulders) > 120:
        posture = "Arm(s) Raised"
    elif knees:
        posture = "Standing / Upright"
    else:
        posture = "Partial Pose Detected"

    cv2.rectangle(rgb, (10, 10), (310, 48), (20, 25, 35), -1)
    cv2.putText(rgb, f"Posture: {posture}", (18, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 160), 2, cv2.LINE_AA)

    return rgb, angles, table_rows, posture


def fit_width(frame, max_w):
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    return cv2.resize(frame, (max_w, int(h * (max_w / w))), interpolation=cv2.INTER_AREA)


st.set_page_config(page_title="Human Pose Estimation", layout="wide")
st.title("Human Pose Estimation & Joint Angle Tracker")
st.caption("OpenCV + MediaPipe Pose Landmarker (33 Landmarks) + Streamlit + Matplotlib")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Input Source", ["Live Webcam", "Upload Video", "Upload Image"])
    variant = st.selectbox("Pose Model", list(MODEL_URLS.keys()), index=1)
    max_w = st.select_slider("Processing Width (px)", options=[320, 480, 640, 960], value=640)
    ui_every = st.slider("Chart/Table Refresh (every N frames)", 1, 15, 4)
    det_conf = st.slider("Min Detection Confidence", 0.1, 1.0, 0.5, 0.05)
    track_conf = st.slider("Min Tracking Confidence", 0.1, 1.0, 0.5, 0.05)
    min_vis = st.slider("Joint Visibility Threshold", 0.1, 0.9, 0.5, 0.05)
    mirror = st.checkbox("Mirror Webcam Feed", value=True)

if mode == "Upload Image":
    up_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if up_img is not None:
        buf = np.frombuffer(up_img.read(), dtype=np.uint8)
        bgr = fit_width(cv2.imdecode(buf, cv2.IMREAD_COLOR), max_w)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        landmarker = load_pose(variant, True, det_conf, track_conf)
        out_rgb, angles, rows, posture = process_pose(rgb, landmarker, min_vis, want_table=True)

        c1, c2 = st.columns([3, 2])
        with c1:
            st.image(out_rgb, channels="RGB", use_container_width=True)
        with c2:
            st.metric("Detected Posture", posture)
            m1, m2 = st.columns(2)
            for idx, (k, v) in enumerate(angles.items()):
                (m1 if idx % 2 == 0 else m2).metric(k, f"{v:.1f} deg" if v is not None else "N/A")
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.stop()

cap = None
tmp_video = None
active = False

if mode == "Live Webcam":
    active = st.checkbox("Start Webcam")
    if active:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
else:
    up_vid = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if up_vid is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(up_vid.read())
            tmp_video = f.name
        active = st.button("Process Video")
        if active:
            cap = cv2.VideoCapture(tmp_video)

vid_col, info_col = st.columns([3, 2])
vid_box = vid_col.empty()
stat_box = info_col.empty()
ang_box = info_col.empty()
plot_box = info_col.empty()
tbl_box = st.empty()

if active and cap is not None:
    if not cap.isOpened():
        st.error("Could not open video source.")
    else:
        hist = deque(maxlen=60)
        reps = 0
        stage = "down"
        frame_idx = 0
        last_t = time.perf_counter()

        landmarker = load_pose(variant, False, det_conf, track_conf)

        fig, ax = plt.subplots(figsize=(6, 2.8))
        plot_lines = {
            "Left Elbow": ax.plot([], [], label="Left Elbow", lw=1.8, color="#2563eb")[0],
            "Right Elbow": ax.plot([], [], label="Right Elbow", lw=1.8, color="#16a34a")[0],
            "Left Knee": ax.plot([], [], label="Left Knee", lw=1.8, color="#ea580c")[0],
            "Right Knee": ax.plot([], [], label="Right Knee", lw=1.8, color="#9333ea")[0],
        }
        ax.set_ylim(0, 190)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angle (deg)")
        ax.set_title("Joint Angle Sequence Over Time")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()

        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                if mode == "Live Webcam" and mirror:
                    frame = cv2.flip(frame, 1)

                frame = fit_width(frame, max_w)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame_idx += 1
                ts_ms = next_timestamp_ms()
                update_ui = (frame_idx % ui_every == 0) or frame_idx == 1

                out_rgb, angles, rows, posture = process_pose(
                    rgb, landmarker, min_vis, want_table=update_ui, ts_ms=ts_ms
                )

                le = angles["Left Elbow"]
                if le is not None:
                    if le > 155:
                        stage = "down"
                    elif le < 45 and stage == "down":
                        stage = "up"
                        reps += 1

                hist.append({
                    "frame": frame_idx,
                    "Left Elbow": angles["Left Elbow"] if angles["Left Elbow"] is not None else np.nan,
                    "Right Elbow": angles["Right Elbow"] if angles["Right Elbow"] is not None else np.nan,
                    "Left Knee": angles["Left Knee"] if angles["Left Knee"] is not None else np.nan,
                    "Right Knee": angles["Right Knee"] if angles["Right Knee"] is not None else np.nan,
                })

                now = time.perf_counter()
                fps = 1.0 / max(now - last_t, 1e-6)
                last_t = now

                vid_box.image(out_rgb, channels="RGB", use_container_width=True)

                if update_ui:
                    stat_box.markdown(
                        f"**Posture:** `{posture}` &nbsp;|&nbsp; **Left Arm Reps:** `{reps}` (`{stage}`) &nbsp;|&nbsp; **FPS:** `{fps:.1f}`"
                    )
                    ang_box.markdown(
                        " | ".join(
                            f"**{k}:** {v:.1f}°" if v is not None else f"**{k}:** --"
                            for k, v in angles.items()
                        )
                    )
                    if len(hist) >= 2:
                        xs = [d["frame"] for d in hist]
                        for k, ln in plot_lines.items():
                            ln.set_data(xs, [d[k] for d in hist])
                        ax.set_xlim(xs[0], xs[-1])
                        fig.canvas.draw_idle()
                    plot_box.pyplot(fig)
                    if rows:
                        tbl_box.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        finally:
            plt.close(fig)
            cap.release()

if tmp_video and os.path.exists(tmp_video):
    os.remove(tmp_video)