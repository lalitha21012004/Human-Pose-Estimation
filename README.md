**#Human Pose Estimation & Joint Tracking**
`A real-time human pose estimation and joint-angle analysis dashboard built with OpenCV, MediaPipe Pose, NumPy, Matplotlib, and Streamlit.`

# Features
## Multi-Source Input: `Supports Live Webcam, Uploaded Video (.mp4, .mov, .avi, .mkv), and Static Images (.jpg, .jpeg, .png)`.
### # 33-Keypoint Skeleton Visualization: Detects full-body landmarks (shoulders, elbows, wrists, hips, knees, ankles) using MediaPipe Pose and draws the connected skeleton (POSE_CONNECTIONS) over the video feed.
#### Joint Angle & Posture Tracking: Computes interior joint angles (0°–180°) for elbows, shoulders, and knees using np.arctan2 on high-confidence landmarks (visibility > 0.5), classifies body posture (Standing / Upright, Squatting / Seated, Arm(s) Raised), and counts left-arm curl repetitions.
###### Live Telemetry & Plotting: Displays normalized (x, y, z) and pixel (x, y) coordinates in a live table alongside a real-time Matplotlib joint-angle trajectory chart.
**Tech Stack**
# ComponentLibrary / ToolPurpose
**Language:**`Python 3.10+Core logic, coordinate extraction, and state tracking`
**UI Dashboard:**`StreamlitInteractive sidebar controls, live feed, and metric readouts`
**Vision & Capture:**`OpenCV (cv2)Frame capture, BGR-to-RGB conversion, and angle text overlays`
**Pose Engine:**`MediaPipe Pose`
`Two-stage detector-tracker for 33 3D body landmarks`
**Math & Plotting:*`*NumPy, Matplotlib, PandasVector angle math, live time-series plots, and coordinate tables`

**Installation & Usage**
# Install dependencies:`pip install streamlit opencv-python mediapipe numpy matplotlib pandas
### Run the Streamlit application:streamlit run human_pose_estimation.py`
#### Select an input mode in the sidebar (Live Webcam, Upload Video, or Upload Image), adjust confidence/visibility thresholds if needed, and view the annotated skeleton, posture classification, and live joint-angle chart.

**Project Structure:**
```markdown
├── human_pose_estimation.py                 # Main Streamlit + OpenCV + MediaPipe app
├── Human_Pose_Estimation_Documentation.pdf  # Single-sheet technical documentation
└── README.md   
```                             
# Project setup and usage guide
**How It Works**
**Capture & Color Conversion:** `OpenCV reads each frame in BGR order and converts it to RGB (cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)).`**Landmark Inference:**` MediaPipe Pose processes the RGB frame and returns 33 normalized 3D landmarks with visibility scores.`**Geometry & State Tracking:** `Visible joints (> 0.5) are converted to pixel coordinates to compute elbow, shoulder, and knee angles, classify posture, and track movement reps across frames.`
**Live Display:** `The annotated RGB frame, coordinate table, and Matplotlib angle trajectory chart render directly in Streamlit.
Live Link:https://human-pose-estimation-cp9u8h43zx3jzy5kfiggng.streamlit.app/`
