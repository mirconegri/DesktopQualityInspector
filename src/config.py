"""
config.py

Centralized configuration for the Desktop Quality Inspector project.
Keeping these values here (instead of hard-coding them in live_inference.py)
makes the script reusable for any 2-class detection task, not just the lever.
"""

# Path to the YOLO weights file produced by the Colab training notebook.
# Update this once you download your trained best.pt into models/.
DEFAULT_MODEL_PATH = "models/best.pt"

# Webcam index. 0 is usually the laptop's built-in camera; a DroidCam
# virtual camera typically shows up as 1, 2, or 3. Run scripts/list_cameras.py
# to find the correct index for your setup.
DEFAULT_SOURCE = 0

# Minimum confidence score (0.0 - 1.0) for a detection to be drawn on screen.
DEFAULT_CONFIDENCE = 0.5

# BGR colors (OpenCV uses BGR, not RGB!) mapped to your class names.
# IMPORTANT: these keys must match EXACTLY the class names you defined
# in Roboflow when annotating the dataset.
CLASS_COLORS = {
    "lever_open": (0, 200, 0),     # green
    "lever_closed": (0, 0, 220),   # red
}
