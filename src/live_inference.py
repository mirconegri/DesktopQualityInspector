"""
live_inference.py

Real-time object detection for the "Desktop Quality Inspector" project.

Captures a live video feed from a webcam (including virtual webcams such as
DroidCam), runs a custom-trained YOLO model on every frame, and overlays
bounding boxes + class labels showing the predicted state of the inspected
object (e.g. "lever_open" / "lever_closed").

Usage:
    python src/live_inference.py
    python src/live_inference.py --model models/best.pt --source 1 --conf 0.6

Press 'q' in the video window to quit.
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from config import CLASS_COLORS, DEFAULT_CONFIDENCE, DEFAULT_MODEL_PATH, DEFAULT_SOURCE


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments so the script can be reused without editing the code."""
    parser = argparse.ArgumentParser(description="Live YOLO inference on a webcam feed.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained YOLO weights (.pt file).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(DEFAULT_SOURCE),
        help="Camera index (0, 1, 2...) or video file / stream URL.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Minimum confidence to draw a detection (0.0 - 1.0).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference resolution. Lower = faster but less accurate.",
    )
    return parser.parse_args()


def open_camera(source: str) -> cv2.VideoCapture:
    """
    Open a video source. Accepts either a numeric webcam index (e.g. "0", "1")
    or a path/URL (useful as a fallback for DroidCam's direct IP stream).
    """
    cam_source = int(source) if source.isdigit() else source

    # CAP_DSHOW avoids the common black-screen issue with virtual cameras
    # (like DroidCam) on Windows when using the default MSMF backend.
    if isinstance(cam_source, int):
        cap = cv2.VideoCapture(cam_source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cam_source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video source '{source}'. "
            "Check that DroidCam is connected and try a different index "
            "(0, 1, 2...). Run scripts/list_cameras.py to scan available cameras."
        )
    return cap


def draw_detections(frame, result, conf_threshold: float):
    """Draw bounding boxes + labels for every detection above the confidence threshold."""
    class_names = result.names  # dict {class_id: class_name}, comes from the trained model

    for box in result.boxes:
        confidence = float(box.conf[0])
        if confidence < conf_threshold:
            continue

        class_id = int(box.cls[0])
        label_name = class_names[class_id]

        # Pixel coordinates of the bounding box (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Pick a color based on the predicted class (default to white if not mapped)
        color = CLASS_COLORS.get(label_name, (255, 255, 255))

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Filled label background for readability, then the text on top
        label_text = f"{label_name} {confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(
            frame, label_text, (x1 + 2, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
        )

    return frame


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "Download your trained weights from the Colab notebook and place "
            "them in models/ (see models/README.md)."
        )

    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(str(model_path))

    print(f"[INFO] Opening video source: {args.source}")
    cap = open_camera(args.source)

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] No frame received, exiting loop.")
                break

            # Run inference on the current frame.
            # verbose=False keeps the terminal clean from per-frame logs.
            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            result = results[0]

            frame = draw_detections(frame, result, args.conf)

            # Compute and overlay FPS so you can judge real-time performance.
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if current_time != prev_time else 0.0
            prev_time = current_time
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )

            cv2.imshow("Desktop Quality Inspector - press 'q' to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Always release the camera and close windows, even if an error occurs.
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released, windows closed. Bye!")


if __name__ == "__main__":
    main()
