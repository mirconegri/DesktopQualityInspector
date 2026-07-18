"""
list_cameras.py

Quick utility to discover which camera index corresponds to which device
on Windows. Useful when you have both the Surface's built-in webcam AND
DroidCam's virtual camera available — they rarely use index 0 and 1 in the
order you'd expect.

Usage:
    python scripts/list_cameras.py

For each index found, a window opens showing the live feed and the index
number. Press any key to move to the next index, or 'q' to stop scanning.
"""

import cv2

MAX_INDEX_TO_TRY = 6


def main() -> None:
    found_any = False

    for index in range(MAX_INDEX_TO_TRY):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            cap.release()
            continue

        ret, frame = cap.read()
        if not ret:
            cap.release()
            continue

        found_any = True
        print(f"[FOUND] Camera index {index} is working.")

        # Overlay the index on the frame so it's obvious which one this is.
        cv2.putText(
            frame, f"INDEX {index} - press any key for next, 'q' to stop",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        cv2.imshow("Camera scan", frame)
        key = cv2.waitKey(0) & 0xFF

        cap.release()
        cv2.destroyAllWindows()

        if key == ord("q"):
            break

    if not found_any:
        print("[INFO] No camera found. Is DroidCam connected and running?")


if __name__ == "__main__":
    main()
