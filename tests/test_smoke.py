"""
test_smoke.py

Minimal smoke tests that don't require a trained model, a camera, or a GPU.
They just confirm the project's basic wiring is sound — useful as a cheap
CI check before pushing.

Run with:
    pytest tests/
"""

import sys
from pathlib import Path

# Make src/ importable when running pytest from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402


def test_config_has_required_keys():
    """Config must define the core settings live_inference.py depends on."""
    assert hasattr(config, "DEFAULT_MODEL_PATH")
    assert hasattr(config, "DEFAULT_SOURCE")
    assert hasattr(config, "DEFAULT_CONFIDENCE")
    assert hasattr(config, "CLASS_COLORS")


def test_confidence_threshold_is_valid():
    """Confidence threshold must be a sane probability value."""
    assert 0.0 <= config.DEFAULT_CONFIDENCE <= 1.0


def test_class_colors_are_bgr_tuples():
    """Every color must be a valid 3-channel BGR tuple for OpenCV."""
    for class_name, color in config.CLASS_COLORS.items():
        assert isinstance(class_name, str)
        assert len(color) == 3
        assert all(0 <= channel <= 255 for channel in color)


def test_ultralytics_and_opencv_importable():
    """Core dependencies must be installed and importable."""
    import cv2  # noqa: F401
    from ultralytics import YOLO  # noqa: F401
