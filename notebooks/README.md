# Training recipe (Google Colab)

Create a new notebook at [colab.research.google.com](https://colab.research.google.com),
set **Runtime → Change runtime type → T4 GPU** (free tier), and run these
cells in order.

## 1. Install Ultralytics

```python
!pip install ultralytics -q
```

## 2. Download your annotated dataset from Roboflow

Get this exact snippet from your Roboflow project: **Dataset version →
Export Dataset → format YOLO11 (or YOLO26) → "show download code"**.
It will look like this (your key and project name will differ):

```python
!pip install roboflow -q

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_PRIVATE_API_KEY")
project = rf.workspace("your-workspace").project("your-project-name")
version = project.version(1)
dataset = version.download("yolov11")
```

## 3. Train

```python
from ultralytics import YOLO

# Start from a pretrained nano checkpoint (transfer learning = fewer images needed)
model = YOLO("yolo11n.pt")

results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=80,
    imgsz=640,
    batch=16,
    patience=15,       # early stopping if no improvement
    name="lever-detector",
)
```

## 4. Validate

```python
metrics = model.val()
print(metrics.box.map50)   # mAP@50 — aim for > 0.90 on a simple 2-class task
```

## 5. Download the trained weights

```python
from google.colab import files
files.download('runs/detect/lever-detector/weights/best.pt')
```

Save the downloaded file as `models/best.pt` in this repository, then run
`src/live_inference.py` locally.

## Tips for a small custom dataset

- 150–300 images per class is plenty for a 2-class nano model.
- Vary lighting, angle, and background between shots — this matters far
  more than raw image count.
- Use Roboflow's built-in augmentations (rotation, brightness, blur) when
  generating the dataset version to multiply your effective dataset size
  for free.
- If `mAP50` is low, check class balance and re-label any ambiguous/blurry
  frames before adding more epochs.
