# Model weights

Trained weights are **not committed to git** (binary files, often 5-50MB+,
and trivially regenerable from `notebooks/README.md`).

## Download

After running the training notebook on Google Colab, download the result:

```python
# Last cell of the Colab notebook
from google.colab import files
files.download('runs/detect/train/weights/best.pt')
```

Place the downloaded file here as:

```
models/best.pt
```

Then run:

```bash
python src/live_inference.py --model models/best.pt
```

## Current model

| Property | Value |
|---|---|
| Base architecture | YOLO11n (or YOLO26n) |
| Classes | `lever_open`, `lever_closed` |
| Training images | `<fill in once trained>` |
| mAP50 (validation) | `<fill in once trained>` |
| Trained on | Google Colab, free T4 GPU |
