# Dataset

The actual image dataset and annotations are **not stored in this repository**
(images are large and binary — bad fit for git). They are hosted and versioned
on Roboflow instead.

- **Roboflow project**: `<link to your Roboflow project here>`
- **Classes**: `lever_open`, `lever_closed`
- **Annotation format used for training**: YOLO (PyTorch TXT)

## Reproducing the dataset

1. Open the Roboflow project link above.
2. Go to your dataset version → "Export Dataset".
3. Choose format **YOLO11** (or **YOLO26**) and select "show download code".
4. Paste the generated snippet into the first cell of
   `notebooks/README.md`'s Colab recipe — it downloads the exact dataset
   version used to train `models/best.pt`.

## Local folders (gitignored)

- `data/raw/` — original unlabeled photos captured during data collection.
- `data/processed/` — Roboflow export, used only locally during training.

These folders exist locally while you work but are intentionally excluded
from version control via `.gitignore`.
