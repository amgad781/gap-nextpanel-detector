# Gap vs Next-Panel Classifier 🧩

A computer vision model that classifies a robot-mounted camera's view as either a **large air gap** (unsafe to cross) or a **small crossable gap to the next solar panel** (safe to cross). Built as part of a larger autonomous solar panel-cleaning robot system, where this model is used to make real-time crossing decisions at panel boundaries.

🔗 **Live demo:** [Streamlit App](#) <!-- replace with your actual Streamlit Cloud URL once deployed -->

---

## Project Context

This model is one component of a multi-panel solar cleaning robot pipeline, which also includes:

- **Localization** — line-counting + Extended Kalman Filter, based on Kim et al. (2024), *"Localization of solar panel cleaning robot combining vision processing and extended Kalman filter,"* Science Progress 107(2).
- **Dust/damage classification** — a separate YOLOv8n-cls model that flags dirty or damaged panel surfaces.
- **Gap vs next-panel classification** (this repo) — decides whether the robot can safely cross to the adjacent panel or must stop/reroute.

Together these feed into the robot's coverage path planner and motor control system, enabling autonomous navigation across a multi-panel solar array.

---

## ⚠️ Safety Note

This is a **safety-critical classifier**. A false negative (predicting "next-panel" when the true situation is a dangerous gap) could cause the robot to fall. The deployed inference logic therefore:

- Requires a **high confidence threshold (≥ 0.85)** before allowing a crossing
- **Fails safe** — any low-confidence or ambiguous prediction defaults to "stop, do not cross"
- Is intended to run as a **secondary check** at detected panel boundaries, not as the sole edge-safety mechanism (a physical distance/ToF sensor is recommended as the primary safety trigger on the real robot)

---

## Model

- **Architecture:** YOLOv8n-cls (Ultralytics), pretrained on ImageNet, fine-tuned via transfer learning
- **Task:** Binary image classification (`gap` vs `next-panel`)
- **Input size:** 224×224
- **Export formats:**
  - `best.pt` — PyTorch weights, used for this Streamlit demo (x86/cloud)
  - NCNN export — used for on-device inference on Raspberry Pi (ARM)

---

## Dataset

Custom dataset collected and labeled for this project:
📊 [Kaggle: gap-nextpanel](https://www.kaggle.com/datasets/ahmedamgad101/gap-nextpanel)

| Split | Purpose |
|---|---|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

Class balance was checked prior to training; see the training notebook for details.

---

## Repository Structure

```
.
├── app.py              # Streamlit app for interactive testing
├── best.pt              # Trained YOLOv8n-cls weights
├── requirements.txt      # Python dependencies
├── packages.txt          # System-level dependencies (for Streamlit Cloud / Debian-based deploys)
└── README.md
```

---

## Running Locally

```bash
git clone https://github.com/<your-username>/gap-nextpanel-detector.git
cd gap-nextpanel-detector

python3 -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows

pip install -r requirements.txt

streamlit run app.py
```

The app will open at `http://localhost:8501`. Upload an image to test.

---

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (already done if you're reading this from the repo).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repo, branch, and `app.py` as the entry point.
4. Deploy.

**Note:** if you hit a `cv2` / `libGL` import error during deployment, confirm `packages.txt` contains `libgl1` and `libglib2.0-0`, and that `requirements.txt` uses `opencv-python-headless` rather than the default `opencv-python`.

---

## Deploying to Raspberry Pi (production / on-robot inference)

The cloud demo uses `best.pt`, but on-device inference on the Pi uses the NCNN export for speed on ARM CPUs.

```bash
# On the Pi, inside a virtual environment
pip install ultralytics

# Point YOLO at the exported NCNN model folder (not a single file)
```

```python
from ultralytics import YOLO

model = YOLO("/path/to/best_ncnn_model")
results = model(frame, imgsz=224, verbose=False)
```

See the training notebook for the export step (`model.export(format="ncnn")`).

---

## Training

The full training pipeline (data splitting, class balance check, augmentation, training, evaluation, export) is available as a Kaggle notebook:

📓 `gap_nextpanel_classification.ipynb`

Key training details:
- Base model: `yolov8n-cls.pt`
- Epochs: 60 (early stopping, patience=15)
- Image size: 224×224
- Augmentation: mild rotation/scale/shear, horizontal flip only (vertical flip disabled — gap orientation is meaningful), HSV jitter for lighting robustness

---

## Evaluation

Refer to the training notebook's confusion matrix output. When reviewing results, prioritize checking **false negatives on the `gap` class** over overall accuracy — this is the failure mode with real safety consequences.

---

## Author

Ahmed Amgad
Communication and Information Engineering, Zewail City University of Science and Technology

Part of the solar panel cleaning robot vision integration project (NANENG × CIE collaboration).
