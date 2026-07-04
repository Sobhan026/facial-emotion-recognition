# User Guide

This guide explains how to run the Facial Emotion Recognition project from setup to webcam demo.

---

## 1. Requirements

Recommended environment:

| Item | Version / Notes |
|---|---|
| Python | 3.12 |
| OS | Windows tested |
| Framework | TensorFlow / Keras |
| Computer vision | OpenCV |
| Dataset | FER2013 CSV |

---

## 2. Create Virtual Environment

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## 3. Add Dataset

Place the FER2013 CSV here:

```text
data/raw/fer2013.csv
```

The file should contain:

```text
emotion,pixels,Usage
```

---

## 4. Process Dataset

Run:

```powershell
python -m scripts.save_processed_data
```

Expected outputs:

```text
data/processed/fer2013_processed.npz
data/processed/class_weights.json
data/processed/processed_summary.csv
```

---

## 5. Train Models

### Baseline CNN

```powershell
python -m scripts.train_baseline --epochs 25 --batch-size 64
```

### Improved CNN with Class Weights

```powershell
python -m scripts.train_improved --epochs 25 --batch-size 64
```

### Improved CNN without Class Weights

```powershell
python -m scripts.train_improved_no_class_weights --epochs 25 --batch-size 64
```

---

## 6. Evaluate Models

```powershell
python -m scripts.evaluate_baseline
python -m scripts.evaluate_improved
python -m scripts.evaluate_improved_no_class_weights
```

---

## 7. Compare Models

```powershell
python -m scripts.compare_model_results
```

Expected outputs:

```text
outputs/metrics/model_comparison.csv
outputs/metrics/model_comparison.json
outputs/figures/model_comparison_metrics.png
```

---

## 8. Run Threshold Analysis

```powershell
python -m scripts.threshold_analysis
```

Expected outputs:

```text
outputs/metrics/threshold_analysis_validation.csv
outputs/metrics/threshold_analysis_validation.json
outputs/metrics/recommended_threshold.json
outputs/figures/threshold_analysis_validation.png
```

Recommended threshold:

```text
0.55
```

---

## 9. Generate Demo Predictions

```powershell
python -m scripts.demo_test_predictions
```

Expected outputs:

```text
outputs/figures/threshold_demo_test_predictions.png
outputs/demo/threshold_demo_test_predictions.png
```

---

## 10. Run Webcam Demo

```powershell
python -m scripts.webcam_demo
```

Controls:

| Key | Action |
|---|---|
| `q` | Quit |
| `ESC` | Quit |
| `s` | Save snapshot |

Snapshots are saved here:

```text
outputs/demo/
```

---

## 11. Run Notebooks

Open and run:

```text
notebooks/01_data_exploration.ipynb
notebooks/02_baseline_cnn_results.ipynb
notebooks/03_improved_cnn_comparison.ipynb
notebooks/04_threshold_analysis.ipynb
```

Run order:

1. Data exploration
2. Baseline results
3. Improved model comparison
4. Threshold analysis

---

## 12. Run Tests

```powershell
python -m pytest
```

Expected current result:

```text
56 passed
```

---

## 13. Common Issues

### TensorFlow GPU Warning on Windows

You may see:

```text
TensorFlow GPU support is not available on native Windows
```

This is expected for TensorFlow versions newer than 2.10 on native Windows. The project still runs on CPU.

### oneDNN Warning

You may see:

```text
oneDNN custom operations are on
```

This is normal. It only means TensorFlow may use optimized CPU operations.

### Webcam Does Not Open

If camera index `0` does not work, edit:

```python
run_webcam_demo(camera_index=0)
```

to:

```python
run_webcam_demo(camera_index=1)
```

Then run again.

### Webcam Detects No Face

Try:

- Better lighting
- Frontal face position
- Less background clutter
- Moving closer to the camera

---

## 14. Recommended Demo Flow

For presentation:

1. Show dataset samples.
2. Show baseline model result.
3. Show improved model comparison.
4. Explain why the selected model was chosen.
5. Show threshold analysis.
6. Explain `Uncertain`.
7. Run the webcam demo.
8. Show tests passing.

Recommended command before presentation:

```powershell
python -m pytest
```

Then:

```powershell
python -m scripts.webcam_demo
```
