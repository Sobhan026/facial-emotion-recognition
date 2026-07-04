# Submission Checklist

Use this checklist before submitting or presenting the project.

---

## 1. Dataset

| Item | Status |
|---|---|
| `data/raw/fer2013.csv` exists |  |
| Official FER2013 splits preserved |  |
| Processed dataset generated |  |
| Class weights file generated |  |
| Processed summary generated |  |

Expected files:

```text
data/processed/fer2013_processed.npz
data/processed/class_weights.json
data/processed/processed_summary.csv
```

---

## 2. Models

| Model File | Status |
|---|---|
| `models/baseline_cnn_best.keras` |  |
| `models/baseline_cnn_final.keras` |  |
| `models/improved_cnn_best.keras` |  |
| `models/improved_cnn_final.keras` |  |
| `models/improved_cnn_no_class_weights_best.keras` |  |
| `models/improved_cnn_no_class_weights_final.keras` |  |

---

## 3. Metrics

| File | Status |
|---|---|
| `outputs/metrics/baseline_cnn_test_metrics.json` |  |
| `outputs/metrics/improved_cnn_test_metrics.json` |  |
| `outputs/metrics/improved_cnn_no_class_weights_test_metrics.json` |  |
| `outputs/metrics/model_comparison.csv` |  |
| `outputs/metrics/threshold_analysis_validation.csv` |  |
| `outputs/metrics/recommended_threshold.json` |  |

---

## 4. Figures

| Figure | Status |
|---|---|
| `outputs/figures/class_distribution.png` |  |
| `outputs/figures/emotion_samples.png` |  |
| `outputs/figures/baseline_cnn_accuracy_curve.png` |  |
| `outputs/figures/baseline_cnn_loss_curve.png` |  |
| `outputs/figures/baseline_cnn_confusion_matrix.png` |  |
| `outputs/figures/improved_cnn_confusion_matrix.png` |  |
| `outputs/figures/improved_cnn_no_class_weights_confusion_matrix.png` |  |
| `outputs/figures/model_comparison_metrics.png` |  |
| `outputs/figures/threshold_analysis_validation.png` |  |
| `outputs/figures/threshold_demo_test_predictions.png` |  |

---

## 5. Notebooks

| Notebook | Status |
|---|---|
| `notebooks/01_data_exploration.ipynb` runs without error |  |
| `notebooks/02_baseline_cnn_results.ipynb` runs without error |  |
| `notebooks/03_improved_cnn_comparison.ipynb` runs without error |  |
| `notebooks/04_threshold_analysis.ipynb` runs without error |  |

---

## 6. Tests

Run:

```powershell
python -m pytest
```

Expected:

```text
56 passed
```

| Test Group | Status |
|---|---|
| Preprocessing tests pass |  |
| Processed loader tests pass |  |
| Model tests pass |  |
| Thresholding tests pass |  |
| Inference tests pass |  |
| Face detection tests pass |  |
| Temporal smoothing tests pass |  |

---

## 7. Demo

| Demo | Status |
|---|---|
| `python -m scripts.demo_test_predictions` runs |  |
| `python -m scripts.webcam_demo` opens webcam |  |
| Webcam shows face box |  |
| Webcam shows emotion / `Uncertain` |  |
| Webcam shows top-3 predictions |  |
| Pressing `s` saves snapshot |  |
| Pressing `q` exits cleanly |  |

---

## 8. Documentation

| Document | Status |
|---|---|
| `README.md` completed |  |
| `docs/TECHNICAL_REPORT.md` completed |  |
| `docs/MODEL_CARD.md` completed |  |
| `docs/USER_GUIDE.md` completed |  |
| `docs/SUBMISSION_CHECKLIST.md` completed |  |

---

## 9. Final Numbers to Mention

| Metric | Value |
|---|---:|
| Best test accuracy | 0.5977 |
| Best weighted F1 | 0.5811 |
| Best macro F1 | 0.5304 |
| Selected threshold | 0.55 |
| Accepted accuracy at threshold | 0.8147 |
| Coverage at threshold | 0.4316 |
| Rejection rate at threshold | 0.5684 |
| Tests passing | 56 |

---

## 10. Presentation Talking Points

1. The project is a full ML pipeline, not only a trained model.
2. FER2013 was processed using official splits.
3. A baseline model was built first.
4. An improved CNN was tested with and without class weights.
5. Class weights did not improve global performance.
6. The selected model achieved the best accuracy and weighted F1.
7. Confidence thresholding improves reliability.
8. `Uncertain` prevents forced low-confidence predictions.
9. Webcam inference includes face detection and temporal smoothing.
10. The project has automated tests for core modules.
