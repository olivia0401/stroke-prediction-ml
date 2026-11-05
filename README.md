# Stroke Prediction ML

A research model that classifies stroke outcomes from tabular patient data. The data is highly imbalanced. Five-fold cross-validation reports ROC-AUC 0.79 ± 0.02; at the reported operating point, F1 is 0.17 ± 0.05 and recall 0.21 ± 0.07. This is not a clinical tool.

## Run

```bash
pip install -r requirements.txt
python scripts/train.py --model xgb
```

The API and Gradio interface require a trained model; see `api/` and `app_ui.py`. Dataset details are in `data/README.md`.

## Tests

```bash
python -m pytest -q
```

Metrics are cross-validation estimates, not prospective clinical validation. See the training script for preprocessing and threshold selection.
