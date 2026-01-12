# Prediction Entropy Metric - AI Security Project

## Overview

This project implements a **prediction entropy** metric for tabular classification models in the A4S evaluation framework. The metric measures the uncertainty of model predictions based on the entropy of the predicted class probabilities on the internal course dataset `lcld_v2` (loan default data) and a pretrained TabTransformer model.

The implementation is integrated into the model metric registry and tested with pytest. An experiment script runs the metric on the real dataset and produces uncertainty distribution plots.

## Metric Definition

Given model predicted probabilities \( p = (p_1, \dots, p_K) \) for a sample, the prediction entropy is:

\[
H(p) = - \sum_{i=1}^K p_i \log p_i
\]

**Interpretation:**
- Higher entropy → model is less confident (more uncertain)
- Lower entropy → model is highly confident in its prediction
- Entropy ranges from 0 (perfect confidence) to ln(K) (uniform distribution)

The metric returns three aggregate measures computed over the test dataset:

- `prediction_entropy_mean`: average entropy across all test samples
- `prediction_entropy_max`: maximum entropy (least confident prediction)
- `prediction_entropy_min`: minimum entropy (most confident prediction)

## Project Files

### Core Implementation
- `a4s_eval/metrics/model_metrics/prediction_entropy_metric.py` - metric implementation
- `tests/metrics/model_metrics/test_prediction_entropy_metric.py` - unit tests

### Experiments
- `experiments/run_prediction_entropy_experiment.py` - experiment script
- `experiments/prediction_entropy_hist.png` - histogram of entropy distribution
- `experiments/entropy_vs_max_proba.png` - scatter plot: entropy vs confidence

### Documentation
- `README_project.md` - this file
- `entropy_experiments.ipynb` - Jupyter notebook with reproducible analysis

## Dataset and Model

**Dataset:** `tests/data/lcld_v2.csv`
- Internal course dataset for loan default prediction
- Features: 28 numeric/categorical features related to loan characteristics
- Target: `charged_off` (binary: 0=repaid, 1=defaulted)
- Date column: `issue_d` (loan origination date)
- Train period: 2013-2015, Test period: 2016-2017

**Metadata:** `tests/data/lcld_v2_metadata_api.csv`
- Used to build DataShape (feature, target, and date schema)
- Defines feature types and names

**Model:** `tests/data/lcld_v2_tabtransformer.pt`
- Pretrained TabTransformer model
- Framework: PyTorch
- Task: Binary classification
- Loaded via A4S ModelFactory as TabularClassificationModel

## Setup & Installation

From repository root:

```bash
uv sync
uv add evidently
