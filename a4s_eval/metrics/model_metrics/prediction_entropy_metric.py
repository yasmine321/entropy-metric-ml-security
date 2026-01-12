from datetime import datetime

import numpy as np

from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.service.functional_model import TabularClassificationModel


@model_metric(name="prediction_entropy")
def prediction_entropy(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    functional_model: TabularClassificationModel,
) -> list[Measure]:
    df = dataset.data

    # Feature objects -> use .name and convert to NumPy array
    feature_names = [f.name for f in datashape.features]
    x_df = df[feature_names]
    x = x_df.to_numpy()

    # Predicted probabilities (n_samples x n_classes)
    proba = np.asarray(functional_model.predict_proba(x))

    # Numerical stability
    proba = np.clip(proba, 1e-12, 1.0)

    # Entropy per sample: H(p) = -sum_i p_i log p_i
    entropy_per_sample = -np.sum(proba * np.log(proba), axis=1)

    mean_entropy = float(entropy_per_sample.mean())
    max_entropy = float(entropy_per_sample.max())
    min_entropy = float(entropy_per_sample.min())

    now = datetime.now()

    return [
        Measure(name="prediction_entropy_mean", score=mean_entropy, time=now),
        Measure(name="prediction_entropy_max", score=max_entropy, time=now),
        Measure(name="prediction_entropy_min", score=min_entropy, time=now),
    ]
