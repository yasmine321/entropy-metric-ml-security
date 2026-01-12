import uuid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from a4s_eval.metric_registries.model_metric_registry import model_metric_registry
from a4s_eval.service.functional_model import TabularClassificationModel
from a4s_eval.service.model_factory import load_model
from a4s_eval.data_model.evaluation import (
    Dataset,
    DataShape,
    Model,
    ModelConfig,
    ModelFramework,
    ModelTask,
)

DATE_FEATURE = "issue_d"
N_SAMPLES: int | None = 1000


def sample(df: pd.DataFrame) -> pd.DataFrame:
    if N_SAMPLES:
        out: pd.DataFrame = df.iloc[:N_SAMPLES]
        return out
    return df


def get_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = pd.to_datetime(df[DATE_FEATURE])
    i_train = np.where(
        (pd.to_datetime("2013-01-01") <= t) & (t <= pd.to_datetime("2015-12-31"))
    )[0]
    i_test = np.where(
        (pd.to_datetime("2016-01-01") <= t) & (t <= pd.to_datetime("2017-12-31"))
    )[0]
    out: tuple[pd.DataFrame, pd.DataFrame] = df.iloc[i_train], df.iloc[i_test]
    return out


def build_data_shape() -> DataShape:
    metadata = pd.read_csv("tests/data/lcld_v2_metadata_api.csv").to_dict(
        orient="records"
    )

    for record in metadata:
        record["pid"] = uuid.uuid4()

    data_shape = {
        "features": [
            item
            for item in metadata
            if item.get("name") not in ["charged_off", "issue_d"]
        ],
        "target": next(rec for rec in metadata if rec.get("name") == "charged_off"),
        "date": next(rec for rec in metadata if rec.get("name") == "issue_d"),
    }

    return DataShape.model_validate(data_shape)


def build_datasets(data_shape: DataShape) -> tuple[Dataset, Dataset]:
    # Same dataset and split logic as tests/conftest.py
    full_df = pd.read_csv("./tests/data/lcld_v2.csv")
    train_df, test_df = get_splits(full_df)

    train_df = sample(train_df)
    test_df = sample(test_df)

    train_df["issue_d"] = pd.to_datetime(train_df["issue_d"])
    test_df["issue_d"] = pd.to_datetime(test_df["issue_d"])

    ref_dataset = Dataset(
        pid=uuid.uuid4(),
        shape=data_shape,
        data=train_df,
    )

    test_dataset = Dataset(
        pid=uuid.uuid4(),
        shape=data_shape,
        data=test_df,
    )

    return ref_dataset, test_dataset


def build_ref_model(ref_dataset: Dataset) -> Model:
    return Model(
        pid=uuid.uuid4(),
        model=None,
        dataset=ref_dataset,
    )


def build_functional_model() -> TabularClassificationModel:
    model_config = ModelConfig(
        path="./tests/data/lcld_v2_tabtransformer.pt",
        framework=ModelFramework.TORCH,
        task=ModelTask.CLASSIFICATION,
    )
    model = load_model(model_config)
    if not isinstance(model, TabularClassificationModel):
        raise TypeError("Loaded model is not TabularClassificationModel")
    return model


def main():
    # Recreate evaluation context from tests
    data_shape = build_data_shape()
    ref_dataset, test_dataset = build_datasets(data_shape)
    ref_model = build_ref_model(ref_dataset)
    functional_model = build_functional_model()

    registry = dict(model_metric_registry)
    entropy_fn = registry["prediction_entropy"]

    # 1) Run the metric (aggregate measures)
    measures = entropy_fn(data_shape, ref_model, test_dataset, functional_model)
    print("Aggregate prediction entropy measures:")
    for m in measures:
        print(f"{m.name}: {m.score:.6f}")

    # 2) Per-sample entropy for visualization
    df = test_dataset.data
    feature_names = [f.name for f in data_shape.features]
    x_df = df[feature_names]
    x = x_df.to_numpy()

    proba = np.asarray(functional_model.predict_proba(x))
    proba = np.clip(proba, 1e-12, 1.0)
    entropy_per_sample = -np.sum(proba * np.log(proba), axis=1)

    # 3) Histogram of entropy
    plt.figure(figsize=(6, 4))
    plt.hist(entropy_per_sample, bins=30, alpha=0.7)
    plt.xlabel("Prediction entropy")
    plt.ylabel("Count")
    plt.title("Distribution of prediction entropy")
    plt.tight_layout()
    plt.savefig("experiments/prediction_entropy_hist.png")
    plt.close()

    # 4) Entropy vs max predicted probability
    max_proba = proba.max(axis=1)
    plt.figure(figsize=(6, 4))
    plt.scatter(max_proba, entropy_per_sample, s=5, alpha=0.4)
    plt.xlabel("Max predicted class probability")
    plt.ylabel("Prediction entropy")
    plt.title("Entropy vs. max predicted probability")
    plt.tight_layout()
    plt.savefig("experiments/entropy_vs_max_proba.png")
    plt.close()


if __name__ == "__main__":
    main()
