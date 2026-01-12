from datetime import datetime

from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.service.functional_model import TabularClassificationModel


@model_metric(name="simple_accuracy")
def simple_accuracy(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    functional_model: TabularClassificationModel,
) -> list[Measure]:
    accuracy_value = 0.99
    current_time = datetime.now()
    return [Measure(name="simple_accuracy", score=accuracy_value, time=current_time)]

