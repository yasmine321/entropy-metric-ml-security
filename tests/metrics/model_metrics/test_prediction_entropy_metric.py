from a4s_eval.metric_registries.model_metric_registry import model_metric_registry


def test_prediction_entropy_is_registered():
    registry = dict(model_metric_registry)
    assert "prediction_entropy" in registry
