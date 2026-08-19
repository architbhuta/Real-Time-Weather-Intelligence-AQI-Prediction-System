from models.evaluate import compute_metrics


def test_compute_metrics_returns_expected_keys_and_reasonable_values():
    y_true = [100, 150, 200]
    y_pred = [110, 140, 210]

    metrics = compute_metrics(y_true, y_pred)

    assert set(metrics.keys()) == {"mae", "rmse", "r2", "mape"}
    assert metrics["mae"] == 10.0
    assert metrics["rmse"] > 0
    assert metrics["mape"] is not None
    assert metrics["mape"] > 0


def test_compute_metrics_perfect_prediction_has_zero_error():
    y_true = [50, 60, 70]
    y_pred = [50, 60, 70]

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mape"] == 0.0


def test_compute_metrics_handles_all_zero_true_values():
    y_true = [0, 0, 0]
    y_pred = [1, 2, 3]

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["mape"] is None
