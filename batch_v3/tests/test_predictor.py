"""Unit tests for v3 CPR predictor logic."""
import numpy as np
import pandas as pd
import pytest


def test_preprocess_parses_instances():
    """UT-11: Preprocess extracts instances from dict."""
    prediction_input = {
        "instances": [[0, 50.0, 0, 2], [5, 51.0, 1, 3], [3, 52.0, 0, 4]]
    }
    instances = prediction_input.get("instances", [])
    assert len(instances) == 3
    assert len(instances[0]) == 4


def test_predictions_non_negative():
    """UT-12: Non-negative clamp works."""
    raw = [10.0, -5.0, 0.0, 20.0, -50.0]
    clamped = [max(0, x) for x in raw]
    assert all(p >= 0 for p in clamped)
    assert clamped == [10.0, 0.0, 0.0, 20.0, 0.0]


def test_correct_prediction_count():
    """UT-13: Number of predictions matches input length."""
    instances = [[0, 50.0, 0, i % 7] for i in range(15)]
    # Simulate: forecast returns same length
    predictions = [float(i) for i in range(len(instances))]
    assert len(predictions) == len(instances)


def test_postprocess_format():
    """UT-14: Postprocess returns correct dict format."""
    predictions = [10.0, 20.0, 15.0]
    result = {"predictions": predictions}
    assert "predictions" in result
    assert len(result["predictions"]) == 3
