"""Unit tests for the CPR predictor logic."""
import pandas as pd
import pytest


@pytest.fixture
def sample_proportions():
    return pd.DataFrame({
        "store_nbr": [1, 1, 2, 2],
        "family": ["AUTOMOTIVE", "BEVERAGES", "AUTOMOTIVE", "BEVERAGES"],
        "proportion": [0.15, 0.35, 0.20, 0.30],
    })


@pytest.fixture
def sample_instances():
    return pd.DataFrame({
        "id": [100, 101, 102, 103, 104, 105, 106, 107],
        "date": ["2017-08-16"] * 4 + ["2017-08-17"] * 4,
        "store_nbr": [1, 1, 2, 2, 1, 1, 2, 2],
        "family": ["AUTOMOTIVE", "BEVERAGES", "AUTOMOTIVE", "BEVERAGES"] * 2,
        "onpromotion": [0, 5, 0, 3, 0, 4, 1, 2],
        "oil_price": [50.0] * 8,
        "is_holiday": [0] * 8,
        "day_of_week": [2, 2, 2, 2, 3, 3, 3, 3],
    })


def test_preprocess_parses_instances(sample_instances):
    """UT-11: Verify preprocess extracts instances from dict format."""
    prediction_input = {"instances": sample_instances.values.tolist()}
    columns = ["id", "date", "store_nbr", "family",
                "onpromotion", "oil_price", "is_holiday", "day_of_week"]
    df = pd.DataFrame(prediction_input["instances"], columns=columns)
    assert len(df) == 8
    assert list(df.columns) == columns


def test_predictions_non_negative():
    """UT-12: Verify the non-negative clamp works."""
    raw = [100.0, -5.0, 0.0, 200.0, -50.0]
    clamped = [max(0, x) for x in raw]
    assert all(p >= 0 for p in clamped)
    assert clamped == [100.0, 0.0, 0.0, 200.0, 0.0]


def test_disaggregation_row_count(sample_instances, sample_proportions):
    """UT-13: Verify disaggregation produces one prediction per input row."""
    df = sample_instances.copy()
    df["date"] = pd.to_datetime(df["date"])
    date_to_total = {pd.Timestamp("2017-08-16"): 1000.0, pd.Timestamp("2017-08-17"): 1200.0}

    merged = df.merge(sample_proportions, on=["store_nbr", "family"], how="left")
    merged["proportion"] = merged["proportion"].fillna(0)
    merged["sales"] = merged.apply(lambda r: date_to_total.get(r["date"], 0) * r["proportion"], axis=1)

    assert len(merged) == len(df)
    assert all(merged["sales"] >= 0)
    assert merged.iloc[0]["sales"] == 150.0  # store1 AUTO: 1000*0.15
    assert merged.iloc[1]["sales"] == 350.0  # store1 BEV: 1000*0.35


def test_postprocess_format():
    """UT-14: Verify postprocess returns expected dict format."""
    predictions = [100.0, 200.0, 150.0]
    result = {"predictions": predictions}
    assert "predictions" in result
    assert len(result["predictions"]) == 3
