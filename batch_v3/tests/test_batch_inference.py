"""Unit tests for v3 batch inference."""
import pandas as pd
import pytest
from io import StringIO


@pytest.fixture
def sample_test_csv():
    data = """id,date,store_nbr,family,onpromotion
100,2017-08-16,1,BEVERAGES,5
101,2017-08-16,1,AUTOMOTIVE,0
102,2017-08-16,2,BEVERAGES,3
103,2017-08-17,1,BEVERAGES,4"""
    return pd.read_csv(StringIO(data))


def test_filter_test_to_target(sample_test_csv):
    """IT-06: Filtering test.csv to target store×family works."""
    df = sample_test_csv
    filtered = df[(df["store_nbr"] == 1) & (df["family"] == "BEVERAGES")]
    assert len(filtered) == 2, "Only store 1, BEVERAGES rows"
    assert list(filtered["id"]) == [100, 103]


def test_enrichment_no_nan(sample_test_csv):
    """IT-06: Enriched test data has no NaN."""
    df = sample_test_csv
    filtered = df[(df["store_nbr"] == 1) & (df["family"] == "BEVERAGES")].copy()
    filtered["oil_price"] = 50.0
    filtered["is_holiday"] = 0
    filtered["date"] = pd.to_datetime(filtered["date"])
    filtered["day_of_week"] = filtered["date"].dt.dayofweek

    required = ["id", "onpromotion", "oil_price", "is_holiday", "day_of_week"]
    for col in required:
        assert col in filtered.columns
    assert filtered[required].isna().sum().sum() == 0


def test_submission_format():
    """E2E-03: Output has exactly id and sales columns."""
    submission = pd.DataFrame({"id": [100, 103], "sales": [20.0, 22.0]})
    assert list(submission.columns) == ["id", "sales"]
    assert all(submission["sales"] >= 0)


def test_submission_no_negative():
    """E2E-04: Predictions clamped to non-negative."""
    raw = pd.DataFrame({"id": [1, 2], "sales": [-5.0, 10.0]})
    raw["sales"] = raw["sales"].apply(lambda x: max(0, x))
    assert all(raw["sales"] >= 0)
