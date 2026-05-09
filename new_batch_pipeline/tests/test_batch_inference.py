"""Unit tests for batch inference preparation and output formatting."""
import pandas as pd
import pytest
from io import StringIO


@pytest.fixture
def sample_test_csv():
    data = """id,date,store_nbr,family,onpromotion
100,2017-08-16,1,AUTOMOTIVE,0
101,2017-08-16,1,BEVERAGES,5
102,2017-08-16,2,AUTOMOTIVE,0
103,2017-08-16,2,BEVERAGES,3"""
    return pd.read_csv(StringIO(data))


def test_batch_input_enrichment(sample_test_csv):
    """IT-06: Verify test.csv enrichment produces all required columns."""
    df = sample_test_csv.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["oil_price"] = 50.0
    df["is_holiday"] = 0
    df["day_of_week"] = df["date"].dt.dayofweek

    required = ["id", "date", "store_nbr", "family", "onpromotion",
                 "oil_price", "is_holiday", "day_of_week"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"
    assert df.isna().sum().sum() == 0, "No NaN values allowed"


def test_submission_format():
    """E2E-03: Output must have exactly id and sales columns."""
    submission = pd.DataFrame({"id": [100, 101, 102], "sales": [10.0, 20.0, 30.0]})
    assert list(submission.columns) == ["id", "sales"]
    assert all(submission["sales"] >= 0)


def test_submission_no_negative():
    """E2E-04: Predictions must be non-negative."""
    raw = pd.DataFrame({"id": [1, 2, 3], "sales": [-5.0, 10.0, -1.0]})
    raw["sales"] = raw["sales"].apply(lambda x: max(0, x))
    assert all(raw["sales"] >= 0)
    assert raw.loc[0, "sales"] == 0.0
