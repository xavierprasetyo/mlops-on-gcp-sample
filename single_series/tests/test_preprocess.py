"""Unit tests for the v3 preprocessing component logic."""
import pandas as pd
import pytest
from io import StringIO


@pytest.fixture
def sample_train_csv():
    """Train CSV with 2 stores, 2 families, 3 dates."""
    data = """id,date,store_nbr,family,sales,onpromotion
0,2017-08-01,1,BEVERAGES,20.0,5
1,2017-08-01,1,AUTOMOTIVE,10.0,0
2,2017-08-01,2,BEVERAGES,25.0,3
3,2017-08-02,1,BEVERAGES,22.0,4
4,2017-08-02,1,AUTOMOTIVE,12.0,0
5,2017-08-02,2,BEVERAGES,28.0,2
6,2017-08-03,1,BEVERAGES,21.0,6
7,2017-08-03,1,AUTOMOTIVE,11.0,0
8,2017-08-03,2,BEVERAGES,26.0,4"""
    return pd.read_csv(StringIO(data))


@pytest.fixture
def sample_oil_csv():
    data = """date,dcoilwtico
2017-08-01,50.0
2017-08-02,
2017-08-03,52.0"""
    return pd.read_csv(StringIO(data))


@pytest.fixture
def sample_holidays_csv():
    data = """date,type,locale,locale_name,description,transferred
2017-08-01,Holiday,National,Ecuador,Test Holiday,False
2017-08-02,Holiday,National,Ecuador,Transferred Holiday,True
2017-08-03,Work Day,National,Ecuador,Make-up Day,False"""
    return pd.read_csv(StringIO(data))


def test_filter_to_target(sample_train_csv):
    """UT-01: Filter to target store×family retains only matching rows."""
    df = sample_train_csv
    filtered = df[(df["store_nbr"] == 1) & (df["family"] == "BEVERAGES")]
    assert len(filtered) == 3, "Should have 3 rows for store 1, BEVERAGES"
    assert (filtered["store_nbr"] == 1).all()
    assert (filtered["family"] == "BEVERAGES").all()


def test_oil_price_ffill(sample_oil_csv):
    """UT-02: Oil price NaN is forward-filled."""
    oil_df = sample_oil_csv.copy()
    oil_df["dcoilwtico"] = oil_df["dcoilwtico"].ffill().bfill()
    assert oil_df["dcoilwtico"].isna().sum() == 0
    assert oil_df.loc[1, "dcoilwtico"] == 50.0


def test_holiday_transfer_exclusion(sample_holidays_csv):
    """UT-03: Transferred holidays and Work Days are excluded."""
    hol_df = sample_holidays_csv
    valid = hol_df[(hol_df["transferred"] == False) & (hol_df["type"] != "Work Day")]
    assert len(valid) == 1
    assert valid.iloc[0]["date"] == "2017-08-01"


def test_train_val_split(sample_train_csv):
    """UT-04: Train/val split works on filtered data."""
    df = sample_train_csv
    filtered = df[(df["store_nbr"] == 1) & (df["family"] == "BEVERAGES")].copy()
    filtered["date"] = pd.to_datetime(filtered["date"])
    filtered = filtered.sort_values("date")

    cutoff = filtered["date"].max() - pd.Timedelta(days=1)
    train = filtered[filtered["date"] <= cutoff]
    val = filtered[filtered["date"] > cutoff]

    assert len(val) == 1
    assert val.iloc[0]["date"] == pd.Timestamp("2017-08-03")
    assert len(train) == 2


def test_filtered_is_daily_series(sample_train_csv):
    """UT-05: Filtered result has one row per date."""
    df = sample_train_csv
    filtered = df[(df["store_nbr"] == 1) & (df["family"] == "BEVERAGES")]
    assert filtered["date"].nunique() == len(filtered), "One row per date"
