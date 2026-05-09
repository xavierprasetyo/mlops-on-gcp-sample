"""Unit tests for the preprocessing component logic."""
import pandas as pd
import pytest
from io import StringIO


# ============================================================================
# Test Fixtures
# ============================================================================
@pytest.fixture
def sample_train_csv():
    """Small synthetic train.csv with 2 stores, 2 families, 3 dates."""
    data = """id,date,store_nbr,family,sales,onpromotion
0,2017-08-01,1,AUTOMOTIVE,10.0,0
1,2017-08-01,1,BEVERAGES,20.0,5
2,2017-08-01,2,AUTOMOTIVE,15.0,0
3,2017-08-01,2,BEVERAGES,25.0,3
4,2017-08-02,1,AUTOMOTIVE,12.0,0
5,2017-08-02,1,BEVERAGES,22.0,4
6,2017-08-02,2,AUTOMOTIVE,18.0,1
7,2017-08-02,2,BEVERAGES,28.0,2
8,2017-08-03,1,AUTOMOTIVE,11.0,0
9,2017-08-03,1,BEVERAGES,21.0,6
10,2017-08-03,2,AUTOMOTIVE,16.0,0
11,2017-08-03,2,BEVERAGES,26.0,4"""
    return pd.read_csv(StringIO(data))


@pytest.fixture
def sample_oil_csv():
    """Oil CSV with a NaN value to test forward-fill."""
    data = """date,dcoilwtico
2017-08-01,50.0
2017-08-02,
2017-08-03,52.0"""
    return pd.read_csv(StringIO(data))


@pytest.fixture
def sample_holidays_csv():
    """Holidays CSV with transferred and Work Day entries to test filtering."""
    data = """date,type,locale,locale_name,description,transferred
2017-08-01,Holiday,National,Ecuador,Test Holiday,False
2017-08-02,Holiday,National,Ecuador,Transferred Holiday,True
2017-08-03,Work Day,National,Ecuador,Make-up Day,False"""
    return pd.read_csv(StringIO(data))


# ============================================================================
# UT-01: Daily aggregation
# ============================================================================
def test_daily_aggregation(sample_train_csv):
    """Verify daily aggregation produces correct totals."""
    df = sample_train_csv
    daily = df.groupby("date").agg({"sales": "sum", "onpromotion": "sum"}).reset_index()

    assert len(daily) == 3, "Should have 1 row per date"
    # Day 1: 10+20+15+25 = 70
    assert daily.loc[daily["date"] == "2017-08-01", "sales"].values[0] == 70.0
    # Day 2: 12+22+18+28 = 80
    assert daily.loc[daily["date"] == "2017-08-02", "sales"].values[0] == 80.0


# ============================================================================
# UT-02: Oil price forward-fill
# ============================================================================
def test_oil_price_ffill(sample_oil_csv):
    """Verify oil price NaN is forward-filled."""
    oil_df = sample_oil_csv.copy()
    oil_df["dcoilwtico"] = oil_df["dcoilwtico"].ffill().bfill()

    assert oil_df["dcoilwtico"].isna().sum() == 0, "No NaN after ffill"
    assert oil_df.loc[1, "dcoilwtico"] == 50.0, "NaN filled with previous value"


# ============================================================================
# UT-03: Holiday transfer exclusion
# ============================================================================
def test_holiday_transfer_exclusion(sample_holidays_csv):
    """Verify transferred holidays and Work Days are excluded."""
    hol_df = sample_holidays_csv
    valid = hol_df[
        (hol_df["transferred"] == False) & (hol_df["type"] != "Work Day")
    ]

    assert len(valid) == 1, "Only 1 valid holiday (non-transferred, non-Work Day)"
    assert valid.iloc[0]["date"] == "2017-08-01"


# ============================================================================
# UT-04: Proportions sum to ~1.0
# ============================================================================
def test_proportions_sum(sample_train_csv):
    """Verify store×family proportions sum to approximately 1.0."""
    df = sample_train_csv
    total_sales = df["sales"].sum()
    props = (
        df.groupby(["store_nbr", "family"])["sales"].sum() / total_sales
    ).reset_index(name="proportion")

    assert abs(props["proportion"].sum() - 1.0) < 1e-6, "Proportions must sum to 1.0"
    assert len(props) == 4, "Should have 4 store×family combos"


# ============================================================================
# UT-05: Train/val split
# ============================================================================
def test_train_val_split(sample_train_csv):
    """Verify train/val split respects forecast horizon."""
    df = sample_train_csv
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date").agg({"sales": "sum"}).reset_index()
    daily = daily.sort_values("date")

    forecast_horizon = 1
    cutoff = daily["date"].max() - pd.Timedelta(days=forecast_horizon)
    train = daily[daily["date"] <= cutoff]
    val = daily[daily["date"] > cutoff]

    assert len(val) == 1, "Val should have 1 day"
    assert val.iloc[0]["date"] == pd.Timestamp("2017-08-03")
    assert len(train) == 2, "Train should have 2 days"
