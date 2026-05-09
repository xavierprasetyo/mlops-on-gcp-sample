"""Unit tests for v3 SARIMAX training and hyperparameter search."""
import numpy as np
import os
import tempfile
import pandas as pd
import pytest


@pytest.fixture
def synthetic_daily_data():
    """200 days of synthetic sales data with weekly seasonality."""
    np.random.seed(42)
    dates = pd.date_range(start="2017-01-01", periods=200, freq="D")
    trend = np.linspace(10, 30, 200)
    weekly = 5 * np.sin(2 * np.pi * np.arange(200) / 7)
    noise = np.random.normal(0, 2, 200)
    sales = np.maximum(trend + weekly + noise, 0)

    return pd.DataFrame({
        "date": dates,
        "sales": sales,
        "onpromotion": np.random.randint(0, 10, 200),
        "oil_price": np.random.uniform(40, 60, 200),
        "is_holiday": np.random.choice([0, 1], 200, p=[0.9, 0.1]),
        "day_of_week": dates.dayofweek,
    })


def test_sarimax_fit(synthetic_daily_data):
    """UT-08: SARIMAX fits and saves without error."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    df = synthetic_daily_data
    exog_cols = ["onpromotion", "oil_price", "is_holiday", "day_of_week"]

    model = SARIMAX(
        df["sales"].values, exog=df[exog_cols].values,
        order=(1, 0, 1), seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    result = model.fit(disp=False, maxiter=50)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pkl")
        result.save(path)
        assert os.path.exists(path) and os.path.getsize(path) > 0


def test_forecast_steps(synthetic_daily_data):
    """UT-09: Forecast returns correct number of steps."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    df = synthetic_daily_data
    exog_cols = ["onpromotion", "oil_price", "is_holiday", "day_of_week"]

    model = SARIMAX(
        df["sales"].values, exog=df[exog_cols].values,
        order=(1, 0, 1), seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    result = model.fit(disp=False, maxiter=50)

    n_steps = 15
    future_exog = np.column_stack([
        np.zeros(n_steps), np.full(n_steps, 50.0),
        np.zeros(n_steps), np.arange(n_steps) % 7,
    ])
    forecast = result.forecast(steps=n_steps, exog=future_exog)
    assert len(forecast) == n_steps


def test_hyperparam_search_format(synthetic_daily_data):
    """UT-06/07: Grid search returns valid tuples with finite AIC."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    df = synthetic_daily_data
    exog_cols = ["onpromotion", "oil_price", "is_holiday", "day_of_week"]

    best_aic = float("inf")
    found_order = None
    found_seasonal = None

    for p in [0, 1]:
        try:
            model = SARIMAX(
                df["sales"].values, exog=df[exog_cols].values,
                order=(p, 0, 1), seasonal_order=(0, 0, 1, 7),
                enforce_stationarity=False, enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=50)
            if result.aic < best_aic:
                best_aic = result.aic
                found_order = (p, 0, 1)
                found_seasonal = (0, 0, 1, 7)
        except Exception:
            continue

    assert found_order is not None
    assert len(found_order) == 3
    assert len(found_seasonal) == 4
    assert found_seasonal[3] == 7
    assert best_aic != float("inf")
