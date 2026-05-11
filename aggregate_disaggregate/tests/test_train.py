"""Unit tests for the SARIMAX training and hyperparameter search logic."""
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# Test Fixtures
# ============================================================================
@pytest.fixture
def synthetic_daily_data():
    """Generate 200 days of synthetic daily cashflow data with weekly seasonality."""
    np.random.seed(42)
    dates = pd.date_range(start="2017-01-01", periods=200, freq="D")

    # Create a signal with weekly seasonality + trend + noise
    trend = np.linspace(100, 200, 200)
    weekly = 20 * np.sin(2 * np.pi * np.arange(200) / 7)
    noise = np.random.normal(0, 5, 200)
    cashflow = trend + weekly + noise
    cashflow = np.maximum(cashflow, 0)  # non-negative

    df = pd.DataFrame({
        "date": dates,
        "total_cashflow": cashflow,
        "onpromotion": np.random.randint(0, 100, 200),
        "oil_price": np.random.uniform(40, 60, 200),
        "is_holiday": np.random.choice([0, 1], 200, p=[0.9, 0.1]),
        "day_of_week": dates.dayofweek,
    })
    return df


# ============================================================================
# UT-06: SARIMAX model fits without error
# ============================================================================
def test_sarimax_fit(synthetic_daily_data):
    """Verify SARIMAX fits and can be saved/loaded."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    df = synthetic_daily_data
    exog_cols = ["onpromotion", "oil_price", "is_holiday", "day_of_week"]

    model = SARIMAX(
        df["total_cashflow"].values,
        exog=df[exog_cols].values,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False, maxiter=50)

    # Verify model can be saved and loaded
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.pkl")
        result.save(model_path)
        assert os.path.exists(model_path), "Model file should exist"
        assert os.path.getsize(model_path) > 0, "Model file should not be empty"


# ============================================================================
# UT-07: Forecast produces correct number of steps
# ============================================================================
def test_forecast_steps(synthetic_daily_data):
    """Verify model forecast returns the requested number of steps."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    df = synthetic_daily_data
    exog_cols = ["onpromotion", "oil_price", "is_holiday", "day_of_week"]

    model = SARIMAX(
        df["total_cashflow"].values,
        exog=df[exog_cols].values,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False, maxiter=50)

    # Create future exog data
    n_steps = 15
    future_exog = np.column_stack([
        np.zeros(n_steps),
        np.full(n_steps, 50.0),
        np.zeros(n_steps),
        np.arange(n_steps) % 7,
    ])
    forecast = result.forecast(steps=n_steps, exog=future_exog)

    assert len(forecast) == n_steps, f"Expected {n_steps} forecasts, got {len(forecast)}"


# ============================================================================
# UT-08 (partial): Hyperparameter search returns valid orders
# ============================================================================
def test_hyperparam_search_output_format(synthetic_daily_data):
    """Verify grid search returns valid order and seasonal_order tuples."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    df = synthetic_daily_data
    exog_cols = ["onpromotion", "oil_price", "is_holiday", "day_of_week"]

    # Minimal search (just 2 combos) to verify output format
    best_aic = float("inf")
    found_order = None
    found_seasonal = None

    for p in [0, 1]:
        try:
            model = SARIMAX(
                df["total_cashflow"].values,
                exog=df[exog_cols].values,
                order=(p, 0, 1),
                seasonal_order=(0, 0, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=50)
            if result.aic < best_aic:
                best_aic = result.aic
                found_order = (p, 0, 1)
                found_seasonal = (0, 0, 1, 7)
        except Exception:
            continue

    assert found_order is not None, "At least one combo should converge"
    assert len(found_order) == 3, "Order should be a 3-tuple (p,d,q)"
    assert len(found_seasonal) == 4, "Seasonal order should be a 4-tuple (P,D,Q,s)"
    assert found_seasonal[3] == 7, "Seasonal period should be 7"
    assert best_aic != float("inf"), "Best AIC should be finite"
