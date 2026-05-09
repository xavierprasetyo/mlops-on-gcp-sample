from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["pandas", "statsmodels"],
)
def hyperparam_search(
    train_data: dsl.Input[dsl.Dataset],
    best_order: dsl.Output[dsl.Artifact],
    best_seasonal_order: dsl.Output[dsl.Artifact],
):
    """Grid search over SARIMAX (p,d,q)×(P,D,Q) with s=7 fixed.

    Uses AIC to select the best model configuration.
    Outputs the best order and seasonal_order as JSON text files.
    """
    import itertools
    import json
    import pandas as pd
    import warnings
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    warnings.filterwarnings("ignore")

    train_df = pd.read_csv(train_data.path)
    exog_cols = ["onpromotion", "oil_price", "is_holiday", "day_of_week"]

    endog = train_df["total_cashflow"].values
    exog = train_df[exog_cols].values

    # Search space
    p_range = [0, 1, 2]
    d_range = [0, 1]
    q_range = [0, 1, 2]
    P_range = [0, 1]
    D_range = [0, 1]
    Q_range = [0, 1]
    s = 7  # Weekly seasonality (fixed)

    best_aic = float("inf")
    found_order = (1, 1, 1)  # Sensible default fallback
    found_seasonal = (1, 1, 1, 7)
    total_combos = 0
    successful_fits = 0

    print("Starting SARIMAX hyperparameter grid search...")
    for p, d, q in itertools.product(p_range, d_range, q_range):
        for P, D, Q in itertools.product(P_range, D_range, Q_range):
            total_combos += 1
            try:
                model = SARIMAX(
                    endog,
                    exog=exog,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False, maxiter=50)
                successful_fits += 1

                if result.aic < best_aic:
                    best_aic = result.aic
                    found_order = (p, d, q)
                    found_seasonal = (P, D, Q, s)
                    print(
                        f"  New best: order={found_order}, "
                        f"seasonal={found_seasonal}, AIC={best_aic:.2f}"
                    )
            except Exception:
                # Skip non-convergent parameter combinations
                continue

    print(f"\nGrid search complete: {successful_fits}/{total_combos} converged")
    print(f"Best order: {found_order}, seasonal: {found_seasonal}, AIC: {best_aic:.2f}")

    # Write results as JSON text files (KFP artifact convention)
    with open(best_order.path, "w") as f:
        json.dump(list(found_order), f)

    with open(best_seasonal_order.path, "w") as f:
        json.dump(list(found_seasonal), f)
