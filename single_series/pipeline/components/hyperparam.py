from kfp import dsl


@dsl.component(
    base_image="python:3.13",
    packages_to_install=["pandas", "statsmodels", "google-cloud-aiplatform"],
)
def hyperparam_search(
    train_data: dsl.Input[dsl.Dataset],
    project: str,
    location: str,
    experiment_name: str,
    best_order: dsl.Output[dsl.Artifact],
    best_seasonal_order: dsl.Output[dsl.Artifact],
):
    """Grid search over SARIMAX (p,d,q)×(P,D,Q) with s=7 fixed.

    Uses AIC to select the best model configuration.
    Logs each trial to Vertex AI Experiments for comparison.
    Outputs the best order and seasonal_order as JSON text files.
    """
    import itertools
    import json
    import math
    import pandas as pd
    import warnings
    from datetime import datetime
    from google.cloud import aiplatform
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    warnings.filterwarnings("ignore")

    def _safe_metric(val, default=-1.0):
        """Vertex AI Experiments rejects NaN / Infinity metric values."""
        v = float(val)
        return default if (math.isnan(v) or math.isinf(v)) else v

    # Initialize Vertex AI Experiments context
    aiplatform.init(
        project=project, location=location, experiment=experiment_name
    )

    train_df = pd.read_csv(train_data.path)
    exog_cols = ["onpromotion", "oil_price", "is_holiday", "day_of_week"]

    endog = train_df["sales"].values
    exog = train_df[exog_cols].values

    # Search space: 3×2×3 × 2×2×2 = 144 combinations
    p_range = [0, 1, 2]
    d_range = [0, 1]
    q_range = [0, 1, 2]
    P_range = [0, 1]
    D_range = [0, 1]
    Q_range = [0, 1]
    s = 7  # Weekly seasonality

    best_aic = float("inf")
    found_order = (1, 1, 1)  # Default fallback
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

                aic_val = float(result.aic)
                bic_val = float(result.bic)

                # Log each trial to Vertex AI Experiments
                ts = datetime.now().strftime("%Y%m%d%H%M%S")
                run_id = f"grid-p{p}d{d}q{q}-P{P}D{D}Q{Q}-{ts}"
                with aiplatform.start_run(run_id) as run:
                    run.log_params({
                        "p": p, "d": d, "q": q,
                        "P": P, "D": D, "Q": Q, "s": s,
                        "training_rows": len(train_df),
                        "stage": "hyperparam_search",
                    })
                    run.log_metrics({
                        "aic": _safe_metric(aic_val),
                        "bic": _safe_metric(bic_val),
                    })

                # Only update best if AIC is a finite number
                if math.isfinite(aic_val) and aic_val < best_aic:
                    best_aic = result.aic
                    found_order = (p, d, q)
                    found_seasonal = (P, D, Q, s)
                    print(
                        f"  New best: order={found_order}, "
                        f"seasonal={found_seasonal}, AIC={best_aic:.2f}"
                    )
            except Exception:
                continue

    print(f"\nGrid search complete: {successful_fits}/{total_combos} converged")
    print(f"Best order: {found_order}, seasonal: {found_seasonal}, AIC: {best_aic:.2f}")

    # Log the best result as a summary run
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    with aiplatform.start_run(f"grid-search-best-{ts}") as run:
        run.log_params({
            "best_p": found_order[0], "best_d": found_order[1], "best_q": found_order[2],
            "best_P": found_seasonal[0], "best_D": found_seasonal[1], "best_Q": found_seasonal[2],
            "s": s, "total_combos": total_combos, "successful_fits": successful_fits,
            "stage": "hyperparam_search_summary",
        })
        run.log_metrics({"best_aic": _safe_metric(best_aic)})

    with open(best_order.path, "w") as f:
        json.dump(list(found_order), f)
    with open(best_seasonal_order.path, "w") as f:
        json.dump(list(found_seasonal), f)
