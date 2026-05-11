from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["pandas", "statsmodels"],
)
def train_model(
    train_data: dsl.Input[dsl.Dataset],
    best_order: dsl.Input[dsl.Artifact],
    best_seasonal_order: dsl.Input[dsl.Artifact],
    model_artifact: dsl.Output[dsl.Model],
):
    """Train a SARIMAX model on a single (store_nbr, family) time series.

    Saves the fitted model as model.pkl inside the model artifact directory.
    """
    import json
    import os
    import pandas as pd
    import warnings
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    warnings.filterwarnings("ignore")

    train_df = pd.read_csv(train_data.path)
    exog_cols = ["onpromotion", "oil_price", "is_holiday", "day_of_week"]

    with open(best_order.path, "r") as f:
        order = tuple(json.load(f))
    with open(best_seasonal_order.path, "r") as f:
        seasonal_order = tuple(json.load(f))

    print(f"Training SARIMAX with order={order}, seasonal_order={seasonal_order}")

    model = SARIMAX(
        train_df["sales"].values,
        exog=train_df[exog_cols].values,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    model_fit = model.fit(disp=False)

    print(f"Model AIC: {model_fit.aic:.2f}, BIC: {model_fit.bic:.2f}")

    # Save model — Vertex AI requires artifact_uri to be a directory
    os.makedirs(model_artifact.path, exist_ok=True)
    model_fit.save(os.path.join(model_artifact.path, "model.pkl"))

    # Save training data for amnesia-curing in the CPR container
    # This avoids downloading the full 120MB train.csv at serving time
    train_df.to_csv(os.path.join(model_artifact.path, "train_history.csv"), index=False)

    with open(os.path.join(model_artifact.path, "params.json"), "w") as f:
        json.dump(
            {"order": list(order), "seasonal_order": list(seasonal_order),
             "aic": model_fit.aic, "bic": model_fit.bic},
            f,
        )
    print(f"Model saved to {model_artifact.path}")
