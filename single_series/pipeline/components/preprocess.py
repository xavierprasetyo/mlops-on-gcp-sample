from kfp import dsl


@dsl.component(
    base_image="python:3.13",
    packages_to_install=["pandas", "fsspec", "gcsfs"],
)
def preprocess(
    train_csv_uri: str,
    oil_csv_uri: str,
    holidays_csv_uri: str,
    target_store_nbr: int,
    target_family: str,
    forecast_horizon: int,
    train_data: dsl.Output[dsl.Dataset],
    val_data: dsl.Output[dsl.Dataset],
):
    """Filter raw data to a single (store_nbr, family) and prepare for SARIMAX.

    Steps:
        1. Filter train.csv to target store_nbr and family.
        2. Merge oil prices and holiday flags as exogenous features.
        3. Add day_of_week feature.
        4. Split into train/validation (last N days held out).

    Outputs:
        train_data: Daily time series for training.
        val_data: Daily time series for validation.
    """
    import pandas as pd

    # ------------------------------------------------------------------
    # 1. Read and filter to target store×family
    # ------------------------------------------------------------------
    print(f"Filtering to store_nbr={target_store_nbr}, family={target_family}")
    train_df = pd.read_csv(train_csv_uri)
    filtered = train_df[
        (train_df["store_nbr"] == target_store_nbr)
        & (train_df["family"] == target_family)
    ].copy()
    print(f"Filtered rows: {len(filtered)}")

    filtered["date"] = pd.to_datetime(filtered["date"])
    filtered = filtered.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Clean oil prices
    # ------------------------------------------------------------------
    oil_df = pd.read_csv(oil_csv_uri)
    oil_df["date"] = pd.to_datetime(oil_df["date"])
    oil_df.rename(columns={"dcoilwtico": "oil_price"}, inplace=True)
    oil_df["oil_price"] = oil_df["oil_price"].ffill().bfill()

    # ------------------------------------------------------------------
    # 3. Clean holidays — exclude transferred and Work Days
    # ------------------------------------------------------------------
    hol_df = pd.read_csv(holidays_csv_uri)
    hol_df["date"] = pd.to_datetime(hol_df["date"])
    valid_holidays = hol_df[
        (hol_df["transferred"] == False) & (hol_df["type"] != "Work Day")
    ].copy()
    valid_holidays["is_holiday"] = 1
    holiday_flags = valid_holidays[["date", "is_holiday"]].drop_duplicates()

    # ------------------------------------------------------------------
    # 4. Merge exogenous features
    # ------------------------------------------------------------------
    master_df = filtered.merge(oil_df[["date", "oil_price"]], on="date", how="left")
    master_df = master_df.merge(holiday_flags, on="date", how="left")
    master_df["is_holiday"] = master_df["is_holiday"].fillna(0)
    master_df["oil_price"] = master_df["oil_price"].ffill().bfill()
    master_df["day_of_week"] = master_df["date"].dt.dayofweek

    # ------------------------------------------------------------------
    # 5. Train / Validation split
    # ------------------------------------------------------------------
    cutoff_date = master_df["date"].max() - pd.Timedelta(days=forecast_horizon)
    train_split = master_df[master_df["date"] <= cutoff_date]
    val_split = master_df[master_df["date"] > cutoff_date]

    print(f"Train: {len(train_split)} rows, Val: {len(val_split)} rows")
    train_split.to_csv(train_data.path, index=False)
    val_split.to_csv(val_data.path, index=False)
    print("Preprocessing complete!")
