from kfp import dsl


@dsl.component(
    base_image="python:3.13",
    packages_to_install=["pandas", "fsspec", "gcsfs"],
)
def preprocess(
    train_csv_uri: str,
    oil_csv_uri: str,
    holidays_csv_uri: str,
    forecast_horizon: int,
    train_data: dsl.Output[dsl.Dataset],
    val_data: dsl.Output[dsl.Dataset],
    proportions: dsl.Output[dsl.Dataset],
):
    """Preprocess raw CSVs into aggregated daily time series and compute
    store×family proportions for disaggregation.

    Outputs:
        train_data: Aggregated daily cashflow for training (up to cutoff).
        val_data: Aggregated daily cashflow for validation (last N days).
        proportions: Store×family sales proportions from last 30 days.
    """
    import pandas as pd

    # ------------------------------------------------------------------
    # 1. Read raw data
    # ------------------------------------------------------------------
    print("Reading raw CSVs...")
    train_df = pd.read_csv(train_csv_uri)
    oil_df = pd.read_csv(oil_csv_uri)
    hol_df = pd.read_csv(holidays_csv_uri)

    # ------------------------------------------------------------------
    # 2. Aggregate to daily totals
    # ------------------------------------------------------------------
    daily_df = (
        train_df.groupby("date")
        .agg({"sales": "sum", "onpromotion": "sum"})
        .reset_index()
    )
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df.rename(columns={"sales": "total_cashflow"}, inplace=True)

    # ------------------------------------------------------------------
    # 3. Clean oil prices
    # ------------------------------------------------------------------
    oil_df["date"] = pd.to_datetime(oil_df["date"])
    oil_df.rename(columns={"dcoilwtico": "oil_price"}, inplace=True)
    oil_df["oil_price"] = oil_df["oil_price"].ffill().bfill()

    # ------------------------------------------------------------------
    # 4. Clean holidays — exclude transferred days and Work Days
    # ------------------------------------------------------------------
    hol_df["date"] = pd.to_datetime(hol_df["date"])
    valid_holidays = hol_df[
        (hol_df["transferred"] == False) & (hol_df["type"] != "Work Day")
    ].copy()
    valid_holidays["is_holiday"] = 1
    holiday_flags = valid_holidays[["date", "is_holiday"]].drop_duplicates()

    # ------------------------------------------------------------------
    # 5. Merge into master daily DataFrame
    # ------------------------------------------------------------------
    master_df = daily_df.merge(oil_df, on="date", how="left")
    master_df = master_df.merge(holiday_flags, on="date", how="left")
    master_df["is_holiday"] = master_df["is_holiday"].fillna(0)
    master_df["oil_price"] = master_df["oil_price"].ffill().bfill()
    master_df["day_of_week"] = master_df["date"].dt.dayofweek
    master_df = master_df.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 6. Train / Validation split
    # ------------------------------------------------------------------
    cutoff_date = master_df["date"].max() - pd.Timedelta(days=forecast_horizon)
    train_split = master_df[master_df["date"] <= cutoff_date]
    val_split = master_df[master_df["date"] > cutoff_date]

    print(f"Train rows: {len(train_split)}, Val rows: {len(val_split)}")
    train_split.to_csv(train_data.path, index=False)
    val_split.to_csv(val_data.path, index=False)

    # ------------------------------------------------------------------
    # 7. Compute store×family proportions from last 30 days of raw data
    # ------------------------------------------------------------------
    train_df["date"] = pd.to_datetime(train_df["date"])
    cutoff_30d = train_df["date"].max() - pd.Timedelta(days=30)
    recent_df = train_df[train_df["date"] >= cutoff_30d]

    total_sales = recent_df["sales"].sum()
    if total_sales > 0:
        props = (
            recent_df.groupby(["store_nbr", "family"])["sales"]
            .sum()
            .div(total_sales)
            .reset_index(name="proportion")
        )
    else:
        # Fallback: uniform distribution
        unique_combos = train_df[["store_nbr", "family"]].drop_duplicates()
        n = len(unique_combos)
        props = unique_combos.copy()
        props["proportion"] = 1.0 / n

    print(f"Proportions: {len(props)} store×family combos, sum={props['proportion'].sum():.4f}")
    props.to_csv(proportions.path, index=False)
    print("Preprocessing complete!")
