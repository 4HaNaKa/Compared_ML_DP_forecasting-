"""
title: Clean preprocessing pipeline for energy-price time series

This script loads an hourly time series dataset parquet with a datetime column and builds a clean, leakage-safe
feature set for forecasting electricity prices. It focuses on clarity and practicality: explicit names, clear
docstrings, and step-by-step reporting. It also removes problematic columns (zero variance and very high
correlations) and imputes missing values using IterativeImputer with a RandomForest estimator. Optionally it runs basic
stationarity and linearity tests and plots an STL decomposition.

Main effects / outputs
1) Prints compact diagnostic reports (missing values, numeric summary, zero-variance columns, high-correlation pairs).
2) Adds leakage-safe features computed only from past information (shifted by 1 hour):
   - calendar and seasonal dummies,
   - rolling statistics of the target,
   - price volatility metrics,
   - price–demand relations (price_over_load, price_minus_norm_load),
   - system imbalance features (sys_margin_MW, imbalance_ratio, price_x_imbalance),
   - renewable share features and interactions,
   - distribution shape (rolling skewness and kurtosis).
3) Saves the final feature table to a parquet file.

Leakage policy:
All features that use contemporaneous system variables are computed with a 1-hour lag (t-1) and rolling windows end at
(t-1). That ensures there is no look-ahead bias when the features are used to predict the price at time t.
"""

from __future__ import annotations
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (required by scikit-learn)
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import skew, kurtosis

# Optional BDS test (either from statsmodels or arch). If unavailable, we skip gracefully.
try:
    from statsmodels.tsa.stattools import acorr_bds  # type: ignore
except Exception:  # pragma: no cover
    try:
        from arch.bootstrap import BDS  # type: ignore

        def acorr_bds(x, max_dim: int = 3):  # type: ignore
            """Lightweight wrapper returning (statistic, p-value) like statsmodels."""
            res = BDS(x, max_dim=max_dim)
            return res.stat, res.pvalue
    except Exception:  # pragma: no cover
        acorr_bds = None  # type: ignore


RAW_DATABASE_PATH: str = "raw_database.parquet"  # input file: .parquet or .csv
DATETIME_COLUMN_NAME: str = "datetime"           # name of the datetime column in RAW_DATABASE_PATH
OUTPUT_PARQUET_PATH: str = "ready_database.parquet" # output feature table
TARGET_COLUMN: str = "fixing_i_price"           # name of the target price column

SHOW_PLOTS: bool = True
ROLLING_WINDOW_HOURS: int = 24
HIGH_CORR_THRESHOLD: float = 0.95

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)



def load_dataset(file_path: str, datetime_column: str) -> pd.DataFrame:
    #Load parquet or CSV by extension and set a datetime index.

    if file_path.lower().endswith(".parquet"):
        dataframe = pd.read_parquet(file_path)
    elif file_path.lower().endswith(".csv"):
        dataframe = pd.read_csv(file_path)
    else:
        raise ValueError("Only .parquet or .csv are supported.")

    if datetime_column not in dataframe.columns:
        raise KeyError(f"Datetime column '{datetime_column}' not found in input file.")

    dataframe[datetime_column] = pd.to_datetime(dataframe[datetime_column])
    dataframe = dataframe.set_index(datetime_column).sort_index()
    print(f"Loaded: {len(dataframe):,} rows x {dataframe.shape[1]} columns")
    return dataframe


def print_missing_report(dataframe: pd.DataFrame) -> None:
    total_rows = len(dataframe)
    missing_count = dataframe.isna().sum()
    missing_percent = (missing_count / total_rows * 100).round(2)
    table = (
        pd.DataFrame({
            "column_name": dataframe.columns,
            "rows": total_rows,
            "missing": missing_count,
            "missing_%": missing_percent,
        })
        .sort_values("missing", ascending=False)
    )
    print("\nColumn | rows | missing | missing %")
    for _, row in table.iterrows():
        print(f"{row['column_name']} | {row['rows']} | {row['missing']} | {row['missing_%']}%")


def deduplicate_datetime_index(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicated datetime index entries and report them."""
    duplicate_count = dataframe.index.duplicated().sum()
    col_duplicate_count = dataframe.columns.duplicated().sum()

    print(f"\nDuplicate index entries: {duplicate_count}")
    print(f"Duplicate column names: {col_duplicate_count}")

    if duplicate_count:
        removed_labels = dataframe.index[dataframe.index.duplicated()].unique().tolist()
        dataframe = dataframe[~dataframe.index.duplicated(keep="first")]
        preview = ", ".join(str(lbl) for lbl in removed_labels[:5])
        tail = "" if len(removed_labels) <= 5 else " …"
        print(f"Removed duplicate index entries: {preview}{tail}")
    return dataframe


def print_head_and_tail(dataframe: pd.DataFrame, n: int = 5) -> None:
    print("\nFirst and last rows:")
    print(pd.concat([dataframe.head(n), dataframe.tail(n)]))


def print_numeric_summary_table(dataframe: pd.DataFrame) -> None:
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    if numeric_columns.empty:
        return
    summary = (
        dataframe[numeric_columns]
        .describe()
        .T.drop(columns=["count"])  # count is often too large to scan
        .apply(lambda col: col.map(lambda x: f"{x:,.2f}"))
    )
    print("\nNumeric summary (mean, std, min, 25%, 50%, 75%, max):")
    print(summary.to_string())


def drop_zero_variance_features(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Remove columns with zero variance among numeric columns and return their names."""
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    selector = VarianceThreshold(threshold=0.0)
    selector.fit(dataframe[numeric_columns])
    removed = [col for col, keep in zip(numeric_columns, selector.get_support()) if not keep]
    if removed:
        print("\nZero-variance columns removed:")
        print(", ".join(removed))
        dataframe = dataframe.drop(columns=removed)
    else:
        print("\nZero-variance columns: none")
    return dataframe, removed


def drop_highly_correlated_features(dataframe: pd.DataFrame, threshold: float = HIGH_CORR_THRESHOLD) -> pd.DataFrame:
    """Drop one column from each highly-correlated pair (|rho| >= threshold).

    Strategy: for any column with at least one partner above the threshold, we drop the member of the pair with the
    higher mean absolute correlation to the rest of the matrix. This tends to keep the more "unique" signal.
    """
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    corr = dataframe[numeric_columns].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    columns_to_drop: List[str] = []
    mean_corr = corr.mean().to_dict()

    for column in upper.columns:
        partners = upper.index[upper[column] >= threshold].tolist()
        if partners:
            candidates = [column] + partners
            # drop the one with the higher average correlation to others
            drop_candidate = max(candidates, key=lambda c: mean_corr.get(c, 0.0))
            keep_candidate = min(candidates, key=lambda c: mean_corr.get(c, 0.0))
            print(f"High corr (|rho|>={threshold:.2f}): {column} vs {partners[0]}  ->  drop {drop_candidate}, keep {keep_candidate}")
            columns_to_drop.append(drop_candidate)

    if columns_to_drop:
        dataframe = dataframe.drop(columns=sorted(set(columns_to_drop)))
    return dataframe


def print_high_correlation_pairs(dataframe: pd.DataFrame, threshold: float = HIGH_CORR_THRESHOLD) -> None:
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    corr = dataframe[numeric_columns].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    highly_corr = (
        upper.stack()
        .reset_index()
        .rename(columns={0: "corr", "level_0": "feature_1", "level_1": "feature_2"})
        .query("corr >= @threshold")
        .sort_values("corr", ascending=False)
    )
    if not highly_corr.empty:
        print(f"Highly correlated pairs (|rho| >= {threshold}):")
        for _, row in highly_corr.iterrows():
            print(f"{row['feature_1']} <-> {row['feature_2']} = {row['corr']:.2f}")
    else:
        print(f"\nHighly correlated pairs (|rho| >= {threshold}): none")



def run_stationarity_tests(dataframe: pd.DataFrame, target_column: str = TARGET_COLUMN) -> None:
    if target_column not in dataframe.columns:
        return
    series = dataframe[target_column].dropna()
    if len(series) < 100:
        return

    adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress InterpolationWarning
        kpss_stat, kpss_p, *_ = kpss(series, regression="c", nlags="auto")

    print("\nStationarity tests for the target column")
    print(f"ADF statistic = {adf_stat:.2f}, p-value = {adf_p:.4f}")
    print(f"KPSS statistic = {kpss_stat:.2f}, p-value = {kpss_p:.4f}")


def plot_stl_decomposition(dataframe: pd.DataFrame, target_column: str = TARGET_COLUMN, period: int = 24 * 7) -> None:
    if not SHOW_PLOTS or target_column not in dataframe.columns:
        return
    series = dataframe[target_column].dropna()
    stl = STL(series, period=period, robust=True).fit()
    stl.plot()
    plt.suptitle(f"STL decomposition ({period}-period)")
    plt.tight_layout()
    plt.show()


def run_bds_linearity_test(dataframe: pd.DataFrame, target_column: str = TARGET_COLUMN) -> None:
    if acorr_bds is None:
        print("\nBDS test not available - skipping linearity test.")
        return
    if target_column not in dataframe.columns:
        print(f"\nTarget column '{target_column}' not in dataset - skipping BDS test.")
        return
    series = dataframe[target_column].dropna()
    if len(series) < 100:
        print("\nSeries too short for BDS test - skipping.")
        return
    bds_stat, bds_p = acorr_bds(series, max_dim=2)
    print("\nLinearity test (BDS)")
    print(f"BDS statistic = {bds_stat:.2f}, p-value = {bds_p:.4f}")


def impute_missing_values_random_forest(dataframe: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """Impute missing values using IterativeImputer (RandomForest) for numeric features except the target,
    and a separate RandomForest for the target itself if needed.
    """
    output = dataframe.copy()
    numeric = output.select_dtypes(np.number).columns.drop(target_column, errors="ignore")

    if numeric.size:
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=800,
                max_depth=12,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=0,
                n_jobs=-1,
            ),
            initial_strategy="median",
            max_iter=10,
            random_state=0,
        )
        output[numeric] = imputer.fit_transform(output[numeric])

    # Target imputation (if required)
    if target_column in output.columns and output[target_column].isna().any():
        mask = output[target_column].isna()
        rf_target = RandomForestRegressor(
            n_estimators=800,
            max_depth=14,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=1,
            n_jobs=-1,
        ).fit(output.loc[~mask, numeric], output.loc[~mask, target_column])
        output.loc[mask, target_column] = rf_target.predict(output.loc[mask, numeric])
    return output


def add_calendar_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features (all computed from the datetime index)."""
    output = dataframe.copy()
    index = output.index
    output["hour"] = index.hour
    output["day_of_week"] = index.dayofweek
    output["month"] = index.month
    output["day_of_year"] = index.dayofyear
    output["weekend"] = (index.dayofweek >= 5).astype(int)
    output["sin24"] = np.sin(2 * np.pi * index.hour / 24)
    output["cos24"] = np.cos(2 * np.pi * index.hour / 24)
    weekly_position = index.dayofweek * 24 + index.hour
    output["sin168"] = np.sin(2 * np.pi * weekly_position / 168)
    output["cos168"] = np.cos(2 * np.pi * weekly_position / 168)
    return output


def add_season_features(dataframe: pd.DataFrame, add_dummies: bool = True) -> pd.DataFrame:
    """Add quarter and seasonal dummies.

    Creates: quarter and, if add_dummies, season_autumn, season_spring, season_summer, season_winter.
    """
    output = dataframe.copy()
    output["quarter"] = output.index.to_series().dt.quarter
    season_map = {
        1: "winter", 2: "winter", 12: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn",
    }
    months = output.index.to_series().dt.month
    output["season"] = months.map(season_map)
    if add_dummies:
        dummies = pd.get_dummies(output["season"], prefix="season", dtype="int8")
        dummies.index = output.index
        output = output.join(dummies)
        output.drop(columns=["season"], inplace=True)
    return output


def add_generation_and_trade_ratios(
    dataframe: pd.DataFrame,
    wind_column: str = "pl_produkcja_wiatr",
    solar_column: str = "pl_produkcja_slonce",
    demand_column: str = "zapotrzebowanie_na_moc",
    import_export_column: str = "saldo_wymiany_miedzysystemowej_rownoleglej",
) -> pd.DataFrame:
    """Compute lagged shares wrt demand.

    wind_share = wind_{t-1} / demand_{t-1}
    solar_share = solar_{t-1} / demand_{t-1}
    import_share = import_export_{t-1} / demand_{t-1}
    """
    needed = {wind_column, solar_column, demand_column, import_export_column}
    if not needed.issubset(dataframe.columns):
        return dataframe
    output = dataframe.copy()
    eps = 1e-9

    wind_prev = output[wind_column].shift(1)
    solar_prev = output[solar_column].shift(1)
    demand_prev = output[demand_column].shift(1)
    import_export_prev = output[import_export_column].shift(1)

    with np.errstate(divide="ignore", invalid="ignore"):
        output["wind_share"] = wind_prev / (demand_prev.replace(0, np.nan) + eps)
        output["solar_share"] = solar_prev / (demand_prev.replace(0, np.nan) + eps)
        output["import_share"] = import_export_prev / (demand_prev.replace(0, np.nan) + eps)
    return output


def add_target_rolling_statistics(
    dataframe: pd.DataFrame,
    column_name: str = TARGET_COLUMN,
    window_size: int = ROLLING_WINDOW_HOURS,
) -> pd.DataFrame:
    """Add rolling mean, std, min, max for the target using windows ending at t-1."""
    if column_name not in dataframe.columns:
        return dataframe
    output = dataframe.copy()
    roll = output[column_name].rolling(window=window_size, min_periods=1)
    output[f"{column_name}_roll{window_size}_mean"] = roll.mean().shift(1)
    output[f"{column_name}_roll{window_size}_std"] = roll.std().shift(1)
    output[f"{column_name}_roll{window_size}_min"] = roll.min().shift(1)
    output[f"{column_name}_roll{window_size}_max"] = roll.max().shift(1)
    return output


def add_price_volatility_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Absolute differences and rolling volatility of the target computed up to t-1."""
    output = dataframe.copy()
    price = output[TARGET_COLUMN]
    output["d_price_1h"] = price.diff().abs().shift(1)
    output["d_price_24h"] = price.diff(24).abs().shift(1)
    output["std_6h"] = price.rolling(6, min_periods=1).std().shift(1)
    output["std_24h"] = price.rolling(24, min_periods=1).std().shift(1)
    return output


def add_price_demand_relationship_features(
    dataframe: pd.DataFrame,
    price_column: str = TARGET_COLUMN,
    load_column: str = "zapotrzebowanie_na_moc",
    window_size: int = 24,
) -> pd.DataFrame:
    """Price-to-demand relation from t-1 and normalized load deviation.

    price_over_load = price_{t-1} / load_{t-1}
    price_minus_norm_load = price_{t-1} - zscore(load_{t-1}) within a rolling window ending at t-1
    """
    if price_column not in dataframe.columns or load_column not in dataframe.columns:
        return dataframe
    output = dataframe.copy()
    eps = 1e-9

    price_prev = output[price_column].shift(1)
    load_prev = output[load_column].shift(1)

    output["price_over_load"] = price_prev / (load_prev.replace(0, np.nan) + eps)

    roll_med = load_prev.rolling(window_size, min_periods=max(6, window_size // 4)).median()
    roll_mad = load_prev.rolling(window_size, min_periods=max(6, window_size // 4)).apply(
        lambda x: (x - x.median()).abs().median(), raw=False
    )
    load_z = (load_prev - roll_med) / (roll_mad + eps)
    output["price_minus_norm_load"] = price_prev - load_z
    return output


def add_system_imbalance_features(
    dataframe: pd.DataFrame,
    up_reserve_column: str = "rezerwa_mocy_ponad_zapotrzebowanie",
    down_reserve_column: str = "rezerwa_mocy_ponizej_zapotrzebowania",
    load_column: str = "zapotrzebowanie_na_moc",
    price_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """System imbalance variables from t-1 and interaction with the previous price.

    sys_margin_MW = (up - down)_{t-1}
    imbalance_ratio = sys_margin_MW_{t-1} / load_{t-1}
    price_x_imbalance = price_{t-1} * imbalance_ratio_{t-1}
    """
    required = {up_reserve_column, down_reserve_column, load_column, price_column}
    if not required.issubset(dataframe.columns):
        return dataframe

    output = dataframe.copy()
    eps = 1e-9

    system_margin_prev = (output[up_reserve_column] - output[down_reserve_column]).shift(1)
    load_prev = output[load_column].shift(1)
    price_prev = output[price_column].shift(1)

    output["sys_margin_MW"] = system_margin_prev
    output["imbalance_ratio"] = system_margin_prev / (load_prev.replace(0, np.nan) + eps)
    output["price_x_imbalance"] = price_prev * output["imbalance_ratio"]
    return output


def add_distribution_shape_features(
    dataframe: pd.DataFrame,
    price_column: str = TARGET_COLUMN,
    window_size: int = 24,
) -> pd.DataFrame:
    """Rolling skewness and kurtosis for the target using windows ending at t-1."""
    if price_column not in dataframe.columns:
        return dataframe
    output = dataframe.copy()
    roll = output[price_column].rolling(window_size, min_periods=max(6, window_size // 4))
    output[f"skew_{window_size}h"] = roll.apply(lambda x: skew(x, bias=False), raw=False).shift(1)
    output[f"kurt_{window_size}h"] = roll.apply(lambda x: kurtosis(x, bias=False), raw=False).shift(1)
    return output


def add_renewable_share_features(
    dataframe: pd.DataFrame,
    price_column: str = TARGET_COLUMN,
    wind_generation_column: str = "generacja_zrodel_wiatrowych",
    solar_generation_column: str = "generacja_zrodel_fotowoltaicznych",
    total_generation_column: str = "sumaryczna_generacja_JGxx",
) -> pd.DataFrame:
    """Renewable share from t-1 and interaction with the previous price.
    renew_share = (wind + solar)_{t-1} / total_{t-1}
    price_x_renew_share = price_{t-1} * renew_share_{t-1}
    """
    needed = {wind_generation_column, solar_generation_column, total_generation_column, price_column}
    if not needed.issubset(dataframe.columns):
        return dataframe

    output = dataframe.copy()
    eps = 1e-9

    wind_prev = output[wind_generation_column].shift(1)
    solar_prev = output[solar_generation_column].shift(1)
    total_prev = output[total_generation_column].shift(1)
    price_prev = output[price_column].shift(1)

    renewable_share_prev = (wind_prev + solar_prev) / (total_prev.replace(0, np.nan) + eps)
    output["renew_share"] = renewable_share_prev
    output["price_x_renew_share"] = price_prev * renewable_share_prev
    return output


def main() -> None:
    # 1) Load
    dataframe = load_dataset(RAW_DATABASE_PATH, DATETIME_COLUMN_NAME)

    # 2) Basic diagnostics
    print_missing_report(dataframe)
    print_head_and_tail(dataframe)
    print_numeric_summary_table(dataframe)

    # 3) De-duplicate datetime index
    dataframe = deduplicate_datetime_index(dataframe)

    # 4) Remove trivial problems
    dataframe, _ = drop_zero_variance_features(dataframe)
    dataframe = drop_highly_correlated_features(dataframe, threshold=HIGH_CORR_THRESHOLD)
    print_high_correlation_pairs(dataframe, threshold=HIGH_CORR_THRESHOLD)

    # 5) Optional tests (no mutation)
    run_stationarity_tests(dataframe)
    run_bds_linearity_test(dataframe)

    # 6) First-pass imputation (for raw features that are needed to compute engineered features)
    dataframe = impute_missing_values_random_forest(dataframe)

    # 7) Feature engineering (all leakage-safe)
    dataframe = add_calendar_features(dataframe)
    dataframe = add_season_features(dataframe, add_dummies=True)
    dataframe = add_generation_and_trade_ratios(dataframe)
    dataframe = add_target_rolling_statistics(dataframe)

    dataframe = add_price_volatility_features(dataframe)
    dataframe[["d_price_1h", "d_price_24h", "std_6h", "std_24h"]] = (
        dataframe[["d_price_1h", "d_price_24h", "std_6h", "std_24h"]].fillna(0)
    )

    dataframe = add_price_demand_relationship_features(dataframe)
    dataframe = add_system_imbalance_features(dataframe)
    dataframe = add_distribution_shape_features(dataframe, window_size=24)
    dataframe = add_renewable_share_features(dataframe)

    # Some rolling columns may still start with NaN; forward-fill them
    rolling_cols = [c for c in dataframe.columns if c.startswith(f"{TARGET_COLUMN}_roll")]
    if rolling_cols:
        dataframe[rolling_cols] = dataframe[rolling_cols].ffill()

    # 8) Second-pass imputation (clean up any NaNs introduced by engineering)
    dataframe = impute_missing_values_random_forest(dataframe)

    # 9) Optional STL plot and final reports
    plot_stl_decomposition(dataframe)
    print_numeric_summary_table(dataframe)
    print_missing_report(dataframe)

    # 10) Save final table (drop rows that still contain NaN)
    cleaned = dataframe.dropna()
    cleaned.to_parquet(OUTPUT_PARQUET_PATH)
    print(f"Saved {len(cleaned):,} rows: {cleaned.shape[1]} and columns: {OUTPUT_PARQUET_PATH}")


if __name__ == "__main__":
    main()
