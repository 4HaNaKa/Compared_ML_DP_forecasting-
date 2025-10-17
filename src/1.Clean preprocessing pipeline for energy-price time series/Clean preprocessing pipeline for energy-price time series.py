# -*- coding: utf-8 -*-
"""
This script loads an hourly time series dataset in parquet format with a datetime column and creates a clean,
leak-resistant feature set for electricity price forecasting. It also removes problematic columns (zero variance
and very high correlations) and imputes missing values using IterativeImputer with the RandomForest estimator.
Optionally, it performs basic stationarity and linearity tests and plots the STL decomposition.

Results:
1) Prints concise diagnostic reports (missing values, numerical summary, zero variance columns, high correlation pairs).
2) Adds leak protection features calculated solely on the basis of historical information (shifted by 1 hour):
   - calendar and seasonal variables,
   - rolling target statistics,
   - price volatility indicators,
   - price-demand relationships (price_over_load, price_minus_norm_load),
   - system imbalance characteristics (sys_margin_MW, imbalance_ratio, price_x_imbalance),
   - renewable energy share characteristics and interactions,
   - distribution shape (moving skewness and kurtosis).
3) Saves the final feature table to a parquet file.

Outputs:
- data/processed/ready_database.parquet
- results/plots/stl_decomposition.png
- console reports: missing values, numeric summary, correlations
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (required)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import IterativeImputer
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss


# ========================= PATHS & GLOBALS =========================

def bootstrap_paths_preprocessing() -> SimpleNamespace:
    """Locate project root and prepare canonical folders (run‑from‑anywhere)."""
    def _find_root(anchors=("README.md", "requirements.txt", ".git")) -> Path:
        try:
            here = Path(__file__).resolve()
        except NameError:  # __file__ can be undefined (e.g., notebooks)
            here = Path.cwd().resolve()
        for parent in [here] + list(here.parents):
            if any((parent / a).exists() for a in anchors):
                return parent
        env = os.environ.get("PROJECT_ROOT")
        return Path(env).resolve() if env else here

    root = _find_root()
    data = root / "data"
    raw = data / "raw"
    processed = data / "processed"

    results = root / "results"
    plots = results / "plots"
    tables = results / "tables"

    for d in (processed, results, plots, tables):
        d.mkdir(parents=True, exist_ok=True)

    return SimpleNamespace(
        project_root=root,
        data_dir=data,
        raw_dir=raw,
        processed_dir=processed,
        results_dir=results,
        plots_dir=plots,
        tables_dir=tables,
        raw_parquet=raw / "raw_database.parquet",
        out_parquet=processed / "ready_database.parquet",
    )


PATHS = bootstrap_paths_preprocessing()
RAW_DATABASE_PATH = PATHS.raw_parquet
OUTPUT_PARQUET_PATH = PATHS.out_parquet

DATETIME_COLUMN_NAME = "datetime"  # preferred name; fallback to "timestamp" if missing
TARGET_COLUMN: str = "fixing_i_price"

SHOW_PLOTS: bool = True
ROLLING_WINDOW_HOURS: int = 24
HIGH_CORR_THRESHOLD: float = 0.95

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)


# ========================= I/O HELPERS =========================

def _coerce_datetime_column(df: pd.DataFrame, datetime_column: str) -> Tuple[pd.DataFrame, str]:
    """Ensure a datetime column exists; accept a common alias if needed."""
    if datetime_column not in df.columns:
        alias = "timestamp" if "timestamp" in df.columns else None
        if alias is None:
            raise KeyError(
                f"Datetime column '{datetime_column}' not found and no 'timestamp' alias present."
            )
        datetime_column = alias

    # Remove embedded header rows like "timestamp"/"datetime" if they appear in data rows
    col = str(datetime_column)
    df[col] = df[col].astype(str).str.strip()
    bad_rows = df[col].str.lower().isin(["timestamp", "datetime"])
    if bad_rows.any():
        df = df.loc[~bad_rows].copy()

    # Parse & index
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=[col]).set_index(col).sort_index()
    return df, datetime_column


def load_dataset(file_path: Union[str, Path], datetime_column: str) -> pd.DataFrame:
    """Load Parquet or CSV. If missing/unreadable, return empty DataFrame."""
    p = Path(file_path)
    if not p.exists():
        print(f"[WARN] Input file not found: {p}")
        return pd.DataFrame()

    try:
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        elif p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
        else:
            print(f"[WARN] Unsupported extension '{p.suffix}'. Expected .parquet or .csv")
            return pd.DataFrame()
    except Exception as exc:  # read failure should not crash the run
        print(f"[WARN] Failed to read {p}: {exc}")
        return pd.DataFrame()

    try:
        df, used_col = _coerce_datetime_column(df, datetime_column)
    except Exception as exc:
        print(f"[WARN] Datetime parsing failed: {exc}")
        return pd.DataFrame()

    print(f"Loaded: {len(df):,} rows x {df.shape[1]} columns from {p} (datetime='{used_col}')")
    return df


# ========================= REPORTS =========================

def print_missing_report(df: pd.DataFrame) -> None:
    if df.empty:
        print("[INFO] Missing report skipped (empty frame)")
        return
    total_rows = len(df)
    missing_count = df.isna().sum()
    missing_percent = (missing_count / max(1, total_rows) * 100).round(2)
    table = (
        pd.DataFrame({
            "column_name": df.columns,
            "rows": total_rows,
            "missing": missing_count,
            "missing_%": missing_percent,
        }).sort_values("missing", ascending=False)
    )
    print("\nColumn | rows | missing | missing %")
    for _, row in table.iterrows():
        print(f"{row['column_name']} | {row['rows']} | {row['missing']} | {row['missing_%']}%")


def deduplicate_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    duplicate_count = df.index.duplicated().sum()
    col_duplicate_count = df.columns.duplicated().sum()

    print(f"\nDuplicate index entries: {duplicate_count}")
    print(f"Duplicate column names: {col_duplicate_count}")

    if duplicate_count:
        removed_labels = df.index[df.index.duplicated()].unique().tolist()
        df = df[~df.index.duplicated(keep="first")]
        preview = ", ".join(str(lbl) for lbl in removed_labels[:5])
        tail = "" if len(removed_labels) <= 5 else " …"
        print(f"Removed duplicate index entries: {preview}{tail}")
    return df


def print_head_and_tail(df: pd.DataFrame, n: int = 5) -> None:
    if df.empty:
        print("[INFO] Head/tail skipped (empty frame).")
        return
    print("\nFirst and last rows:")
    print(pd.concat([df.head(n), df.tail(n)]))


def print_numeric_summary_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("[INFO] Numeric summary skipped (empty frame).")
        return
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) == 0:
        print("[INFO] No numeric columns for summary.")
        return
    summary = (
        df[numeric_columns]
        .describe()
        .T.drop(columns=["count"], errors="ignore")
        .apply(lambda col: col.map(lambda x: f"{x:,.2f}"))
    )
    print("\nNumeric summary (mean, std, min, 25%, 50%, 75%, max):")
    print(summary.to_string())


# ========================= CLEANING & FILTERS =========================

def drop_zero_variance_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    if df.empty:
        return df, []
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) == 0:
        return df, []

    selector = VarianceThreshold(threshold=0.0)
    selector.fit(df[numeric_columns])
    removed = [col for col, keep in zip(numeric_columns, selector.get_support()) if not keep]
    if removed:
        print("\nZero‑variance columns removed:")
        print(", ".join(removed))
        df = df.drop(columns=removed)
    else:
        print("\nZero‑variance columns: none")
    return df, removed


def drop_highly_correlated_features(
    df: pd.DataFrame,
    threshold: float = HIGH_CORR_THRESHOLD,
    *,
    protected_cols: Iterable[str] = (TARGET_COLUMN,),
) -> pd.DataFrame:
    """Drop one column from each highly‑correlated pair (|rho| >= threshold).

    Strategy: within any correlated group, drop the member with the higher mean absolute
    correlation to the rest (keep the more "unique" signal). Protected columns are never dropped.
    """
    if df.empty:
        return df

    protected = set(protected_cols or [])
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) < 2:
        return df

    corr = df[numeric_columns].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    columns_to_drop: List[str] = []
    mean_corr = corr.mean().to_dict()

    for col in upper.columns:
        partners = upper.index[upper[col] >= threshold].tolist()
        if not partners:
            continue
        candidates = [col] + partners

        # Manual rule for Fixing I/II (keep Fixing I)
        if "fixing_i_price" in candidates and "fixing_ii_price" in candidates:
            if "fixing_ii_price" not in protected:
                print(
                    f"High corr (|rho|>={threshold:.2f}): fixing_ii_price vs fixing_i_price  ->  drop fixing_ii_price, keep fixing_i_price"
                )
                columns_to_drop.append("fixing_ii_price")
            continue

        cands_wo_protected = [c for c in candidates if c not in protected]
        if not cands_wo_protected:
            continue

        drop_candidate = max(cands_wo_protected, key=lambda c: mean_corr.get(c, 0.0))
        print(f"High corr (|rho|>={threshold:.2f}): drop {drop_candidate}")
        columns_to_drop.append(drop_candidate)

    columns_to_drop = sorted(set(columns_to_drop))
    if columns_to_drop:
        df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])
    return df


def print_high_correlation_pairs(df: pd.DataFrame, threshold: float = HIGH_CORR_THRESHOLD) -> None:
    if df.empty:
        print("\nHighly correlated pairs: none (empty frame)")
        return
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) < 2:
        print("\nHighly correlated pairs: none (too few numeric columns)")
        return
    corr = df[numeric_columns].corr().abs()
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


# ========================= TESTS (NO MUTATION) =========================

def run_stationarity_tests(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> None:
    if df.empty or target_column not in df.columns:
        return
    series = df[target_column].dropna()
    if len(series) < 100:
        return

    adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, *_ = kpss(series, regression="c", nlags="auto")

    print("\nStationarity tests for the target column")
    print(f"ADF statistic = {adf_stat:.2f}, p‑value = {adf_p:.4f}")
    print(f"KPSS statistic = {kpss_stat:.2f}, p‑value = {kpss_p:.4f}")


# ========================= IMPUTATION =========================

def impute_missing_values_random_forest(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """Impute numeric features with IterativeImputer(RandomForest). If target has gaps, fill it with a separate RF."""
    if df.empty:
        return df
    out = df.copy()
    numeric = out.select_dtypes(np.number).columns.drop(target_column, errors="ignore")

    if numeric.size:
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=600,
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
        out[numeric] = imputer.fit_transform(out[numeric])

    # Target imputation (if needed)
    if target_column in out.columns and out[target_column].isna().any():
        mask = out[target_column].isna()
        rf_target = RandomForestRegressor(
            n_estimators=600,
            max_depth=14,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=1,
            n_jobs=-1,
        ).fit(out.loc[~mask, numeric], out.loc[~mask, target_column])
        out.loc[mask, target_column] = rf_target.predict(out.loc[mask, numeric])
    return out


# ========================= FEATURE ENGINEERING =========================

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    idx = out.index
    out["hour"] = idx.hour
    out["day_of_week"] = idx.dayofweek
    out["month"] = idx.month
    out["day_of_year"] = idx.dayofyear
    out["weekend"] = (idx.dayofweek >= 5).astype(int)
    out["sin24"] = np.sin(2 * np.pi * idx.hour / 24)
    out["cos24"] = np.cos(2 * np.pi * idx.hour / 24)
    weekly_position = idx.dayofweek * 24 + idx.hour
    out["sin168"] = np.sin(2 * np.pi * weekly_position / 168)
    out["cos168"] = np.cos(2 * np.pi * weekly_position / 168)
    return out


def add_season_features(df: pd.DataFrame, add_dummies: bool = True) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["quarter"] = out.index.to_series().dt.quarter
    season_map = {
        1: "winter", 2: "winter", 12: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn",
    }
    months = out.index.to_series().dt.month
    out["season"] = months.map(season_map)
    if add_dummies:
        dummies = pd.get_dummies(out["season"], prefix="season", dtype="int8")
        dummies.index = out.index
        out = out.join(dummies)
        out.drop(columns=["season"], inplace=True)
    return out


def add_generation_and_trade_ratios(
    df: pd.DataFrame,
    wind_column: str = "pl_produkcja_wiatr",
    solar_column: str = "pl_produkcja_slonce",
    demand_column: str = "zapotrzebowanie_na_moc",
    import_export_column: str = "saldo_wymiany_miedzysystemowej_rownoleglej",
) -> pd.DataFrame:
    """Compute lagged shares wrt demand (all at t‑1)."""
    needed = {wind_column, solar_column, demand_column, import_export_column}
    if df.empty or not needed.issubset(df.columns):
        return df
    out = df.copy()
    eps = 1e-9

    wind_prev = out[wind_column].shift(1)
    solar_prev = out[solar_column].shift(1)
    demand_prev = out[demand_column].shift(1)
    import_export_prev = out[import_export_column].shift(1)

    with np.errstate(divide="ignore", invalid="ignore"):
        out["wind_share"] = wind_prev / (demand_prev.replace(0, np.nan) + eps)
        out["solar_share"] = solar_prev / (demand_prev.replace(0, np.nan) + eps)
        out["import_share"] = import_export_prev / (demand_prev.replace(0, np.nan) + eps)
    return out


def add_target_rolling_statistics(
    df: pd.DataFrame,
    column_name: str = TARGET_COLUMN,
    window_size: int = ROLLING_WINDOW_HOURS,
) -> pd.DataFrame:
    if df.empty or column_name not in df.columns:
        return df
    out = df.copy()
    roll = out[column_name].rolling(window=window_size, min_periods=1)
    out[f"{column_name}_roll{window_size}_mean"] = roll.mean().shift(1)
    out[f"{column_name}_roll{window_size}_std"] = roll.std().shift(1)
    out[f"{column_name}_roll{window_size}_min"] = roll.min().shift(1)
    out[f"{column_name}_roll{window_size}_max"] = roll.max().shift(1)
    return out


def add_price_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or TARGET_COLUMN not in df.columns:
        return df
    out = df.copy()
    price = out[TARGET_COLUMN]
    out["d_price_1h"] = price.diff().abs().shift(1)
    out["d_price_24h"] = price.diff(24).abs().shift(1)
    out["std_6h"] = price.rolling(6, min_periods=1).std().shift(1)
    out["std_24h"] = price.rolling(24, min_periods=1).std().shift(1)
    return out


def add_price_demand_relationship_features(
    df: pd.DataFrame,
    price_column: str = TARGET_COLUMN,
    load_column: str = "zapotrzebowanie_na_moc",
    window_size: int = 24,
) -> pd.DataFrame:
    """Price‑to‑demand relation from t‑1 and normalized load deviation."""
    if df.empty or price_column not in df.columns or load_column not in df.columns:
        return df
    out = df.copy()
    eps = 1e-9

    price_prev = out[price_column].shift(1)
    load_prev = out[load_column].shift(1)

    out["price_over_load"] = price_prev / (load_prev.replace(0, np.nan) + eps)

    roll_med = load_prev.rolling(window_size, min_periods=max(6, window_size // 4)).median()
    roll_mad = load_prev.rolling(window_size, min_periods=max(6, window_size // 4)).apply(
        lambda x: (x - x.median()).abs().median(), raw=False
    )
    load_z = (load_prev - roll_med) / (roll_mad + eps)
    out["price_minus_norm_load"] = price_prev - load_z
    return out


def add_system_imbalance_features(
    df: pd.DataFrame,
    up_reserve_column: str = "rezerwa_mocy_ponad_zapotrzebowanie",
    down_reserve_column: str = "rezerwa_mocy_ponizej_zapotrzebowania",
    load_column: str = "zapotrzebowanie_na_moc",
    price_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    if df.empty:
        return df
    required = {up_reserve_column, down_reserve_column, load_column, price_column}
    if not required.issubset(df.columns):
        return df

    out = df.copy()
    eps = 1e-9

    system_margin_prev = (out[up_reserve_column] - out[down_reserve_column]).shift(1)
    load_prev = out[load_column].shift(1)
    price_prev = out[price_column].shift(1)

    out["sys_margin_MW"] = system_margin_prev
    out["imbalance_ratio"] = system_margin_prev / (load_prev.replace(0, np.nan) + eps)
    out["price_x_imbalance"] = price_prev * out["imbalance_ratio"]
    return out


def add_distribution_shape_features(
    df: pd.DataFrame,
    price_column: str = TARGET_COLUMN,
    window_size: int = 24,
) -> pd.DataFrame:
    if df.empty or price_column not in df.columns:
        return df
    out = df.copy()
    roll = out[price_column].rolling(window_size, min_periods=max(6, window_size // 4))
    out[f"skew_{window_size}h"] = roll.apply(lambda x: skew(x, bias=False), raw=False).shift(1)
    out[f"kurt_{window_size}h"] = roll.apply(lambda x: kurtosis(x, bias=False), raw=False).shift(1)
    return out


def add_renewable_share_features(
    df: pd.DataFrame,
    price_column: str = TARGET_COLUMN,
    wind_generation_column: str = "generacja_zrodel_wiatrowych",
    solar_generation_column: str = "generacja_zrodel_fotowoltaicznych",
    total_generation_column: str = "sumaryczna_generacja_JGxx",
) -> pd.DataFrame:
    needed = {wind_generation_column, solar_generation_column, total_generation_column, price_column}
    if df.empty or not needed.issubset(df.columns):
        return df

    out = df.copy()
    eps = 1e-9

    wind_prev = out[wind_generation_column].shift(1)
    solar_prev = out[solar_generation_column].shift(1)
    total_prev = out[total_generation_column].shift(1)
    price_prev = out[price_column].shift(1)

    renewable_share_prev = (wind_prev + solar_prev) / (total_prev.replace(0, np.nan) + eps)
    out["renew_share"] = renewable_share_prev
    out["price_x_renew_share"] = price_prev * renewable_share_prev
    return out


# ========================= PLOTS =========================

def plot_stl_decomposition(df: pd.DataFrame, target_column: str = TARGET_COLUMN, period: int = 24 * 7) -> None:
    if not SHOW_PLOTS or df.empty or target_column not in df.columns:
        return
    series = df[target_column].dropna()
    if len(series) < period * 2:
        print("[INFO] STL skipped (too short series).")
        return
    stl = STL(series, period=period, robust=True).fit()
    fig = stl.plot()
    plt.suptitle(f"STL decomposition ({period}-period)")
    plt.tight_layout()

    plots_dir = PATHS.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_png = plots_dir / "stl_decomposition.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved STL decomposition plot -> {out_png}")


# ========================= MAIN FLOW =========================

def main() -> None:
    # 1) Load
    df = load_dataset(RAW_DATABASE_PATH, DATETIME_COLUMN_NAME)
    if df.empty:
        print("No input data. Nothing to preprocess.")
        return

    # 2) Basic diagnostics
    print_missing_report(df)
    print_head_and_tail(df)
    print_numeric_summary_table(df)

    # 3) De‑duplicate datetime index
    df = deduplicate_datetime_index(df)

    # 4) Remove trivial problems
    df, _ = drop_zero_variance_features(df)
    df = drop_highly_correlated_features(
        df,
        threshold=HIGH_CORR_THRESHOLD,
        protected_cols=[TARGET_COLUMN],
    )
    print_high_correlation_pairs(df, threshold=HIGH_CORR_THRESHOLD)

    # 5) Optional tests (no mutation)
    run_stationarity_tests(df)

    # 6) First‑pass imputation (for raw features needed in engineering)
    df = impute_missing_values_random_forest(df)

    # 7) Feature engineering (all leakage‑safe)
    df = add_calendar_features(df)
    df = add_season_features(df, add_dummies=True)
    df = add_generation_and_trade_ratios(df)
    df = add_target_rolling_statistics(df)

    df = add_price_volatility_features(df)
    if {"d_price_1h", "d_price_24h", "std_6h", "std_24h"}.issubset(df.columns):
        df[["d_price_1h", "d_price_24h", "std_6h", "std_24h"]] = df[
            ["d_price_1h", "d_price_24h", "std_6h", "std_24h"]
        ].fillna(0)

    df = add_price_demand_relationship_features(df)
    df = add_system_imbalance_features(df)
    df = add_distribution_shape_features(df, window_size=24)
    df = add_renewable_share_features(df)

    # Some rolling columns may still start with NaN, forward‑fill them
    rolling_cols = [c for c in df.columns if c.startswith(f"{TARGET_COLUMN}_roll")]
    if rolling_cols:
        df[rolling_cols] = df[rolling_cols].ffill()

    # 8) Second‑pass imputation (clean up any NaNs from engineering)
    df = impute_missing_values_random_forest(df)

    # 9) Optional STL plot and final reports
    plot_stl_decomposition(df)
    print_numeric_summary_table(df)
    print_missing_report(df)

    # 10) Save final table
    before = len(df)
    cleaned = df.dropna(how="any")
    dropped = before - len(cleaned)
    cleaned.to_parquet(OUTPUT_PARQUET_PATH)
    print(
        f"Saved {len(cleaned):,} rows x {cleaned.shape[1]} columns -> {OUTPUT_PARQUET_PATH}"
        + (f" (dropped {dropped:,} rows with NaN)" if dropped > 0 else "")
    )


if __name__ == "__main__":
    main()

