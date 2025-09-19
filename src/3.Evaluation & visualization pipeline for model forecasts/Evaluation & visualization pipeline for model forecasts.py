"""
Title: Evaluation & visualization pipeline for model forecasts (single predictions file)

What this script does:
- Loads a single predictions table (CSV or Parquet) that contains a timestamp column, the observed target,
  and any number of model prediction columns.
- Cleans & validates the input (parses datetimes, sorts, coerces numeric columns, drops rows with missing
  timestamps or target values).
- Detects model columns automatically (all numeric columns except the timestamp & target) with an optional
  minimum non‑missing coverage filter.
- Computes general metrics (MAE, RMSE, RMSLE, R², MAPE_ε, sMAPE) on: full sample and standard windows
  (24h, 7d, 14d, 31d).
- Computes peak‑aware metrics using either a training threshold or a fallback quantile from current data
  (counts peaks, MAE on peaks/non‑peaks, weighted WMAE, peak precision/recall/F1 by thresholding predictions).
- Runs Diebold‑Mariano tests (MAE & MSE losses) for all model pairs, with HAC (Newey‑West) variance and a
  small‑sample correction; saves raw results and human‑friendly summary tables.
- Produces clear plots for: sliding windows (all models), Observed vs TiDE (with peak markers), Observed vs
  selected models with peak markers.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # formatowanie osi czasu


PROJECT_ROOT: Path = Path(__file__).resolve().parent
RESULTS_DIR: Path = PROJECT_ROOT / "results"
PLOTS_DIR: Path = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE: Path = RESULTS_DIR / "merged_predictions.csv"

DATETIME_COLUMN_NAME: str = "timestamp"
TARGET_COLUMN_NAME: str = "Observed"

# (PL) Parametry metryk i pików
TRAIN_PEAK_THRESHOLD_VALUE: Optional[float] = None  # jeśli znany z treningu; w p.p. użyty będzie kwantyl
PEAK_QUANTILE: float = 0.95                        # fallback: kwantyl do wyznaczenia progu piku
PEAK_WEIGHT: float = 3.0                           # waga pików w WMAE
MAPE_EPSILON: float = 10.0                         # stabilizator mianownika dla MAPE_ε

# (PL) Minimalny udział nie‑NaN w kolumnie modelu, aby ją uwzględnić (np. 50%)
MIN_VALID_RATIO: float = 0.5

# I/O & VALIDATION HELPERS
def _try_read_table(path: Path) -> pd.DataFrame:
    """Load a table from CSV/Parquet by extension."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("INPUT_FILE must be .csv or .parquet")


def load_predictions(path: Path, datetime_col: str, target_col: str) -> pd.DataFrame:
    """Load predictions table, parse datetimes, sort, coerce numerics, drop rows with missing time/target."""
    data = _try_read_table(path)
    if datetime_col not in data.columns or target_col not in data.columns:
        raise ValueError(f"Input must contain columns '{datetime_col}' and '{target_col}'. Columns: {list(data.columns)}")
    data[datetime_col] = pd.to_datetime(data[datetime_col], errors="coerce")
    data = data.sort_values(datetime_col).reset_index(drop=True)
    for col in data.columns:
        if col == datetime_col:
            continue
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=[datetime_col, target_col])
    return data


def list_model_columns(table: pd.DataFrame, datetime_col: str, target_col: str, *, min_valid_ratio: float = MIN_VALID_RATIO) -> List[str]:
    """Return model columns (numeric, not datetime/target) with at least `min_valid_ratio` non‑missing values."""
    candidates = [c for c in table.columns if c not in (datetime_col, target_col)]
    numeric = table[candidates].select_dtypes(include=[np.number])
    valid_counts = numeric.notna().mean(axis=0)
    keep = valid_counts[valid_counts >= min_valid_ratio].index.tolist()
    return keep


def compute_basic_metrics(true_values: pd.Series, predicted_values: pd.Series, mape_epsilon: float = MAPE_EPSILON) -> Dict[str, float]:
    """Compute MAE, RMSE, RMSLE, R², MAPE_ε and sMAPE."""
    y_true = true_values.astype(float)
    y_pred = predicted_values.astype(float)

    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # RMSLE (clip to non‑negative)
    y_true_pos = np.clip(y_true, a_min=0, a_max=None)
    y_pred_pos = np.clip(y_pred, a_min=0, a_max=None)
    rmsle = float(np.sqrt(np.mean((np.log1p(y_true_pos) - np.log1p(y_pred_pos)) ** 2)))

    sse = float(np.sum(errors ** 2))
    sst = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float(1 - sse / sst) if sst > 0 else float("nan")

    mape_eps = float(np.mean(np.abs(errors) / np.maximum(np.abs(y_true), mape_epsilon))) * 100.0
    smape = float(np.mean(np.abs(errors) / ((np.abs(y_true) + np.abs(y_pred)) / 2.0))) * 100.0

    return {"MAE": mae, "RMSE": rmse, "RMSLE": rmsle, "R2": r2, "MAPE_eps": mape_eps, "sMAPE": smape}


def choose_peak_threshold(train_like_series: pd.Series, quantile_value: float) -> float:
    """Fallback peak threshold based on a quantile of the given series."""
    threshold = float(train_like_series.quantile(quantile_value))
    print(f"[INFO] Fallback peak threshold from current data: q={quantile_value:.3f} -> threshold={threshold:.4f}")
    return threshold


def compute_peak_aware_metrics(true_values: pd.Series, predicted_values: pd.Series, threshold: float, peak_weight: float = PEAK_WEIGHT) -> Dict[str, float]:
    """Compute peak vs non‑peak MAE, weighted MAE (WMAE) and peak classification metrics (precision/recall/F1)."""
    y_true = true_values.astype(float)
    y_pred = predicted_values.astype(float)

    is_peak = y_true >= threshold

    def _mae_part(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a - b))) if a.size else float("nan")

    mae_on_peaks = _mae_part(y_true[is_peak].values, y_pred[is_peak].values)
    mae_off_peaks = _mae_part(y_true[~is_peak].values, y_pred[~is_peak].values)

    weights = np.where(is_peak.values, peak_weight, 1.0)
    wmae = float(np.sum(weights * np.abs(y_true.values - y_pred.values)) / np.sum(weights))

    predicted_peak = y_pred >= threshold
    tp = int(np.sum(predicted_peak & is_peak))
    fp = int(np.sum(predicted_peak & (~is_peak)))
    fn = int(np.sum((~predicted_peak) & is_peak))

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision > 0 and recall > 0 else float("nan")

    return {
        "threshold": threshold,
        "n_peaks": int(is_peak.sum()),
        "n_non": int((~is_peak).sum()),
        "MAE_peaks": mae_on_peaks,
        "MAE_nonpeaks": mae_off_peaks,
        "WMAE": wmae,
        "precision_peak": precision,
        "recall_peak": recall,
        "f1_peak": f1,
    }

# =========================== DM TEST ===========================

def _loss_series(true_values: pd.Series, predicted_values: pd.Series, loss: str = "mae") -> np.ndarray:
    """Return elementwise loss series for 'mae' (|e|) or 'mse' (e^2)."""
    residuals = (true_values.values - predicted_values.values).astype(np.float64)
    if loss.lower() in {"mae", "l1"}:
        return np.abs(residuals)
    if loss.lower() in {"mse", "l2"}:
        return residuals ** 2
    raise ValueError("loss must be 'mae' or 'mse'")


def _t_cdf_approx(x_value: float, degrees_freedom: int) -> float:
    """Simple t‑CDF approximation – we use normal approx for df>30; for simplicity we return normal CDF here."""
    return 0.5 * (1.0 + math.erf(x_value / math.sqrt(2)))


def diebold_mariano_test(
    true_values: pd.Series,
    model_predictions: pd.Series,
    reference_predictions: pd.Series,
    forecast_horizon: int = 1,
    loss: str = "mae",
    newey_west_lags: Optional[int] = None,
    use_small_sample_correction: bool = True,
    auto_select_nw: bool = True,
    label_model: str = "ModelA",
    label_reference: str = "ModelB",
) -> Dict[str, float]:
    """Diebold–Mariano test with HAC (Newey–West) variance and optional small‑sample correction."""
    common_index = true_values.index.intersection(model_predictions.index).intersection(reference_predictions.index)
    y_true = true_values.loc[common_index]
    y_m = model_predictions.loc[common_index]
    y_r = reference_predictions.loc[common_index]

    l_m = _loss_series(y_true, y_m, loss=loss)
    l_r = _loss_series(y_true, y_r, loss=loss)
    d = l_m - l_r

    n = d.size
    if n < 5:
        raise ValueError("Not enough observations for DM test (min 5).")

    d_mean = float(np.mean(d))
    d_centered = d - d_mean

    if newey_west_lags is None:
        newey_west_lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))) if auto_select_nw else max(0, forecast_horizon - 1)

    gamma0 = float(np.dot(d_centered, d_centered) / n)
    S = gamma0
    for k in range(1, newey_west_lags + 1):
        cov_k = float(np.dot(d_centered[k:], d_centered[:-k]) / n)
        weight = 1.0 - k / (newey_west_lags + 1.0)
        S += 2.0 * weight * cov_k

    var_dbar = S / n
    if var_dbar <= 0 or not np.isfinite(var_dbar):
        raise ValueError("Invalid DM variance estimate (<=0 or non‑finite).")

    dm_raw = d_mean / math.sqrt(var_dbar)

    if use_small_sample_correction:
        kappa = math.sqrt((n + 1 - 2 * forecast_horizon + (forecast_horizon * (forecast_horizon - 1)) / n) / n)
        dm_used = dm_raw * kappa
        df = n - 1
        p_norm = 2.0 * (1.0 - 0.5 * (1 + math.erf(abs(dm_used) / math.sqrt(2))))
        p_t = 2.0 * (1.0 - _t_cdf_approx(abs(dm_used), df))
    else:
        dm_used = dm_raw
        p_norm = 2.0 * (1.0 - 0.5 * (1 + math.erf(abs(dm_used) / math.sqrt(2))))
        p_t = float("nan")

    better_label = label_model if d_mean < 0 else label_reference

    print("\n=== Diebold–Mariano test ===")
    print(f"Loss: {loss.upper()} | h={forecast_horizon} | NW lags={newey_west_lags} | small_sample={use_small_sample_correction} | auto_nw={auto_select_nw}")
    print(f"Compared: {label_model} vs {label_reference} (d_t = L_model - L_ref)")
    print(f"N: {n}")
    print(f"mean(d_t): {d_mean:.6f} (negative => {label_model} better)")
    print(f"DM stat: {dm_used:.4f} (raw: {dm_raw:.4f})")
    print(f"p-value ~N(0,1): {p_norm:.4f}")
    if use_small_sample_correction:
        print(f"p-value ~t(df={n-1}): {p_t:.4f}")
    print(f"Better by mean loss: {better_label}")
    print("========================================\n")

    return {
        "n": n,
        "h": forecast_horizon,
        "loss": loss,
        "nw_lags": newey_west_lags,
        "dm_raw": dm_raw,
        "dm": dm_used,
        "p_norm": p_norm,
        "p_t": p_t,
        "mean_diff": d_mean,
        "better": better_label,
    }


def _compute_loss_by_model(data_slice: pd.DataFrame, target_col: str, model_columns: List[str], loss: str = "mae") -> Dict[str, float]:
    """Average loss for each model in a window (MAE/MSE). Used for Δ% vs reference in DM table."""
    loss = loss.lower()
    out: Dict[str, float] = {}
    for m in model_columns:
        pair = data_slice[[target_col, m]].dropna()
        if pair.shape[0] < 5:
            out[m] = float("nan")
            continue
        y_true = pair[target_col].astype(float).values
        y_pred = pair[m].astype(float).values
        err = y_true - y_pred
        if loss in {"mae", "l1"}:
            out[m] = float(np.mean(np.abs(err)))
        elif loss in {"mse", "l2"}:
            out[m] = float(np.mean(err ** 2))
        else:
            raise ValueError("loss must be 'mae' or 'mse'")
    return out


def _sig_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return ""


def build_dm_summary_table(dm_df: pd.DataFrame, loss_map_mae: Dict[str, float], loss_map_mse: Dict[str, float], prefer_loss: str = "MAE", keep_pairs: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
    """Readable DM table with Δabs and Δ% vs the reference model in each pair; optional filtering by key pairs."""
    df = dm_df.copy()
    df["loss"] = df["loss"].str.upper()

    lhs, rhs = [], []
    for s in df["pair"].astype(str).tolist():
        parts = s.split(" vs ")
        lhs.append(parts[0]); rhs.append(parts[1] if len(parts) > 1 else "")
    df["lhs"], df["rhs"] = lhs, rhs

    df["delta_abs"] = df["mean_diff"].abs().astype(float)

    def _ref_loss(row: pd.Series) -> float:
        return float((loss_map_mae if row["loss"] == "MAE" else loss_map_mse).get(row["rhs"], np.nan))

    df["ref_loss"] = df.apply(_ref_loss, axis=1)

    def _delta_pct(row: pd.Series) -> float:
        ref = row["ref_loss"]
        if not np.isfinite(ref) or ref == 0.0:
            return float("nan")
        return float(100.0 * row["delta_abs"] / ref)

    df["delta_pct_vs_ref"] = df.apply(_delta_pct, axis=1)

    df["winner"] = df["better"]
    df["loser"] = np.where(df["better"] == df["lhs"], df["rhs"], df["lhs"])
    df["sig"] = df["p_t"].apply(_sig_stars)

    def _effect_str(row: pd.Series) -> str:
        return f"{row['delta_abs']:.2f} ({row['delta_pct_vs_ref']:.1f}%)" if np.isfinite(row["delta_pct_vs_ref"]) else f"{row['delta_abs']:.2f}"

    df["effect"] = df.apply(_effect_str, axis=1)

    if keep_pairs:
        keep_set = {f"{a} vs {b}" for a, b in keep_pairs}
        df = df[df["pair"].isin(keep_set)].copy()

    cols = [
        "pair", "loss", "winner", "mean_diff", "effect", "dm", "p_t", "sig",
        "delta_abs", "delta_pct_vs_ref", "ref_loss", "nw_lags", "n", "h",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    df["loss_sort"] = np.where(df["loss"] == prefer_loss.upper(), 0, 1)
    df = df.sort_values(["loss_sort", "p_t", "delta_abs"], ascending=[True, True, False]).drop(columns=["loss_sort"])

    for c in ["mean_diff", "dm", "p_t", "delta_abs", "delta_pct_vs_ref", "ref_loss"]:
        if c in df.columns:
            df[c] = df[c].astype(float).round(4)
    return df

# ======================= TIME WINDOW HELPERS =======================

def slice_last_hours(data: pd.DataFrame, datetime_col: str, hours: int) -> pd.DataFrame:
    """Return the last `hours` of data, sorted by time."""
    if data.empty:
        return data.copy()
    end_time = data[datetime_col].max()
    start_time = end_time - pd.Timedelta(hours=hours)
    out = data[data[datetime_col] > start_time].copy()
    return out.sort_values(datetime_col).reset_index(drop=True)

# ============================== PLOTS ===============================

def beautify_time_axis_numeric(ax: plt.Axes, n_days: int, tz: Optional[str] = None) -> None:
    """Clean numeric time axis (YYYY‑MM‑DD); spacing adjusts for 31 days vs shorter windows."""
    if n_days >= 28:
        major = mdates.DayLocator(interval=3, tz=tz)
        minor = mdates.DayLocator(interval=1, tz=tz)
    else:
        major = mdates.AutoDateLocator(minticks=6, maxticks=10, tz=tz)
        minor = mdates.AutoDateLocator(minticks=12, maxticks=20, tz=tz)

    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_minor_locator(minor)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d", tz=tz))

    for label in ax.get_xticklabels(which="major"):
        label.set_rotation(30)
        label.set_ha("right")
    ax.margins(x=0.01)


def plot_window_all_models(data: pd.DataFrame, datetime_col: str, target_col: str, model_columns: List[str], hours: int, title: Optional[str] = None, save_path: str = "results/plots/plot_window_all_models.png", aggregation_hours: Optional[int] = None) -> None:
    """Plot Observed vs all model predictions over the last `hours` window; optional aggregation by resampling."""
    data_slice = slice_last_hours(data, datetime_col, hours=hours)
    if aggregation_hours is not None and aggregation_hours > 1:
        df = data_slice.set_index(datetime_col)
        data_slice = df.resample(f"{aggregation_hours}H").mean(numeric_only=True).dropna().reset_index()

    if data_slice.empty or data_slice.shape[0] < 5:
        print(f"[WARN] Not enough data for plot ({hours}h).")
        return

    # Compute MAE for legend labels
    mae_by_model: Dict[str, float] = {}
    for model_name in model_columns:
        pair = data_slice[[target_col, model_name]].dropna()
        if pair.shape[0] < 5:
            mae_by_model[model_name] = float("nan")
        else:
            err = np.abs(pair[target_col].astype(float) - pair[model_name].astype(float))
            mae_by_model[model_name] = float(err.mean())

    # Color policy: TiDE = red, others cycle predefined palette
    manual_colors: Dict[str, str] = {}
    base_colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'black']
    color_iter = iter(base_colors)
    for model_name in model_columns:
        manual_colors[model_name] = 'red' if model_name.lower() == 'tide' else next(color_iter, None)

    # Sort models by MAE ascending for plotting order
    mae_sorted = sorted(mae_by_model.items(), key=lambda x: x[1] if np.isfinite(x[1]) else float('inf'))

    plt.figure(figsize=(14, 6))
    plt.plot(data_slice[datetime_col], data_slice[target_col], label="Observed", linewidth=3)

    for model_name, mae_value in mae_sorted:
        label_mae = f"{model_name} (MAE={mae_value:.2f})" if np.isfinite(mae_value) else f"{model_name} (MAE=NA)"
        plt.plot(
            data_slice[datetime_col],
            data_slice[model_name],
            label=label_mae,
            linewidth=2.0 if model_name.lower() == "tide" else 1.3,
            color=manual_colors.get(model_name, None),
        )

    if title is None:
        title = f"Last {hours} hours: model forecasts"
    # plt.title(title)
    plt.xlabel("Time"); plt.ylabel("Value"); plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=9, frameon=True)
    beautify_time_axis_numeric(plt.gca(), n_days=int(hours // 24))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


def plot_observed_vs_models_with_peaks(data: pd.DataFrame, datetime_col: str, target_col: str, model_cols: Optional[Sequence[str]] = None, days: int = 31, quantile: float = 0.95, aggregation_hours: int = 2, title: Optional[str] = None, save_path: Optional[str] = None) -> Tuple[float, int]:
    """Plot Observed and selected models; mark observed peaks (>= given quantile). Returns (threshold, n_peaks)."""
    if model_cols is None:
        model_cols = ("TiDE", "XGB")

    hours = 24 * days
    data_slice = slice_last_hours(data, datetime_col, hours=hours)
    if data_slice.empty or data_slice.shape[0] < 5:
        print(f"[WARN] Not enough data for plot ({days}d).")
        return float("nan"), 0

    if aggregation_hours is not None and aggregation_hours > 1:
        df = data_slice.set_index(datetime_col)
        keep_cols = [c for c in [target_col] + list(model_cols) if c in df.columns]
        missing = [c for c in [target_col] + list(model_cols) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        df = df[keep_cols]
        data_slice = df.resample(f"{aggregation_hours}H").mean(numeric_only=True).dropna().reset_index()

    threshold = float(data_slice[target_col].quantile(quantile))
    is_peak = data_slice[target_col] >= threshold
    n_peaks = int(is_peak.sum())

    plt.figure(figsize=(14, 6))
    plt.plot(data_slice[datetime_col], data_slice[target_col], label="Observed", linewidth=3.5)
    plt.scatter(data_slice.loc[is_peak, datetime_col], data_slice.loc[is_peak, target_col], s=28, marker="o", color="red", label=f"Peaks (q={quantile:.2f})")

    for m in model_cols:
        if m not in data_slice.columns:
            raise ValueError(f"Column '{m}' not found in data.")
        lw = 2.2 if m.lower() == "tide" else 1.6
        plt.plot(data_slice[datetime_col], data_slice[m], label=m, linewidth=lw)

    if title is None:
        title = f"Observed vs {', '.join(model_cols)} | last {days} days (agg {aggregation_hours}h)"
    # plt.title(title)
    plt.xlabel("Time"); plt.ylabel("Value"); plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=9, frameon=True)
    beautify_time_axis_numeric(plt.gca(), n_days=days)

    if save_path is None:
        save_path = str(PLOTS_DIR / f"plot_obs_vs_models_{days}d_agg{aggregation_hours}h.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()

    print(f"[INFO] Peaks threshold={threshold:.4f}, n_peaks={n_peaks}, saved: {save_path}")
    return threshold, n_peaks


def plot_observed_vs_tide_with_peaks(data: pd.DataFrame, datetime_col: str, target_col: str, tide_col: str = "TiDE", days: int = 31, quantile: float = 0.95, aggregation_hours: int = 2, title: Optional[str] = None, save_path: Optional[str] = None) -> Tuple[float, int]:
    """Plot Observed vs TiDE with observed peaks highlighted. Returns (threshold, n_peaks)."""
    hours = 24 * days
    data_slice = slice_last_hours(data, datetime_col, hours=hours)
    if data_slice.empty or data_slice.shape[0] < 5:
        print(f"[WARN] Not enough data for plot ({days}d).")
        return float("nan"), 0

    if aggregation_hours is not None and aggregation_hours > 1:
        df = data_slice.set_index(datetime_col)
        keep_cols = [c for c in [target_col, tide_col] if c in df.columns]
        if len(keep_cols) < 2:
            raise ValueError(f"Columns '{target_col}' and '{tide_col}' must exist in data.")
        df = df[keep_cols]
        data_slice = df.resample(f"{aggregation_hours}H").mean(numeric_only=True).dropna().reset_index()

    if tide_col not in data_slice.columns:
        raise ValueError(f"Column '{tide_col}' not found. Available: {list(data_slice.columns)}")

    threshold = float(data_slice[target_col].quantile(quantile))
    is_peak = data_slice[target_col] >= threshold
    n_peaks = int(is_peak.sum())

    plt.figure(figsize=(14, 6))
    plt.plot(data_slice[datetime_col], data_slice[target_col], label="Observed", linewidth=3)
    plt.plot(data_slice[datetime_col], data_slice[tide_col], label="TiDE", linewidth=2)
    plt.scatter(data_slice.loc[is_peak, datetime_col], data_slice.loc[is_peak, target_col], s=28, marker="o", color="red", label=f"Peaks (q={quantile:.2f})")

    if title is None:
        title = f"Observed vs TiDE | last {days} days (agg {aggregation_hours}h)"
    # plt.title(title)
    plt.xlabel("Time"); plt.ylabel("Value"); plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=9, frameon=True)
    beautify_time_axis_numeric(plt.gca(), n_days=days)

    if save_path is None:
        save_path = str(PLOTS_DIR / f"plot_tide_vs_observed_{days}d_agg{aggregation_hours}h.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()

    print(f"[INFO] Peaks threshold={threshold:.4f}, n_peaks={n_peaks}, saved: {save_path}")
    return threshold, n_peaks

# ====================== GENERAL METRICS BY WINDOW ======================

def compute_general_metrics_for_window(data: pd.DataFrame, datetime_col: str, target_col: str, model_columns: List[str], hours: int, label: str, save_path: Path, mape_epsilon: float, print_table: bool = True) -> pd.DataFrame:
    """Compute & save general metrics for the last `hours` window."""
    data_slice = slice_last_hours(data, datetime_col, hours=hours)
    if data_slice.shape[0] < 5:
        print(f"\n=== {label}: not enough observations ({data_slice.shape[0]}) ===")
        return pd.DataFrame()

    rows: List[Dict[str, float]] = []
    for model_name in model_columns:
        pair = data_slice[[target_col, model_name]].dropna()
        metrics = {"MAE": np.nan, "RMSE": np.nan, "RMSLE": np.nan, "R2": np.nan, "MAPE_eps": np.nan, "sMAPE": np.nan}
        if pair.shape[0] >= 5:
            metrics = compute_basic_metrics(pair[target_col], pair[model_name], mape_epsilon=mape_epsilon)
        rows.append({"model": model_name, **metrics})

    result_df = pd.DataFrame(rows).set_index("model").sort_values("MAE")

    if print_table:
        print(f"\n=== {label}: General metrics (lower = better) ===")
        print(result_df.round(4).to_string())

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(save_path)
    return result_df


def compute_standard_windows_general_metrics(data: pd.DataFrame, datetime_col: str, target_col: str, model_columns: List[str], out_dir: Path = RESULTS_DIR, mape_epsilon: float = MAPE_EPSILON, print_tables: bool = True) -> Dict[str, pd.DataFrame]:
    """Compute general metrics for four standard windows and save to CSV."""
    windows = [
        ("24h", 24, "Last 24h"),
        ("7d", 24 * 7, "Last 7 days"),
        ("14d", 24 * 14, "Last 14 days"),
        ("31d", 24 * 31, "Last 31 days"),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, pd.DataFrame] = {}
    for tag, hrs, lbl in windows:
        save_path = out_dir / f"general_metrics_{tag}.csv"
        outputs[tag] = compute_general_metrics_for_window(
            data=data,
            datetime_col=datetime_col,
            target_col=target_col,
            model_columns=model_columns,
            hours=hrs,
            label=lbl,
            save_path=save_path,
            mape_epsilon=mape_epsilon,
            print_table=print_tables,
        )
    return outputs

# ========================= DM TESTS OVER WINDOWS =========================

def run_dm_tests_for_window(data_slice: pd.DataFrame, model_columns: List[str], target_col: str, label_suffix: str) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """Run MAE/MSE DM tests for all model pairs in the given slice; save raw CSV and return (df, mae_map, mse_map)."""
    if data_slice.shape[0] < 5:
        print(f"\n=== DM tests {label_suffix}: not enough observations ({data_slice.shape[0]}) ===")
        return pd.DataFrame(), {}, {}

    mae_map = _compute_loss_by_model(data_slice, target_col, model_columns, loss="mae")
    mse_map = _compute_loss_by_model(data_slice, target_col, model_columns, loss="mse")

    pairs: List[Tuple[str, str]] = []
    for i in range(len(model_columns)):
        for j in range(i + 1, len(model_columns)):
            pairs.append((model_columns[i], model_columns[j]))

    dm_results: List[Dict[str, float]] = []
    for a, b in pairs:
        res_mae = diebold_mariano_test(data_slice[target_col], data_slice[a], data_slice[b], forecast_horizon=1, loss="mae", newey_west_lags=None, use_small_sample_correction=True, auto_select_nw=True, label_model=a, label_reference=b)
        res_mse = diebold_mariano_test(data_slice[target_col], data_slice[a], data_slice[b], forecast_horizon=1, loss="mse", newey_west_lags=None, use_small_sample_correction=True, auto_select_nw=True, label_model=a, label_reference=b)
        dm_results.append({"pair": f"{a} vs {b}", "loss": "MAE", **res_mae})
        dm_results.append({"pair": f"{a} vs {b}", "loss": "MSE", **res_mse})

    dm_df = pd.DataFrame(dm_results)
    out_path = RESULTS_DIR / f"dm_tests_{label_suffix}.csv"
    dm_df.to_csv(out_path, index=False)
    print(f"\n=== DM tests saved to {out_path.name} ===")
    return dm_df, mae_map, mse_map

# ================================ MAIN ================================

def main() -> None:
    """Main flow: load data, compute metrics, thresholds, plots, and DM tests."""
    # 1) Load predictions + detect models
    data = load_predictions(INPUT_FILE, DATETIME_COLUMN_NAME, TARGET_COLUMN_NAME)
    model_columns = list_model_columns(data, DATETIME_COLUMN_NAME, TARGET_COLUMN_NAME, min_valid_ratio=MIN_VALID_RATIO)
    if not model_columns:
        raise RuntimeError("No valid model columns detected (check numeric types and MIN_VALID_RATIO).")
    print(f"Loaded {len(data):,} rows. Models: {', '.join(model_columns)}")

    # 2) General metrics on full sample
    overall_rows: List[Dict[str, float]] = []
    for model_name in model_columns:
        metrics = compute_basic_metrics(data[TARGET_COLUMN_NAME], data[model_name], mape_epsilon=MAPE_EPSILON)
        overall_rows.append({"model": model_name, **metrics})
    overall_df = pd.DataFrame(overall_rows).set_index("model").sort_values("MAE")
    print("\n=== General metrics on full sample ===")
    print(overall_df.round(4).to_string())
    overall_df.to_csv(RESULTS_DIR / "general_metrics_full.csv")

    # 3) Peak threshold (training value or fallback quantile)
    if TRAIN_PEAK_THRESHOLD_VALUE is not None:
        peak_threshold = float(TRAIN_PEAK_THRESHOLD_VALUE)
        print(f"\n[INFO] Using TRAIN peak threshold: {peak_threshold:.4f}")
    else:
        peak_threshold = choose_peak_threshold(data[TARGET_COLUMN_NAME], quantile_value=PEAK_QUANTILE)

    # 4) Peak‑aware metrics on full sample
    peak_rows: List[Dict[str, float]] = []
    for model_name in model_columns:
        peak = compute_peak_aware_metrics(data[TARGET_COLUMN_NAME], data[model_name], threshold=peak_threshold, peak_weight=PEAK_WEIGHT)
        peak_rows.append({"model": model_name, **peak})
    peak_df = pd.DataFrame(peak_rows).set_index("model")
    print("\n=== Peak-aware metrics on full sample ===")
    print(peak_df[["n_peaks", "n_non", "MAE_peaks", "MAE_nonpeaks", "WMAE", "precision_peak", "recall_peak", "f1_peak"]].round(4).to_string())

    # 5) Time windows
    last_7d_slice = slice_last_hours(data, DATETIME_COLUMN_NAME, hours=24 * 7)
    last_14d_slice = slice_last_hours(data, DATETIME_COLUMN_NAME, hours=24 * 14)
    last_31d_slice = slice_last_hours(data, DATETIME_COLUMN_NAME, hours=24 * 31)

    # 6) Save peak tables per window (CSV)
    peak_tables = {"full_sample": peak_df}

    m7_gen, m7_peak = compute_tables_for_window(
        data_slice=last_7d_slice,
        model_columns=model_columns,
        target_col=TARGET_COLUMN_NAME,
        mape_epsilon=MAPE_EPSILON,
        peak_threshold=peak_threshold,
        peak_weight=PEAK_WEIGHT,
        label="Last 7 days (168h)",
        out_prefix=str(RESULTS_DIR / "metrics_last7d"),
    )
    if not m7_peak.empty:
        peak_tables["last_7d"] = m7_peak

    m14_gen, m14_peak = compute_tables_for_window(
        data_slice=last_14d_slice,
        model_columns=model_columns,
        target_col=TARGET_COLUMN_NAME,
        mape_epsilon=MAPE_EPSILON,
        peak_threshold=peak_threshold,
        peak_weight=PEAK_WEIGHT,
        label="Last 14 days (336h)",
        out_prefix=str(RESULTS_DIR / "metrics_last14d"),
    )
    if not m14_peak.empty:
        peak_tables["last_14d"] = m14_peak

    m31_gen, m31_peak = compute_tables_for_window(
        data_slice=last_31d_slice,
        model_columns=model_columns,
        target_col=TARGET_COLUMN_NAME,
        mape_epsilon=MAPE_EPSILON,
        peak_threshold=peak_threshold,
        peak_weight=PEAK_WEIGHT,
        label="Last 31 days (744h)",
        out_prefix=str(RESULTS_DIR / "metrics_last31d"),
    )
    if not m31_peak.empty:
        peak_tables["last_31d"] = m31_peak

    for label, df in peak_tables.items():
        out_path = RESULTS_DIR / f"peak_metrics_{label}.csv"
        df.to_csv(out_path)
        print(f"Saved: {out_path.as_posix()}")

    # 7) Plots
    plot_standard_windows_all_models(
        data=data,
        datetime_col=DATETIME_COLUMN_NAME,
        target_col=TARGET_COLUMN_NAME,
        model_columns=model_columns,
        out_dir=str(PLOTS_DIR),
    )

    _ = compute_standard_windows_general_metrics(
        data=data,
        datetime_col=DATETIME_COLUMN_NAME,
        target_col=TARGET_COLUMN_NAME,
        model_columns=model_columns,
        out_dir=RESULTS_DIR,
        mape_epsilon=MAPE_EPSILON,
    )

    dm_7d, mae_map_7d, mse_map_7d = run_dm_tests_for_window(
        data_slice=last_7d_slice,
        model_columns=model_columns,
        target_col=TARGET_COLUMN_NAME,
        label_suffix="last7d",
    )

    dm_full, mae_map_full, mse_map_full = run_dm_tests_for_window(
        data_slice=data,
        model_columns=model_columns,
        target_col=TARGET_COLUMN_NAME,
        label_suffix="full",
    )

    if not dm_full.empty:
        key_pairs = [
            ("TiDE", "XGB"),
            ("TiDE", "GBR"),
            ("XGB", "GBR"),
            ("TiDE", "Seq2Seq"),
            ("TiDE", "Transformer"),
            ("TiDE", "AutoML"),
        ]
        (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
        dm_summary_full = build_dm_summary_table(dm_full, mae_map_full, mse_map_full, prefer_loss="MAE", keep_pairs=None)
        dm_summary_key = build_dm_summary_table(dm_full, mae_map_full, mse_map_full, prefer_loss="MAE", keep_pairs=key_pairs)
        dm_summary_full.to_csv(RESULTS_DIR / "tables" / "dm_summary_full.csv", index=False)
        dm_summary_key.to_csv(RESULTS_DIR / "tables" / "dm_summary_keypairs.csv", index=False)
        print("\n=== Diebold–Mariano (full) – MAE first (top 12) ===")
        print(dm_summary_full.head(12).to_string(index=False))
        print("\n=== Diebold–Mariano (full) – KEY PAIRS (MAE first) ===")
        print(dm_summary_key.to_string(index=False))

    plot_window_all_models(
        data=data,
        datetime_col=DATETIME_COLUMN_NAME,
        target_col=TARGET_COLUMN_NAME,
        model_columns=model_columns,
        hours=24 * 31,
        title="Last 31 days (aggregated 12h): model forecasts",
        save_path=str(PLOTS_DIR / "plot_31d_agg12h_all_models.png"),
        aggregation_hours=12,
    )

    _ = plot_observed_vs_tide_with_peaks(
        data=data,
        datetime_col=DATETIME_COLUMN_NAME,
        target_col=TARGET_COLUMN_NAME,
        tide_col="TiDE",
        days=31,
        quantile=0.95,
        aggregation_hours=2,
        title="Observed vs TiDE with peak markers (31 days, 2h mean)",
        save_path=str(PLOTS_DIR / "plot_tide_vs_observed_31d_agg2h.png"),
    )

    _ = plot_observed_vs_models_with_peaks(
        data=data,
        datetime_col=DATETIME_COLUMN_NAME,
        target_col=TARGET_COLUMN_NAME,
        model_cols=["TiDE", "Seq2Seq"],
        days=31,
        quantile=0.95,
        aggregation_hours=2,
        title="Observed vs TiDE & Seq2Seq with peaks (31 days, 2h mean)",
        save_path=str(PLOTS_DIR / "plot_obs_vs_TiDE_Seq2Seq_31d_agg2h.png"),
    )

if __name__ == "__main__":
    main()
