# -*- coding: utf-8 -*-
"""
Evaluation pipeline

This script evaluates a single predictions table. It detects model columns,
computes peak-aware metrics at multiple thresholds, runs Diebold-Mariano tests
with HAC (Newey West) and small-sample correction, prints all results to the
console, writes CSVs, and plots the last-month view with observed values, peak
markers, and the top-2 models by peak recall (tie-break by lower WMAE).

Outputs
- results/tables/peak_metrics_<stem>.csv
- results/tables/dm_raw_<stem>.csv
- results/tables/dm_summary_<stem>.csv
- results/plots/plot_peaks_month_<stem>.png
"""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SciPy is optional. If missing, p-values fall back to normal approx.
try:
    from scipy.stats import t as student_t
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ========================= PATHS BOOTSTRAP =========================

def bootstrap_paths_models() -> SimpleNamespace:
    """Locate project root and prepare common paths."""
    try:
        here = Path(__file__).resolve()
    except NameError:
        here = Path.cwd().resolve()

    root = None
    # Find a repo folder that surely exists
    for p in [here] + list(here.parents):
        if (p / "data" / "merged predictions").exists() or (p / "data" / "processed").exists():
            root = p
            break

    if root is None:
        env = os.environ.get("PROJECT_ROOT")
        if env and ((Path(env) / "data" / "merged predictions").exists() or (Path(env) / "data" / "processed").exists()):
            root = Path(env).resolve()

    if root is None:
        # Safe fallback two levels up
        root = here.parent.parent

    data_dir = root / "data"
    merged_predictions_dir = data_dir / "merged predictions"
    processed_dir = data_dir / "processed"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    return SimpleNamespace(
        project_root=root,
        data_dir=data_dir,
        processed_dir=processed_dir,
        merged_predictions_dir=merged_predictions_dir,
        results_dir=results_dir,
    )


PATHS = bootstrap_paths_models()


# ========================= SETTINGS =========================

DATETIME_COLUMN_NAME = "timestamp"
TARGET_COLUMN_NAME = "Observed"

MIN_VALID_RATIO = 0.5              # minimal non-missing fraction for model columns

TRAIN_PEAK_THRESHOLD_VALUE = None  # set float to override quantile-based threshold
PEAK_WEIGHT = 3.0
MAPE_EPSILON = 10.0
PEAK_QUANTILES = [0.95, 0.92, 0.89]
PLOT_QUANTILE = PEAK_QUANTILES[0]

# Default input file stem; can override with env var PREDICTIONS_STEM
DEFAULT_PREDICTIONS_STEM = os.environ.get("PREDICTIONS_STEM", "recursive_blocks_h168x10")

# Output folders
TABLES_DIR = PATHS.results_dir / "tables"
PLOTS_DIR = PATHS.results_dir / "plots"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# --- logging and visualization toggles ---
SHOW_PLOTS = True            # show figure windows after saving
SHOW_PLOTS_BLOCK = False     # block on plt.show
PRINT_FULL_TABLE = True      # print full tables in the console
PRINT_ROWS_LIMIT = None      # None prints all rows, or set an integer
ROUND_LOG_DECIMALS = 4       # numeric display precision in logs


# ========================= GENERIC HELPERS =========================

def _ts() -> str:
    """Timestamp for fallback filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_to_csv(df: pd.DataFrame, path: Path, **kwargs) -> Path:
    """Save DataFrame with a fallback name when the file is locked."""
    path = Path(path)
    try:
        df.to_csv(path, **kwargs)
        print(f"[SAVE] {path.as_posix()}")
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{_ts()}{path.suffix}")
        df.to_csv(alt, **kwargs)
        print(f"[WARN] Locked file. Saved to: {alt.as_posix()}")
        return alt


def safe_savefig(path: Path, **kwargs) -> Path:
    """Save figure with a fallback name when the file is locked."""
    path = Path(path)
    try:
        plt.savefig(path, **kwargs)
        print(f"[SAVE] {path.as_posix()}")
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{_ts()}{path.suffix}")
        plt.savefig(alt, **kwargs)
        print(f"[WARN] Locked file. Saved to: {alt.as_posix()}")
        return alt


def print_table(df: pd.DataFrame, title: str) -> None:
    """Pretty print a DataFrame to the console according to toggles."""
    with pd.option_context(
        "display.max_rows", None if PRINT_ROWS_LIMIT is None else PRINT_ROWS_LIMIT,
        "display.max_columns", None,
        "display.width", 2000,
        "display.max_colwidth", None,
        "display.float_format", (lambda x: f"{x:.{ROUND_LOG_DECIMALS}f}"),
    ):
        print(f"\n=== {title} ===")
        if PRINT_FULL_TABLE or PRINT_ROWS_LIMIT is not None:
            if PRINT_ROWS_LIMIT is None:
                print(df.to_string(index=True))
            else:
                print(df.head(PRINT_ROWS_LIMIT).to_string(index=True))
        else:
            print(df.head(10).to_string(index=True))


# ========================= INPUT HELPERS =========================

def resolve_predictions_file(base_dir: Path, preferred_stem: str) -> Path:
    """Find predictions file by stem; support CSV and Parquet."""
    candidates = [
        base_dir / f"{preferred_stem}.csv",
        base_dir / f"{preferred_stem}.parquet",
        base_dir / f"{preferred_stem}.pq",
        base_dir / f"{preferred_stem}_DP.csv",
        base_dir / f"{preferred_stem}_DP.parquet",
        base_dir / f"{preferred_stem}_DP.pq",
    ]
    for c in candidates:
        if c.exists():
            return c
    matches = sorted(base_dir.glob(f"{preferred_stem}*.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find '{preferred_stem}*' in: {base_dir.as_posix()}")


def _read_csv_or_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path.as_posix()}")
    sfx = path.suffix.lower()
    if sfx == ".csv":
        return pd.read_csv(path)
    if sfx in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Input must be .csv or .parquet")


def load_predictions(path: Path, datetime_col: str, target_col: str) -> pd.DataFrame:
    """Parse timestamp, sort, coerce numerics, drop rows with missing time or target."""
    df = _read_csv_or_parquet(path).copy()
    if datetime_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Input must contain '{datetime_col}' and '{target_col}'. Columns: {list(df.columns)}")
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.dropna(subset=[datetime_col, target_col]).sort_values(datetime_col).reset_index(drop=True)
    for col in df.columns:
        if col != datetime_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def list_model_columns(table: pd.DataFrame,
                       datetime_col: str,
                       target_col: str,
                       *,
                       min_valid_ratio: float) -> List[str]:
    """Numeric columns except datetime and target; filtered by coverage."""
    candidates = [c for c in table.columns if c not in (datetime_col, target_col)]
    numeric = table[candidates].select_dtypes(include=[np.number])
    coverage = numeric.notna().mean(axis=0)
    keep = coverage[coverage >= min_valid_ratio].index.tolist()
    return keep


# ========================= METRICS =========================

def compute_peak_aware_metrics(
    true_values: pd.Series,
    predicted_values: pd.Series,
    threshold: float,
    peak_weight: float = PEAK_WEIGHT,
) -> Dict[str, float]:
    """Peak vs non-peak MAE, WMAE, and peak classification metrics."""
    y_true = true_values.astype(float)
    y_pred = predicted_values.astype(float)
    is_peak = y_true >= threshold

    def _mae(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a - b))) if a.size else float("nan")

    mae_peaks = _mae(y_true[is_peak].values, y_pred[is_peak].values)
    mae_non = _mae(y_true[~is_peak].values, y_pred[~is_peak].values)

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
        "threshold": float(threshold),
        "n_peaks": int(is_peak.sum()),
        "n_non": int((~is_peak).sum()),
        "MAE_peaks": mae_peaks,
        "MAE_nonpeaks": mae_non,
        "WMAE": wmae,
        "precision_peak": precision,
        "recall_peak": recall,
        "f1_peak": f1,
    }


def choose_peak_threshold(train_like_series: pd.Series, quantile_value: float) -> float:
    """Fallback threshold: quantile from current data."""
    thr = float(train_like_series.quantile(quantile_value))
    print(f"[INFO] Peak threshold from data q={quantile_value:.3f}: {thr:.4f}")
    return thr


# ========================= DIEBOLD-MARIANO (HAC) =========================

def _loss_series(y_true: pd.Series, y_pred: pd.Series, loss: str) -> np.ndarray:
    residuals = (y_true.values - y_pred.values).astype(np.float64)
    if loss.lower() in {"mae", "l1"}:
        return np.abs(residuals)
    if loss.lower() in {"mse", "l2"}:
        return residuals ** 2
    raise ValueError("loss must be 'mae' or 'mse'")


def _t_cdf(x_value: float, df: int) -> float:
    if _HAVE_SCIPY and df > 0:
        return float(student_t.cdf(x_value, df=df))
    # Normal approximation
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
    """DM test with HAC (Newey West) and optional small-sample correction."""
    idx = true_values.index.intersection(model_predictions.index).intersection(reference_predictions.index)
    y_true = true_values.loc[idx]
    y_m = model_predictions.loc[idx]
    y_r = reference_predictions.loc[idx]

    d = _loss_series(y_true, y_m, loss) - _loss_series(y_true, y_r, loss)
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
        raise ValueError("Invalid DM variance estimate.")

    dm_raw = d_mean / math.sqrt(var_dbar)

    if use_small_sample_correction:
        kappa = math.sqrt((n + 1 - 2 * forecast_horizon + (forecast_horizon * (forecast_horizon - 1)) / n) / n)
        dm = dm_raw * kappa
        df = n - 1
        p_norm = 2.0 * (1.0 - 0.5 * (1 + math.erf(abs(dm) / math.sqrt(2))))
        p_t = 2.0 * (1.0 - _t_cdf(abs(dm), df))
    else:
        dm = dm_raw
        p_norm = 2.0 * (1.0 - 0.5 * (1 + math.erf(abs(dm) / math.sqrt(2))))
        p_t = float("nan")

    better = label_model if d_mean < 0 else label_reference

    print("\n=== Diebold-Mariano ===")
    print(f"Pair: {label_model} vs {label_reference} | Loss={loss.upper()} | N={n} | NW lags={newey_west_lags}")
    print(f"mean(diff)={d_mean:.6f} | DM={dm:.4f} | p_t={p_t:.4f} | better={better}")

    return {
        "pair": f"{label_model} vs {label_reference}",
        "loss": loss.upper(),
        "n": n,
        "h": forecast_horizon,
        "nw_lags": int(newey_west_lags),
        "dm_raw": float(dm_raw),
        "dm": float(dm),
        "p_norm": float(p_norm),
        "p_t": float(p_t),
        "mean_diff": float(d_mean),
        "better": better,
    }


def run_dm_for_all_pairs(df: pd.DataFrame, model_cols: List[str], target_col: str) -> pd.DataFrame:
    """Run MAE and MSE DM tests for all model pairs."""
    if len(df) < 5 or len(model_cols) < 2:
        return pd.DataFrame()

    results: List[Dict[str, float]] = []
    for i in range(len(model_cols)):
        for j in range(i + 1, len(model_cols)):
            a, b = model_cols[i], model_cols[j]
            for loss in ("mae", "mse"):
                try:
                    res = diebold_mariano_test(
                        true_values=df[target_col],
                        model_predictions=df[a],
                        reference_predictions=df[b],
                        forecast_horizon=1,
                        loss=loss,
                        newey_west_lags=None,
                        use_small_sample_correction=True,
                        auto_select_nw=True,
                        label_model=a,
                        label_reference=b,
                    )
                except Exception as e:
                    print(f"[WARN] DM failed for {a} vs {b} ({loss}): {e}")
                    res = {
                        "pair": f"{a} vs {b}",
                        "loss": loss.upper(),
                        "n": np.nan,
                        "h": 1,
                        "nw_lags": np.nan,
                        "dm_raw": np.nan,
                        "dm": np.nan,
                        "p_norm": np.nan,
                        "p_t": np.nan,
                        "mean_diff": np.nan,
                        "better": "NA",
                    }
                results.append(res)

    return pd.DataFrame(results)


# ========================= PLOTS =========================

def _slice_last_hours(df: pd.DataFrame, datetime_col: str, hours: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    end_time = df[datetime_col].max()
    start_time = end_time - pd.Timedelta(hours=hours)
    out = df[df[datetime_col] > start_time].copy()
    return out.sort_values(datetime_col).reset_index(drop=True)


def plot_month_peaks(
    df: pd.DataFrame,
    datetime_col: str,
    target_col: str,
    model_cols: List[str],
    threshold: float,
    top_from_table: pd.DataFrame,
    save_path: Path,
    show: bool = False,
) -> List[str]:
    """
    Draw last 31 days: Observed, red dots on peaks, and top-2 models by recall_peak
    with tie-break by lower WMAE.
    """
    # Accept either index='model' or column 'model'
    if "model" in top_from_table.columns:
        rank = top_from_table.copy()
    else:
        rank = top_from_table.reset_index().rename(columns={"index": "model"})

    if "recall_peak" not in rank.columns:
        rank["recall_peak"] = np.nan
    if "WMAE" not in rank.columns:
        rank["WMAE"] = np.nan

    # Primary: recall of peaks; tie-break: lower WMAE
    rank = rank.sort_values(by=["recall_peak", "WMAE"], ascending=[False, True])
    chosen = [m for m in rank["model"].tolist() if m in model_cols][:2]

    sub = _slice_last_hours(df, datetime_col, hours=24 * 31)
    if sub.shape[0] < 5:
        print("[WARN] Not enough data to plot month peaks.")
        return chosen

    is_peak = sub[target_col].astype(float) >= float(threshold)

    plt.figure(figsize=(14, 6))
    plt.plot(sub[datetime_col], sub[target_col], label="Observed", linewidth=2.4)
    plt.scatter(sub.loc[is_peak, datetime_col], sub.loc[is_peak, target_col], s=26, color="red", label="Peaks")

    for m in chosen:
        if m in sub.columns:
            plt.plot(sub[datetime_col], sub[m], label=m, linewidth=1.6)

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=9, frameon=True)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    safe_savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show(block=SHOW_PLOTS_BLOCK)
    plt.close()
    return chosen


# ========================= MAIN =========================

def main() -> None:
    # Resolve input
    input_dir = PATHS.merged_predictions_dir
    input_file = resolve_predictions_file(input_dir, DEFAULT_PREDICTIONS_STEM)
    print(f"[INFO] Project root: {PATHS.project_root.as_posix()}")
    print(f"[INFO] Input dir:     {input_dir.as_posix()}")
    print(f"[INFO] Input file:    {input_file.name}")

    # Load and detect models
    data = load_predictions(input_file, DATETIME_COLUMN_NAME, TARGET_COLUMN_NAME)
    model_columns = list_model_columns(
        data, DATETIME_COLUMN_NAME, TARGET_COLUMN_NAME, min_valid_ratio=MIN_VALID_RATIO
    )
    if not model_columns:
        raise RuntimeError("No valid model columns detected. Check numeric types and MIN_VALID_RATIO.")
    print(f"[INFO] Rows: {len(data):,} | Models: {', '.join(model_columns)}")

    # Peak threshold for plots
    if TRAIN_PEAK_THRESHOLD_VALUE is not None:
        peak_threshold_for_plot = float(TRAIN_PEAK_THRESHOLD_VALUE)
        print(f"[INFO] Using fixed training threshold (plot): {peak_threshold_for_plot:.4f}")
    else:
        peak_threshold_for_plot = choose_peak_threshold(data[TARGET_COLUMN_NAME], PLOT_QUANTILE)

    # Peak-aware metrics across quantiles
    all_rows: List[Dict[str, float]] = []
    for q in PEAK_QUANTILES:
        if TRAIN_PEAK_THRESHOLD_VALUE is not None:
            peak_threshold = float(TRAIN_PEAK_THRESHOLD_VALUE)
            print(f"[INFO] Using fixed training threshold: {peak_threshold:.4f} (q={q:.2f} requested)")
        else:
            peak_threshold = choose_peak_threshold(data[TARGET_COLUMN_NAME], q)

        for m in model_columns:
            pair = data[[TARGET_COLUMN_NAME, m]].dropna()
            if pair.shape[0] < 5:
                all_rows.append({
                    "model": m, "Peak Quantile": q, "threshold": peak_threshold,
                    "n_peaks": 0, "n_non": 0,
                    "MAE_peaks": np.nan, "MAE_nonpeaks": np.nan, "WMAE": np.nan,
                    "precision_peak": np.nan, "recall_peak": np.nan, "f1_peak": np.nan
                })
                continue

            met = compute_peak_aware_metrics(
                pair[TARGET_COLUMN_NAME], pair[m],
                threshold=peak_threshold, peak_weight=PEAK_WEIGHT
            )
            all_rows.append({"model": m, "Peak Quantile": q, **met})

    peak_df = pd.DataFrame(all_rows)[[
        "model", "Peak Quantile", "threshold",
        "n_peaks", "n_non", "MAE_peaks", "MAE_nonpeaks",
        "WMAE", "precision_peak", "recall_peak", "f1_peak"
    ]]

    stem = input_file.stem
    out_peak = TABLES_DIR / f"peak_metrics_{stem}.csv"
    safe_to_csv(peak_df, out_peak, index=False)

    print_table(peak_df.round(ROUND_LOG_DECIMALS), "Peak metrics (all quantiles)")

    # DM tests (full file, all pairs)
    dm_df = run_dm_for_all_pairs(data, model_columns, TARGET_COLUMN_NAME)
    out_dm_raw = TABLES_DIR / f"dm_raw_{stem}.csv"
    out_dm_sum = TABLES_DIR / f"dm_summary_{stem}.csv"

    if dm_df.empty:
        print("[INFO] DM tests skipped (not enough data or too few model columns).")
    else:
        safe_to_csv(dm_df, out_dm_raw, index=False)
        print_table(dm_df.round(ROUND_LOG_DECIMALS), "DM raw")

        # Build compact summary
        def _sig_stars(p: float) -> str:
            if not np.isfinite(p):
                return ""
            if p < 1e-3: return "***"
            if p < 1e-2: return "**"
            if p < 5e-2: return "*"
            return ""

        df_sum = dm_df.copy()
        df_sum["delta_abs"] = df_sum["mean_diff"].abs().astype(float)
        df_sum["sig"] = df_sum["p_t"].apply(_sig_stars)
        df_sum = df_sum.sort_values(["p_t", "delta_abs"], ascending=[True, False])
        keep = ["pair", "loss", "better", "mean_diff", "dm", "p_t", "sig", "n", "nw_lags"]
        keep = [c for c in keep if c in df_sum.columns]
        dm_summary = df_sum[keep].copy()

        safe_to_csv(dm_summary, out_dm_sum, index=False)
        print_table(dm_summary.round(ROUND_LOG_DECIMALS), "DM summary (sorted)")

    # Month peaks plot
    plot_path = PLOTS_DIR / f"plot_peaks_month_{stem}.png"
    peak_df_q = peak_df[peak_df["Peak Quantile"] == PLOT_QUANTILE].copy()
    chosen = plot_month_peaks(
        df=data,
        datetime_col=DATETIME_COLUMN_NAME,
        target_col=TARGET_COLUMN_NAME,
        model_cols=model_columns,
        threshold=float(peak_df_q["threshold"].iloc[0]) if not peak_df_q.empty else float("nan"),
        top_from_table=peak_df_q.set_index("model"),
        save_path=plot_path,
        show=SHOW_PLOTS,
    )
    if chosen:
        print(f"[INFO] Month peaks plot models (q={PLOT_QUANTILE:.2f}): {', '.join(chosen)}")
    else:
        print("[INFO] Month peaks plot skipped (no eligible models).")


if __name__ == "__main__":
    main()
