# -*- coding: utf-8 -*-
"""
Scope
- Loads a prepared feature table with a datetime index.
- Evaluates models with walk-forward time-series cross-validation (fixed 7-day horizon by default).
- Reports MAE, RMSE, MAPE, RMSLE, R²; saves a ranked CSV; draws last-fold plots.
- Provide feature importance helpers (GBR/XGB) and residual diagnostics.
- Include a robust SARIMA/SARIMAX wrapper with safe fallback and hybrid option.

Outputs
- results/ranking_models.csv
- results/plot_LAST_FOLD_ALL.png
- results/plot_LAST_FOLD_<MODEL>.png
- results/gbr_top_features.csv, results/xgb_top_features.csv
"""
from __future__ import annotations

import copy
import os
import gc
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ========================= PATHS =========================

def bootstrap_paths_models() -> SimpleNamespace:
    """Locate project root (having data/processed) and prep results paths."""
    try:
        here = Path(__file__).resolve()
    except NameError:
        here = Path.cwd().resolve()

    root = None
    for p in [here] + list(here.parents):
        if (p / "data" / "processed").exists():
            root = p
            break

    if root is None:
        env = os.environ.get("PROJECT_ROOT")
        if env and (Path(env) / "data" / "processed").exists():
            root = Path(env).resolve()

    if root is None:
        root = here.parent.parent

    data = root / "data"
    processed = data / "processed"
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)

    return SimpleNamespace(
        project_root=root,
        data_dir=data,
        processed_dir=processed,
        results_dir=results,
    )


PATHS = bootstrap_paths_models()
PROJECT_ROOT = PATHS.project_root
DATA_PROCESSED = PATHS.processed_dir
RESULTS_DIR = PATHS.results_dir

FEATURES_FILE = DATA_PROCESSED / "ready_database.parquet"
TARGET_COLUMN = "fixing_i_price"
DATETIME_COLUMN_NAME = "datetime"

DROP_EXOG_FEATURES = {
     "co2_objetosc", "gaz_objetosc",
     "season_summer", "hour", "month",
     "season_spring", "season_winter",
     "kurs_eur_pln","kurs_usd_pln",
}

# Multi-horizon setup
HORIZON_DAYS = [1,7,14,31]
HORIZONS_HOURS = [d * 24 for d in HORIZON_DAYS]

# Validation tests
DEFAULT_TEST_HORIZON = 24 * 7
DEFAULT_N_SPLITS = 5


# ========================= DATA =========================

def load_features_table(path: Path, datetime_column: str = DATETIME_COLUMN_NAME) -> pd.DataFrame:
    """Load Parquet/CSV by extension; ensure datetime index and hourly frequency.
    On failure, return empty DataFrame and continue gracefully.
    """
    if not path.exists():
        print(f"[WARN] Feature table not found: {path}")
        return pd.DataFrame()

    try:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            print(f"[WARN] Unsupported extension '{path.suffix}'. Expected .parquet or .csv")
            return pd.DataFrame()
    except Exception as exc:
        print(f"[WARN] Failed to read {path}: {exc}")
        return pd.DataFrame()

    if datetime_column in df.columns:
        df[datetime_column] = pd.to_datetime(df[datetime_column], errors="coerce")
        df = df.dropna(subset=[datetime_column]).set_index(datetime_column)

    if not isinstance(df.index, pd.DatetimeIndex):
        print("[WARN] Feature table must have a datetime index or a 'datetime' column.")
        return pd.DataFrame()

    df = df.sort_index().asfreq("h")
    df = df.dropna(how="any")
    print(f"[INFO] Using features file: {path} -> {len(df):,} rows x {df.shape[1]} cols")
    return df


# ========================= CONTEXT & METRICS =========================

class RunContext:
    """Holds last-fold series and aggregate metrics for a run."""
    def __init__(self):
        self.last_fold: Optional[Tuple[pd.Series, Dict[str, pd.Series]]] = None
        self.all_results: List[Dict] = []
        self.preds_by_model: Dict[str, pd.Series] = {}
        self.full_series: Optional[pd.Series] = None


class Metrics:
    @staticmethod
    def mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mape(y_true, y_pred, eps: float = 1e-9):
        tv = np.asarray(y_true, dtype=float)
        pv = np.asarray(y_pred, dtype=float)
        denom = np.clip(np.abs(tv), eps, None)
        return np.mean(np.abs(tv - pv) / denom) * 100.0

    @staticmethod
    def rmsle(y_true, y_pred, eps: float = 1e-9):
        tv = np.maximum(np.asarray(y_true, dtype=float), 0.0)
        pv = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
        return np.sqrt(np.mean((np.log1p(pv + eps) - np.log1p(tv + eps)) ** 2))

    @staticmethod
    def r2(y_true, y_pred):
        return r2_score(y_true, y_pred)


# === ANALYSIS TOOLS (validation, saving, multi-horizon, recursive blocks) ===

class AnalysisTools:
    """Utilities for validation, saving outputs and multi-horizon orchestration"""
    @staticmethod
    def validate_time_series_model(
        input_df: pd.DataFrame,
        target_column: str,
        model_obj,
        test_horizon: int = DEFAULT_TEST_HORIZON,
        n_splits: int = DEFAULT_N_SPLITS,
        model_name: str = "",
        print_results: bool = True,
        run_context: Optional[RunContext] = None,
    ) -> np.ndarray:
        """Walk-forward CV with fixed test horizon. Returns metrics [MAE, RMSE, MAPE, RMSLE, R2]."""
        name = model_name or model_obj.__class__.__name__
        rc = run_context or RunContext()

        if input_df is None or input_df.empty:
            print(f"[SKIP] {name}: empty input table.")
            return np.array([np.nan] * 5)

        need_rows = test_horizon * (n_splits + 1)
        if len(input_df) <= need_rows:
            print(f"[SKIP] {name}: not enough rows for {n_splits} splits of {test_horizon}h.")
            return np.array([np.nan] * 5)

        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_horizon)
        metrics_rows: List[List[float]] = []

        def _update_last_fold(y_true: pd.Series, y_pred: np.ndarray) -> None:
            pred_s = pd.Series(y_pred, index=y_true.index, name=name)
            if rc.last_fold is None:
                rc.last_fold = (y_true.copy(), {name: pred_s})
                return
            y_ref, prev = rc.last_fold
            common_idx = y_ref.index.intersection(y_true.index)
            if len(common_idx) == 0:
                rc.last_fold = (y_true.copy(), {name: pred_s})
                return
            y_ref = y_ref.loc[common_idx]
            prev = {k: v.reindex(common_idx) for k, v in prev.items()}
            prev[name] = pred_s.reindex(common_idx)
            rc.last_fold = (y_ref, prev)

        for idx_tr, idx_te in tscv.split(input_df):
            train_df = input_df.iloc[idx_tr]
            test_df = input_df.iloc[idx_te]

            X_tr = train_df.drop(columns=[target_column])
            y_tr = train_df[target_column]
            X_te = test_df.drop(columns=[target_column])
            y_te = test_df[target_column]

            model_obj.fit(X_tr, y_tr)
            y_pred = model_obj.predict(X_te)

            if len(y_pred) == len(y_te):
                rc.preds_by_model[name] = pd.Series(y_pred, index=y_te.index)
                _update_last_fold(y_te, y_pred)

            mae = Metrics.mae(y_te, y_pred)
            rmse = Metrics.rmse(y_te, y_pred)
            mape = Metrics.mape(y_te, y_pred)
            rmsle = Metrics.rmsle(y_te, y_pred)
            r2 = Metrics.r2(y_te, y_pred)
            metrics_rows.append([mae, rmse, mape, rmsle, r2])

        mean_vals = np.array(metrics_rows).mean(axis=0)

        if print_results:
            print(f"\nValidation - {name} (h={test_horizon}h)")
            print(f"MAE   : {mean_vals[0]:8.2f}")
            print(f"RMSE  : {mean_vals[1]:8.2f}")
            print(f"MAPE  : {mean_vals[2]:8.2f} %")
            print(f"RMSLE : {mean_vals[3]:8.4f}")
            print(f"R2    : {mean_vals[4]:8.4f}")

        rc.all_results.append(dict(
            model=name,
            HorizonHours=test_horizon,
            MAE=mean_vals[0],
            RMSE=mean_vals[1],
            MAPE=mean_vals[2],
            RMSLE=mean_vals[3],
            R2=mean_vals[4],
        ))
        return mean_vals

    @staticmethod
    def plot_forecast_vs_observed(
        y_observed: pd.Series,
        y_pred: pd.Series,
        model_name: str,
        *,
        title_prefix: str = "Forecast vs Observed (last fold)",
        save_path: Optional[Path] = None,
    ) -> Dict[str, float]:
        mae = Metrics.mae(y_observed, y_pred)
        rmse = Metrics.rmse(y_observed, y_pred)
        mape = Metrics.mape(y_observed, y_pred)

        plt.figure(figsize=(12, 4))
        plt.plot(y_observed, label="Observed", linewidth=2, color="black")
        plt.plot(
            y_pred,
            "--",
            label=f"{model_name}  MAE={mae:.2f} | RMSE={rmse:.2f} | MAPE={mape:.2f}%",
        )
        plt.title(f"{title_prefix} – {model_name}")
        plt.legend()
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=120)
            plt.close()
        else:
            plt.show()
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    @staticmethod
    def run_recursive_blocks(
        models: Dict[str, BaseEstimator],
        features_dataframe: pd.DataFrame,
        target_column: str,
        sarima_dataset: pd.DataFrame,
        sarimax_dataset: Optional[pd.DataFrame],
        block_hours: int = 24 * 7,
        n_blocks: int = 10,
        save_csv_path: Path = RESULTS_DIR / "recursive_blocks_h168x10_ML.csv",
        run_context: Optional[RunContext] = None,
    ) -> pd.DataFrame:
        """Train/predict in fixed-size blocks over the tail window; merge blocks and score on a common horizon"""
        rc = run_context or RunContext()

        total_hours = block_hours * n_blocks
        if len(features_dataframe) <= total_hours:
            raise ValueError(
                f"Not enough data for {n_blocks} blocks of {block_hours}h (need > {total_hours} rows)."
            )

        eval_index = features_dataframe.index[-total_hours:]
        start_idx = len(features_dataframe) - total_hours
        block_bounds = [
            (start_idx + i * block_hours, start_idx + (i + 1) * block_hours)
            for i in range(n_blocks)
        ]

        y_obs_full = features_dataframe.loc[eval_index, target_column].astype(float)
        predictions: Dict[str, pd.Series] = {}
        ranking_rows: List[Dict] = []

        for name, model_object in models.items():
            if name == "SARIMAX":
                input_df = sarimax_dataset
            elif name == "SARIMA":
                input_df = sarima_dataset
            else:
                input_df = features_dataframe

            if input_df is None or input_df.empty or target_column not in input_df.columns:
                print(f"[SKIP] {name}: input missing or no '{target_column}'")
                continue

            preds_full: List[pd.Series] = []
            for block_id, (b_start, b_end) in enumerate(block_bounds, start=1):
                train = input_df.iloc[:b_start]
                test = input_df.iloc[b_start:b_end]

                if len(test) != block_hours or len(train) == 0:
                    print(f"[{name}] Skip block {block_id}: train={len(train)} test={len(test)}")
                    continue

                X_tr = train.drop(columns=[target_column])
                y_tr = train[target_column].astype(float)
                X_te = test.drop(columns=[target_column])

                try:
                    model_object.fit(X_tr, y_tr)
                    y_block = pd.Series(
                        model_object.predict(X_te), index=X_te.index, name=name
                    ).astype(float)
                except Exception as e:
                    print(f"[ERROR] {name} block {block_id}: {e!s}")
                    y_block = pd.Series(
                        [np.nan] * len(X_te), index=X_te.index, name=name
                    )
                preds_full.append(y_block)

            if not preds_full:
                print(f"[WARN] {name}: no blocks produced.")
                continue

            pred_series = pd.concat(preds_full).sort_index().reindex(eval_index)
            predictions[name] = pred_series

            yt, yp = y_obs_full.align(pred_series, join="inner")
            mask = yt.notna() & yp.notna()
            if mask.any():
                mae = Metrics.mae(yt[mask].values, yp[mask].values)
                rmse = Metrics.rmse(yt[mask].values, yp[mask].values)
                mape = Metrics.mape(yt[mask].values, yp[mask].values)
                print(
                    f"[RECURSIVE] {name}: MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%  over {mask.sum()} points"
                )
                ranking_rows.append(dict(model=name, MAE=mae, RMSE=rmse, MAPE=mape))
                rc.all_results.append(
                    dict(
                        model=name,
                        HorizonHours=total_hours,
                        MAE=mae,
                        RMSE=rmse,
                        MAPE=mape,
                        RMSLE=np.nan,
                        R2=np.nan,
                    )
                )
            else:
                print(f"[RECURSIVE] {name}: no valid points to score.")

            try:
                if hasattr(model_object, "model_") and model_object.model_ is not None:
                    try:
                        if hasattr(model_object.model_, "remove_data"):
                            model_object.model_.remove_data()
                    except Exception:
                        pass
                    model_object.model_ = None
                if hasattr(model_object, "fallback_mean_value"):
                    model_object.fallback_mean_value = None
                if hasattr(model_object, "use_fallback"):
                    model_object.use_fallback = False
            except Exception:
                pass

        out_df = pd.DataFrame({"Observed": y_obs_full})
        for name, pred in predictions.items():
            out_df[name] = pred.reindex(out_df.index)

        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(save_csv_path, index_label="timestamp", float_format="%.6f")
        print(f"[SAVE] Recursive blocks -> {save_csv_path}")

        if ranking_rows:
            rank_df = pd.DataFrame(ranking_rows).sort_values("MAE").reset_index(drop=True)
            rank_path = RESULTS_DIR / "ranking_recursive_blocks_ML.csv"
            rank_df.to_csv(rank_path, index=False)
            print("[RANK] Recursive blocks ranking")
            print(
                rank_df.to_string(
                    index=False,
                    formatters={
                        "MAE": "{:.2f}".format,
                        "RMSE": "{:.2f}".format,
                        "MAPE": "{:.2f}".format,
                    },
                )
            )
            print(f"[SAVE] {rank_path}")

        try:
            plt.figure(figsize=(12, 4))
            plt.plot(out_df.index, out_df["Observed"].values, label="Observed", linewidth=2, color="black")
            for name in predictions.keys():
                plt.plot(out_df.index, out_df[name].values, label=name, alpha=0.9)
            plt.title(f"Recursive block forecasts – {n_blocks}×{block_hours}h")
            plt.legend()
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / f"plot_recursive_blocks_h{block_hours}x{n_blocks}_ML.png", dpi=120)
            plt.close()
        except Exception:
            pass

        return out_df

    @staticmethod
    def save_horizon_outputs(
        horizon_hours: int,
        y_true: pd.Series,
        preds: Dict[str, pd.Series],
    ) -> Path:
        """Save last-fold predictions for a given horizon to CSV and plot."""
        preferred = ["XGB", "GBR", "SARIMA", "SARIMAX"]
        model_order = [m for m in preferred if m in preds] + [m for m in preds if m not in preferred]

        aligned = {name: preds[name].reindex(y_true.index) for name in model_order}

        out_df = pd.DataFrame(index=y_true.index)
        for name in model_order:
            out_df[name] = aligned[name].astype(float).values
        out_df["Observed"] = y_true.astype(float).values

        csv_path = RESULTS_DIR / f"forecasts_h{horizon_hours}_hours.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(csv_path, index_label="timestamp", float_format="%.6f")

        plt.figure(figsize=(12, 4))
        plt.plot(y_true.index, y_true.values, label="Observed", linewidth=2, color="black")
        for name in model_order:
            series = aligned[name]
            plt.plot(series.index, series.values, label=name, alpha=0.9)
        plt.title(f"Forecasts (last fold) – horizon={horizon_hours}h")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"plot_h{horizon_hours}_hours.png", dpi=120)
        plt.close()

        return csv_path

    @staticmethod
    def run_walkforward_multi_horizon(
        models: Dict[str, BaseEstimator],
        features_dataframe: pd.DataFrame,
        target_column: str,
        sarima_dataset: pd.DataFrame,
        sarimax_dataset: Optional[pd.DataFrame],
        horizons_hours: List[int],
        global_context: RunContext,
    ) -> pd.DataFrame:
        """Loop over horizons, run CV per model, save per-horizon outputs and collect ranking rows."""
        for h in horizons_hours:
            print(f"\n=== Walk-forward for horizon {h} hours ===")
            local_ctx = RunContext()

            for name, model_obj in models.items():
                if name == "SARIMAX":
                    input_df = sarimax_dataset
                elif name == "SARIMA":
                    input_df = sarima_dataset
                else:
                    input_df = features_dataframe

                AnalysisTools.validate_time_series_model(
                    input_df=input_df,
                    target_column=target_column,
                    model_obj=model_obj,
                    test_horizon=h,
                    n_splits=DEFAULT_N_SPLITS,
                    model_name=name,
                    run_context=local_ctx,
                )

            if local_ctx.preds_by_model:
                y_true_last = features_dataframe[target_column].tail(h).astype(float)
                idx_common = y_true_last.index
                for ser in local_ctx.preds_by_model.values():
                    idx_common = idx_common.intersection(ser.index)
                if len(idx_common) == 0:
                    idx_common = y_true_last.index
                y_true_last = y_true_last.loc[idx_common]
                preds_dict = {n: s.reindex(idx_common) for n, s in local_ctx.preds_by_model.items()}

                csv_path = AnalysisTools.save_horizon_outputs(h, y_true=y_true_last, preds=preds_dict)
                print(f"[INFO] Saved horizon outputs -> {csv_path}")

            for rec in local_ctx.all_results:
                global_context.all_results.append(rec)

            for m in models.values():
                try:
                    if hasattr(m, "model_") and m.model_ is not None:
                        try:
                            if hasattr(m.model_, "remove_data"):
                                m.model_.remove_data()
                        except Exception:
                            pass
                        m.model_ = None
                    if hasattr(m, "fallback_mean_value"):
                        m.fallback_mean_value = None
                    if hasattr(m, "use_fallback"):
                        m.use_fallback = False
                except Exception:
                    pass
            gc.collect()

        return pd.DataFrame()

# ========================= FEATURE IMPORTANCE =========================

class FeatureImportance:
    @staticmethod
    def permutation_importance_csv(
        fitted_estimator,
        features_dataframe: pd.DataFrame,
        target_column: str,
        top_n: int,
        save_csv_path: Path,
        save_plot_path: Optional[Path] = None,
        window_hours: int = 24 * 7,
        n_repeats: int = 10,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """
        Permutation importance measured as ΔMAE on the last `window_hours`.
        Works for sklearn models and for SarimaX (uses its exogenous_columns if set).
        """
        rng = np.random.default_rng(random_seed)
        h = int(np.clip(window_hours, 24, max(24, len(features_dataframe) - 1)))

        train_df = features_dataframe.iloc[:-h]
        val_df = features_dataframe.iloc[-h:]

        X_tr = (
            train_df.drop(columns=[target_column])
            .select_dtypes(include=[np.number])
            .loc[:, lambda d: ~d.columns.duplicated()]
        )
        y_tr = train_df[target_column].astype(float)

        X_val_full = (
            val_df.drop(columns=[target_column])
            .select_dtypes(include=[np.number])
            .loc[:, lambda d: ~d.columns.duplicated()]
            .copy()
        )
        y_val = val_df[target_column].astype(float)

        # Restrict to SARIMAX exog, if applicable
        if isinstance(fitted_estimator, SarimaX) and fitted_estimator.exogenous_columns:
            cols = [c for c in fitted_estimator.exogenous_columns if c in X_val_full.columns]
            X_val = X_val_full.loc[:, cols].copy()
        else:
            X_val = X_val_full

        # Fit a fresh copy if possible
        try:
            est = fitted_estimator.__class__(**{k: v for k, v in fitted_estimator.__dict__.items()
                                                if not k.endswith("_")})
            est.fit(X_tr, y_tr)
        except Exception:
            est = fitted_estimator

        base_pred = est.predict(X_val)
        base_mae = mean_absolute_error(y_val, base_pred)

        rows: List[tuple[str, float]] = []
        for col in list(X_val.columns):
            scores = []
            for _ in range(n_repeats):
                Xp = X_val.copy()
                Xp[col] = rng.permutation(Xp[col].values)
                yp = est.predict(Xp)
                scores.append(mean_absolute_error(y_val, yp))
            rows.append((col, float(np.mean(scores) - base_mae)))

        df = (
            pd.DataFrame(rows, columns=["feature", "importance"])
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv_path, index=False)

        if save_plot_path is not None:
            plt.figure(figsize=(10, max(4, 0.35 * len(df) + 1)))
            plt.barh(df["feature"][::-1], df["importance"][::-1])
            plt.xlabel("Permutation importance (ΔMAE)")
            plt.title(f"TOP-{top_n} permutation importance")
            plt.tight_layout()
            plt.savefig(save_plot_path, dpi=120)
            plt.close()

        print(f"[INFO] Saved permutation importance -> {save_csv_path}")
        return df

    @staticmethod
    def plot_permutation_importance(
        model_obj,
        features_dataframe: pd.DataFrame,
        target_column: str,
        top_n_features: int = 20,
        n_repeats: int = 10,
        random_seed: int = 42,
    ) -> None:
        """
        Quick visual-only permutation importance on the whole table (ΔMAE).
        For consistent CSV outputs, prefer `permutation_importance_csv`.
        """
        rng = np.random.default_rng(random_seed)
        X = features_dataframe.drop(columns=[target_column]).select_dtypes(include=[np.number])
        y = features_dataframe[target_column].astype(float)

        # fit a fresh copy to avoid overwriting
        try:
            est = model_obj.__class__(**{k: v for k, v in model_obj.__dict__.items() if not k.endswith("_")})
            est.fit(X, y)
        except Exception:
            est = model_obj

        base_pred = est.predict(X)
        base_mae = mean_absolute_error(y, base_pred)

        rows = []
        for col in list(X.columns):
            scores = []
            for _ in range(n_repeats):
                Xp = X.copy()
                Xp[col] = rng.permutation(Xp[col].values)
                yp = est.predict(Xp)
                scores.append(mean_absolute_error(y, yp))
            rows.append((col, float(np.mean(scores) - base_mae)))

        df = (
            pd.DataFrame(rows, columns=["feature", "importance"])
            .sort_values("importance", ascending=False)
            .head(top_n_features)
        )

        plt.figure(figsize=(10, max(4, 0.35 * len(df) + 1)))
        plt.barh(df["feature"][::-1], df["importance"][::-1])
        plt.xlabel("Permutation importance (ΔMAE)")
        plt.title("Permutation importance")
        plt.tight_layout()
        plt.show()

# ========================= MODELS =========================

class SarimaX(BaseEstimator, RegressorMixin):
    """SARIMA/SARIMAX wrapper with safe fallback (mean) and optional truncation for long series."""
    def __init__(self,
                 exogenous_columns: Optional[List[str]] = None,
                 order: Tuple[int, int, int] = (1, 0, 2),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24),
                 maximum_observations: Optional[int] = None):
        self.exogenous_columns = exogenous_columns
        self.order = order
        self.seasonal_order = seasonal_order
        self.maximum_observations = maximum_observations
        self.model_ = None
        self.use_fallback = False
        self.fallback_mean_value = None


    def select_exogenous_columns(features_dataframe: pd.DataFrame, target_column: str,
                                 blacklist: Optional[List[str] | set[str]] = None, ) -> List[str]:
        """Select numeric columns except the target as exogenous features (with optional blacklist)."""
        if features_dataframe.empty:
            return []
        cols = (
            features_dataframe.drop(columns=[target_column], errors="ignore")
            .select_dtypes(include=[np.number])
            .loc[:, lambda d: ~d.columns.duplicated()]
            .columns.tolist()
        )

        if blacklist:
            bl = set(map(str, blacklist))
            before = set(cols)
            cols = [c for c in cols if c not in bl]
            dropped = sorted(before - set(cols))
            if dropped:
                print(f"[EXOG] Dropped by blacklist ({len(dropped)}): {dropped}")
        print(f"[EXOG] Using {len(cols)} exogenous columns.")
        return cols

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.maximum_observations is not None and len(y) > self.maximum_observations:
            y = y.tail(self.maximum_observations)
            X = X.tail(self.maximum_observations)

        exog = X[self.exogenous_columns] if self.exogenous_columns else None

        if len(y) < 2 * self.seasonal_order[3]:
            self.fallback_mean_value = y.tail(24).mean()
            if np.isnan(self.fallback_mean_value):
                self.fallback_mean_value = y.mean()
            self.use_fallback = True
            print("SARIMA fallback: too little data, using 24h mean")
            return self

        try:
            self.model_ = SARIMAX(
                y, exog=exog, order=self.order, seasonal_order=self.seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
            print("SARIMA fitted successfully")
        except Exception as e:
            self.fallback_mean_value = y.tail(24).mean()
            if np.isnan(self.fallback_mean_value):
                self.fallback_mean_value = y.mean()
            self.use_fallback = True
            print(f"SARIMA fallback due to exception: {str(e)} - using 24h mean")
        return self

    def predict(self, X: pd.DataFrame):
        steps = len(X)
        if self.use_fallback or self.model_ is None:
            return pd.Series(np.full(steps, self.fallback_mean_value), index=X.index)
        exog = X[self.exogenous_columns] if self.exogenous_columns else None
        fc = self.model_.forecast(steps=steps, exog=exog)
        return pd.Series(fc.values, index=X.index)
class GradientBoostingModel(BaseEstimator, RegressorMixin):
    """GradientBoostingRegressor with ES or RS modes."""
    def __init__(self,
                 mode: str = "es",
                 n_iter_rs: int = 20,
                 cross_validation_splits: int = 5,
                 max_boosting_rounds: int = 1500,
                 validation_fraction: float = 0.1,
                 random_seed: int = 42,
                 learning_rate: float = 0.015,
                 maximum_tree_depth: int = 4,
                 subsample_ratio: float = 0.8,
                 minimum_samples_per_leaf: int = 15):
        self.mode = mode
        self.n_iter_rs = n_iter_rs
        self.max_boosting_rounds = max_boosting_rounds
        self.cross_validation_splits = cross_validation_splits
        self.validation_fraction = validation_fraction
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.maximum_tree_depth = maximum_tree_depth
        self.subsample_ratio = subsample_ratio
        self.minimum_samples_per_leaf = minimum_samples_per_leaf
        self.model_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if X.shape[1] == 0:
            raise ValueError("GBR has 0 columns")
        if X.shape[0] == 0:
            raise ValueError("GBR has 0 rows")

        if self.mode == "es":
            split = int(len(X) * (1 - self.validation_fraction))
            X_tr, X_va = X.iloc[:split], X.iloc[split:]
            y_tr, y_va = y.iloc[:split], y.iloc[split:]

            tmp = GradientBoostingRegressor(
                n_estimators=self.max_boosting_rounds,
                learning_rate=self.learning_rate,
                max_depth=self.maximum_tree_depth,
                subsample=self.subsample_ratio,
                min_samples_leaf=self.minimum_samples_per_leaf,
                random_state=self.random_seed,
            ).fit(X_tr, y_tr)

            mae_val = [mean_absolute_error(y_va, p) for p in tmp.staged_predict(X_va)]
            best_iter = int(np.argmin(mae_val) + 1)
            print(f"GBR: best trees = {best_iter} (MAE val = {mae_val[best_iter-1]:.2f})")

            self.model_ = GradientBoostingRegressor(
                n_estimators=best_iter,
                learning_rate=self.learning_rate,
                max_depth=self.maximum_tree_depth,
                subsample=self.subsample_ratio,
                min_samples_leaf=self.minimum_samples_per_leaf,
                random_state=self.random_seed,
            ).fit(X, y)
        else:
            param_grid = {
                "n_estimators": [1300, 1700, 2600],
                "learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05],
                "max_depth": [3, 4, 6, 7, 8],
                "subsample": [0.7, 0.8, 1.0],
            }
            base = GradientBoostingRegressor(min_samples_leaf=self.minimum_samples_per_leaf, random_state=self.random_seed)
            tscv = TimeSeriesSplit(n_splits=self.cross_validation_splits, test_size=DEFAULT_TEST_HORIZON)
            rs = RandomizedSearchCV(base, param_grid, n_iter=self.n_iter_rs, scoring="neg_mean_absolute_error", cv=tscv, n_jobs=8, verbose=0).fit(X, y)
            print("GBR_RS: best params:", rs.best_params_)
            self.model_ = rs.best_estimator_
        return self

    def predict(self, X: pd.DataFrame):
        return self.model_.predict(X)
class XGBoostModel(BaseEstimator, RegressorMixin):
    """XGBRegressor with ES (Early stop) via xgb.cv or RS (Random Search)."""
    def __init__(self,
                 mode: str = "es",
                 n_iter_rs: int = 250,
                 cross_validation_splits: int = 5,
                 early_stopping_folds: int = 3,
                 max_boosting_rounds: int = 5000,
                 early_stopping_rounds: int = 50,
                 random_seed: int = 42):
        self.mode = mode
        self.n_iter_rs = n_iter_rs
        self.cross_validation_splits = cross_validation_splits
        self.early_stopping_folds = early_stopping_folds
        self.max_boosting_rounds = max_boosting_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.random_seed = random_seed
        self.model_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.mode == "es":
            dmatrix = xgb.DMatrix(X, label=y)
            params = dict(
                objective="reg:squarederror",
                learning_rate=0.04,
                max_depth=7,
                subsample=0.7,
                colsample_bytree=0.6,
                eval_metric="mae",
                seed=self.random_seed,
            )
            cv = xgb.cv(params, dmatrix, num_boost_round=self.max_boosting_rounds, nfold=self.early_stopping_folds,
                        early_stopping_rounds=self.early_stopping_rounds, metrics="mae", verbose_eval=False)
            best_iter = len(cv)
            print(f"XGB: best trees = {best_iter} (MAE cv = {cv['test-mae-mean'].iloc[-1]:.2f})")
            self.model_ = xgb.XGBRegressor(
                n_estimators=best_iter,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                n_jobs=8,
                random_state=self.random_seed,
                eval_metric="mae",
            ).fit(X, y)
        else:
            param_grid = {
                "n_estimators": [1500, 2500, 3500],
                "learning_rate": [0.03, 0.05, 0.07],
                "max_depth": [3, 4, 6],
                "subsample": [0.7, 0.8],
                "colsample_bytree": [0.6, 0.7, 0.8],
            }
            base = xgb.XGBRegressor(objective="reg:squarederror", n_jobs=8, random_state=self.random_seed)
            tscv = TimeSeriesSplit(n_splits=self.cross_validation_splits, test_size=DEFAULT_TEST_HORIZON)
            rs = RandomizedSearchCV(base, param_grid, n_iter=self.n_iter_rs, scoring="neg_mean_absolute_error", cv=tscv, n_jobs=8, verbose=0).fit(X, y)
            print("XGB_RS: best params:", rs.best_params_)
            self.model_ = rs.best_estimator_
        return self

    def predict(self, X: pd.DataFrame):
        return self.model_.predict(X)


def build_model_registry(exogenous_cols: Optional[List[str]]):
    return {
         "XGB": XGBoostModel(mode="es"),
         "GBR": GradientBoostingModel(mode="es"),
        # "GBR_RS": GradientBoostingModel(mode="rs", n_iter_rs=20, cross_validation_splits=5),
        # "XGB_RS": XGBoostModel(mode="rs", n_iter_rs=50, cross_validation_splits=5),
         "SARIMA": SarimaX(exogenous_columns=None, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24), maximum_observations=8000),
         "SARIMAX": SarimaX(exogenous_columns=exogenous_cols, order=(2, 0, 1), seasonal_order=(1, 1, 0, 24), maximum_observations=8000),
    }


# ========================= MAIN =========================

def main() -> None:
    warnings.filterwarnings("ignore")

    # 1) Load features
    features_dataframe = load_features_table(FEATURES_FILE)
    if features_dataframe.empty:
        print("[STOP] Features table is empty - nothing to evaluate.")
        return

    # 2) Exogenous selection (use module-level function if masz; w tej wersji jest w SarimaX)
    try:
        exogenous_cols = select_exogenous_columns(  # noqa: F821 if not defined at module level
            features_dataframe,
            TARGET_COLUMN,
            blacklist=DROP_EXOG_FEATURES,
        )
    except NameError:
        # fallback if the function lives in the SarimaX class
        exogenous_cols = SarimaX.select_exogenous_columns(
            features_dataframe,
            TARGET_COLUMN,
            blacklist=DROP_EXOG_FEATURES,
        )

    # 3) Model-specific views
    sarima_dataset = features_dataframe[[TARGET_COLUMN]]
    sarimax_dataset = features_dataframe[[TARGET_COLUMN] + exogenous_cols] if exogenous_cols else None

    # 4) Registry and context
    run_context = RunContext()
    run_context.full_series = features_dataframe[TARGET_COLUMN]
    models = build_model_registry(exogenous_cols)

    # 5) Walk-forward for multiple horizons
    AnalysisTools.run_walkforward_multi_horizon(
        models=models,
        features_dataframe=features_dataframe,
        target_column=TARGET_COLUMN,
        sarima_dataset=sarima_dataset,
        sarimax_dataset=sarimax_dataset,
        horizons_hours=HORIZONS_HOURS,
        global_context=run_context,
    )

    # 6) Recursive blocks benchmark
    AnalysisTools.run_recursive_blocks(
        models=models,
        features_dataframe=features_dataframe,
        target_column=TARGET_COLUMN,
        sarima_dataset=sarima_dataset,
        sarimax_dataset=sarimax_dataset,
        block_hours=24 * 7,
        n_blocks=10,
        save_csv_path=RESULTS_DIR / "recursive_blocks_h168x10_ML.csv",
        run_context=run_context,
    )

    # 7) Permutation importance snapshots
    try:
        if "XGB" in models:
            FeatureImportance.permutation_importance_csv(
                fitted_estimator=models["XGB"],
                features_dataframe=features_dataframe,
                target_column=TARGET_COLUMN,
                top_n=20,
                save_csv_path=RESULTS_DIR / "feature_importance_xgb.csv",
                save_plot_path=RESULTS_DIR / "plot_feature_importance_xgb.png",
                window_hours=DEFAULT_TEST_HORIZON,
            )
        if "GBR" in models:
            FeatureImportance.permutation_importance_csv(
                fitted_estimator=models["GBR"],
                features_dataframe=features_dataframe,
                target_column=TARGET_COLUMN,
                top_n=20,
                save_csv_path=RESULTS_DIR / "feature_importance_gbr.csv",
                save_plot_path=RESULTS_DIR / "plot_feature_importance_gbr.png",
                window_hours=DEFAULT_TEST_HORIZON,
            )
        if "SARIMAX" in models and sarimax_dataset is not None:
            FeatureImportance.permutation_importance_csv(
                fitted_estimator=models["SARIMAX"],
                features_dataframe=sarimax_dataset,
                target_column=TARGET_COLUMN,
                top_n=20,
                save_csv_path=RESULTS_DIR / "feature_importance_sarimax.csv",
                save_plot_path=RESULTS_DIR / "plot_feature_importance_sarimax.png",
                window_hours=DEFAULT_TEST_HORIZON,
            )
    except Exception as ex:
        print(f"[WARN] Feature importance skipped: {ex}")

    # 8) Summary table (ranking) and last-fold plots
    if run_context.all_results:
        df_rank = (
            pd.DataFrame(run_context.all_results)
            .sort_values(["HorizonHours", "MAE"])
            .reset_index(drop=True)
        )
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df_rank.to_csv(RESULTS_DIR / "ranking_models.csv", index=False)
        print("[SAVE] Ranking ->", RESULTS_DIR / "ranking_models.csv")
        print(
            df_rank.to_string(
                index=False,
                formatters={
                    "MAE": "{:.2f}".format,
                    "RMSE": "{:.2f}".format,
                    "MAPE": "{:.2f}".format,
                    "RMSLE": "{:.4f}".format,
                    "R2": "{:.4f}".format,
                },
            )
        )

    # Plot last fold (all models + per model) if available
    if run_context.last_fold is not None:
        y_obs_last, preds_dict = run_context.last_fold
        full_series = run_context.full_series

        # All models plot
        plt.figure(figsize=(11, 4))
        if full_series is not None:
            plt.plot(full_series.index, full_series.values, color="lightgray", label="Full series")
        plt.plot(y_obs_last.index, y_obs_last.values, color="black", linewidth=2, label="Observed - last fold")
        for name, pred in preds_dict.items():
            plt.plot(pred.index, pred.values, label=name, alpha=0.9)
        plt.legend()
        plt.title("Original data and forecasts - last fold")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "plot_LAST_FOLD_ALL.png", dpi=120)
        plt.close()

        # Per model plots
        for name, pred in preds_dict.items():
            plt.figure(figsize=(11, 4))
            if full_series is not None:
                plt.plot(full_series.index, full_series.values, color="lightgray", label="Full series")
            plt.plot(y_obs_last.index, y_obs_last.values, color="black", linewidth=2, label="Observed - last fold")
            plt.plot(pred.index, pred.values, label=name, alpha=0.9)
            plt.legend()
            plt.title(f"Last fold - {name}")
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / f"plot_LAST_FOLD_{name.replace('+', '_')}.png", dpi=120)
            plt.close()

if __name__ == "__main__":
    main()
