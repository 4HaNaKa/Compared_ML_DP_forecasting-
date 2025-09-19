"""
Title: Walk-forward forecasting pipeline
- Loads a prepared feature table with a datetime index.
- Evaluates models with walk-forward time-series cross-validation (fixed 7-day horizon by default).
- Reports MAE, RMSE, MAPE, RMSLE, R²; saves a ranked CSV; draws last-fold plots.
- Provides model-agnostic tooling: permutation importance (GBR/XGB), residual diagnostics, and feature comparisons.
- Implements a robust SARIMA/SARIMAX wrapper with safe fallback and two hybrid residual models.

Outputs
- results/ranking_models.csv                #averaged metrics across CV folds
- results/plot_LAST_FOLD_ALL.png            #last fold overlay plot (true vs. each model)
- results/plot_LAST_FOLD_<MODEL>.png        #per-model plots for the last fold
- results/top_features_gbr_xgb.csv          #comparative top features for GBR vs XGB (if both are run)
"""

from __future__ import annotations
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin

from xgboost import XGBRegressor
import xgboost as xgb

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
import copy

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# File with engineered, leakage-safe features
FEATURES_FILE = DATA_PROCESSED / "ready_database.parquet"

# Column names
TARGET_COLUMN = "fixing_i_price"
DATETIME_COLUMN_NAME = "datetime"  # used only when input is not already datetime-indexed

# Validation
DEFAULT_TEST_HORIZON = 24 * 7
DEFAULT_N_SPLITS = 5


def load_features_table(path: Path, datetime_column: str = DATETIME_COLUMN_NAME) -> pd.DataFrame:
    """Load parquet or CSV by extension; ensure datetime index and hourly frequency."""
    if not path.exists():
        raise FileNotFoundError(f"Feature table not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Only .parquet or .csv files are supported for FEATURES_FILE.")

    # If datetime column exists, set index; otherwise assume already indexed
    if datetime_column in df.columns:
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        df = df.set_index(datetime_column)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("The feature table must be indexed by datetime or include a 'datetime' column.")

    df = df.sort_index().asfreq("h")
    return df


def select_exogenous_columns(features_dataframe: pd.DataFrame, target_column: str) -> List[str]:
    """Select all numeric columns except the target as exogenous features. Safe drop of duplicates."""
    cols = (
        features_dataframe.drop(columns=[target_column])
        .select_dtypes(include=[np.number])
        .loc[:, lambda d: ~d.columns.duplicated()]
        .columns.tolist()
    )
    return cols


class RunContext:
    """Holds last-fold series and aggregate metrics for a single run."""
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

    @staticmethod
    def validate_time_series_model(
        input_df: pd.DataFrame,
        target_column: str,
        model_obj,
        test_horizon: int = DEFAULT_TEST_HORIZON,
        n_splits: int = DEFAULT_N_SPLITS,
        model_name: str = "",
        print_results: bool = True,
        ctx: Optional[RunContext] = None,
    ):
        """Walk-forward CV with fixed test horizon, returns average metrics array."""
        if ctx is None:
            ctx = RunContext()
        if not model_name:
            model_name = model_obj.__class__.__name__

        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_horizon)
        fold_metrics = []

        for train_idx, test_idx in tscv.split(input_df):
            train, test = input_df.iloc[train_idx], input_df.iloc[test_idx]
            X_train, y_train = train.drop(columns=[target_column]), train[target_column]
            X_test, y_test = test.drop(columns=[target_column]), test[target_column]

            model_obj.fit(X_train, y_train)
            y_pred = model_obj.predict(X_test)

            if len(y_pred) == len(y_test):
                ctx.preds_by_model[model_name] = pd.Series(y_pred, index=y_test.index)

            # metrics for this fold
            mae = Metrics.mae(y_test, y_pred)
            rmse = Metrics.rmse(y_test, y_pred)
            mape = Metrics.mape(y_test, y_pred)
            rmsle = Metrics.rmsle(y_test, y_pred)
            r2 = Metrics.r2(y_test, y_pred)
            fold_metrics.append([mae, rmse, mape, rmsle, r2])

            # last fold store
            preds_last = pd.Series(y_pred, index=y_test.index)
            if ctx.last_fold is None:
                ctx.last_fold = (y_test, {model_name: preds_last})
            else:
                y_prev, preds_prev = ctx.last_fold
                if not y_prev.index.equals(y_test.index):
                    y_prev = y_test
                    preds_prev = {}
                preds_prev[model_name] = preds_last
                ctx.last_fold = (y_prev, preds_prev)

        fold_metrics = np.array(fold_metrics).mean(axis=0)

        if print_results:
            print(f"\nValidation – {model_name}")
            print(f"MAE   : {fold_metrics[0]:8.2f}")
            print(f"RMSE  : {fold_metrics[1]:8.2f}")
            print(f"MAPE  : {fold_metrics[2]:8.2f} %")
            print(f"RMSLE : {fold_metrics[3]:8.4f}")
            print(f"R²    : {fold_metrics[4]:8.4f}")

        ctx.all_results.append(dict(
            model=model_name,
            MAE=fold_metrics[0], RMSE=fold_metrics[1], MAPE=fold_metrics[2], RMSLE=fold_metrics[3], R2=fold_metrics[4]
        ))
        return fold_metrics



class AnalysisTools:
    @staticmethod
    def top_features_by_gain_weight(
        model_obj,
        top_n_features: int = 40,
        *,
        bar_width: float = 0.4,
        cmap: Tuple[str, str] = ("tab:blue", "tab:orange"),
    ) -> pd.DataFrame:
        """Show TOP-N features by XGBoost gain & weight; draw a dual bar chart with auto scaling for readability."""
        booster = getattr(model_obj, "model_", model_obj).get_booster()
        gain_raw = booster.get_score(importance_type="gain")
        weight_raw = booster.get_score(importance_type="weight")

        df_feat = (
            pd.DataFrame({
                "feature": list(gain_raw.keys()),
                "gain": [float(v) for v in gain_raw.values()],
                "weight": [weight_raw.get(k, 0) for k in gain_raw.keys()],
            })
            .sort_values("gain", ascending=False)
            .head(top_n_features)
            .reset_index(drop=True)
        )

        w_max = df_feat["weight"].max() or 1
        g_max = df_feat["gain"].max() or 1

        import math
        scale_factor = 10 ** max(0, math.ceil(math.log10(g_max / (1.2 * w_max))))
        df_feat["gain_scaled"] = df_feat["gain"] / scale_factor
        df_feat["scale_factor"] = scale_factor

        print(f"\nTOP-{top_n_features} features (gain/weight):")
        print(df_feat[["feature", "gain", "weight"]].to_string(
            index=False,
            formatters={"gain": lambda g: f"{g:,.0f}", "weight": lambda w: f"{int(w):d}"}
        ))

        y_pos = np.arange(len(df_feat))
        plt.figure(figsize=(10, 0.35 * len(df_feat) + 3))
        plt.barh(y_pos - bar_width / 2, df_feat["gain_scaled"], height=bar_width, label=f"gain ÷{scale_factor}", color=cmap[0])
        plt.barh(y_pos + bar_width / 2, df_feat["weight"], height=bar_width, label="weight", color=cmap[1])
        plt.yticks(y_pos, df_feat["feature"])
        plt.xlabel("scaled value")
        plt.title(f"XGBoost – TOP {top_n_features} features (gain vs weight)")
        plt.legend(); plt.tight_layout(); plt.show()
        return df_feat

    @staticmethod
    def plot_forecast_vs_actual_with_metrics(
        y_true: pd.Series,
        y_pred: pd.Series,
        model_name: str,
        *,
        title_prefix: str = "Forecast vs actual (last fold)",
        save_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """Draw overlay plot and print MAE/RMSE/MAPE."""
        mae = Metrics.mae(y_true, y_pred)
        rmse = Metrics.rmse(y_true, y_pred)
        mape = Metrics.mape(y_true, y_pred)

        plt.figure(figsize=(12, 4))
        plt.plot(y_true, label="Actual", linewidth=2, color="black")
        plt.plot(y_pred, "--", label=(f"Forecast – {model_name}\nMAE={mae:.2f} | RMSE={rmse:.2f} | MAPE={mape:.2f}%"))
        plt.title(f"{title_prefix} – {model_name}")
        plt.legend(); plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120); plt.close()
        else:
            plt.show()
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    @staticmethod
    def plot_permutation_importance(model_obj, features_dataframe, target_column_name, top_n_features: int = 20):
        """Model-agnostic permutation importance (higher = more important)."""
        result = permutation_importance(
            model_obj,
            features_dataframe.drop(columns=[target_column_name]),
            features_dataframe[target_column_name],
            n_repeats=10,
            scoring="neg_mean_absolute_error",
            random_state=42,
            n_jobs=8,
        )
        idx = np.argsort(result.importances_mean)[::-1][:top_n_features]
        names = features_dataframe.drop(columns=[target_column_name]).columns[idx]
        scores = result.importances_mean[idx]

        print(f"\nPermutation importance (TOP-{top_n_features}):")
        for f, v in zip(names, scores):
            print(f"{f:35s} {v:.4f}")

        plt.figure(figsize=(8, 4))
        plt.barh(names[::-1], scores[::-1])
        plt.xlabel("MAE increase after permutation")
        plt.title("Permutation importance")
        plt.tight_layout(); plt.show()

    @staticmethod
    def analyze_residuals(y_true, y_pred, lags: int = 48):
        """Residual diagnostics: time plot, histogram, ACF, QQ-plot."""
        residuals = (y_true - y_pred).dropna()
        if len(residuals) < 2:
            print("Not enough observations for residual diagnostics.")
            return
        lags = min(lags, len(residuals) - 1)

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        axes[0, 0].plot(residuals.values); axes[0, 0].set_title("Residuals over time")
        axes[0, 1].hist(residuals, bins=30, edgecolor="k"); axes[0, 1].set_title("Histogram")
        plot_acf(residuals, lags=lags, ax=axes[1, 0]); axes[1, 0].set_title("ACF")
        sm.qqplot(residuals, line="s", ax=axes[1, 1]); axes[1, 1].set_title("QQ-plot")
        plt.tight_layout(); plt.show()

    @staticmethod
    def permutation_exogenous_columns(
        fitted_model: BaseEstimator,
        full_df: pd.DataFrame,
        target_column: str,
        exogenous_columns: List[str],
        metric=Metrics.mae,
        n_repeats: int = 5,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Permutation importance for SARIMAX exogenous variables (delta MAE)."""
        X_full = full_df.drop(columns=[target_column])
        y_full = full_df[target_column]
        base_mae = metric(y_full, fitted_model.predict(X_full))

        rng = np.random.default_rng(random_state)
        results = []
        for col in exogenous_columns:
            maes = []
            X_perm = X_full.copy()
            for _ in range(n_repeats):
                X_perm[col] = rng.permutation(X_perm[col].values)
                maes.append(metric(y_full, fitted_model.predict(X_perm)))
            delta = np.mean(maes) - base_mae
            results.append((col, delta))

        df_imp = (
            pd.DataFrame(results, columns=["feature", "delta_MAE"]).sort_values("delta_MAE", ascending=False).reset_index(drop=True)
        )
        print("\nPermutation importance (SARIMAX)")
        print(df_imp.to_string(index=False, formatters={"delta_MAE": "{:,.4f}".format}))
        return df_imp

    @staticmethod
    def summarize_results(ctx: RunContext, save_to_csv: bool = True) -> None:
        if not ctx.all_results:
            print("No aggregated results available.")
            return
        df_rank = pd.DataFrame(ctx.all_results).sort_values("MAE").reset_index(drop=True)
        print("\nModel ranking (7-day horizon)")
        print(df_rank.to_string(index=False, formatters={
            "MAE": "{:.2f}".format, "RMSE": "{:.2f}".format, "MAPE": "{:.2f}".format, "RMSLE": "{:.4f}".format, "R2": "{:.4f}".format
        }))
        if save_to_csv:
            RESULTS_DIR.mkdir(exist_ok=True, parents=True)
            df_rank.to_csv(RESULTS_DIR / "ranking_models.csv", index=False)

        if ctx.last_fold is not None:
            full = ctx.full_series
            y_true_last, preds_dict = ctx.last_fold

            plt.figure(figsize=(11, 4))
            if full is not None:
                plt.plot(full.index, full, color="lightgray", label="Original full series")
            plt.plot(y_true_last.index, y_true_last, color="black", linewidth=2, label="True – last fold")
            for name, pred in preds_dict.items():
                plt.plot(pred.index, pred, label=name, alpha=0.8)
            plt.legend(); plt.title("Original data and forecasts – last fold")
            plt.tight_layout(); plt.savefig(RESULTS_DIR / "plot_LAST_FOLD_ALL.png", dpi=120); plt.close()

            for name, pred in preds_dict.items():
                plt.figure(figsize=(11, 4))
                if full is not None:
                    plt.plot(full.index, full, color="lightgray", label="Original full series")
                plt.plot(y_true_last.index, y_true_last, color="black", linewidth=2, label="True – last fold")
                plt.plot(pred.index, pred, label=name, alpha=0.8)
                plt.legend(); plt.title(f"Last fold – {name}")
                fname = RESULTS_DIR / f"plot_LAST_FOLD_{name.replace('+', '_')}.png"
                plt.tight_layout(); plt.savefig(fname, dpi=120); plt.close()


class SarimaX(BaseEstimator, RegressorMixin):
    """SARIMA/SARIMAX wrapper with safe fallback (mean) and optional truncation for long series."""
    def __init__(
        self,
        exogenous_columns: Optional[List[str]] = None,
        order: Tuple[int, int, int] = (1, 0, 2),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24),
        maximum_observations: Optional[int] = None,
    ):
        self.exogenous_columns = exogenous_columns
        self.order = order
        self.seasonal_order = seasonal_order
        self.maximum_observations = maximum_observations
        self.model_ = None
        self.use_fallback = False
        self.fallback_mean_value = None

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
            print(f"SARIMA fallback due to exception: {str(e)} – using 24h mean")
        return self

    def predict(self, X: pd.DataFrame):
        steps = len(X)
        if self.use_fallback or self.model_ is None:
            return pd.Series(np.full(steps, self.fallback_mean_value), index=X.index)
        exog = X[self.exogenous_columns] if self.exogenous_columns else None
        fc = self.model_.forecast(steps=steps, exog=exog)
        return pd.Series(fc.values, index=X.index)

class HybridResidualModel(BaseEstimator, RegressorMixin):
    """Train a base model with walk-forward; fit a correction model on OOF residuals; retrain base on full data."""
    def __init__(self, base_model, residual_correction_model, n_splits: int = 5, model_name: str = "hybrid"):
        self.base_model = base_model
        self.residual_correction_model = residual_correction_model
        self.n_splits = n_splits
        self.model_name = model_name
        self.trained_base_model = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        oof_pred = pd.Series(index=y.index, dtype=float)
        fallback_mask = pd.Series(False, index=y.index)
        for tr_idx, va_idx in tscv.split(X):
            base_clone = copy.deepcopy(self.base_model)
            base_clone.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            oof_pred.iloc[va_idx] = base_clone.predict(X.iloc[va_idx])
            fallback_mask.iloc[va_idx] = getattr(base_clone, "use_fallback", False)

        good_mask = oof_pred.notna() & ~fallback_mask
        if good_mask.sum() < 100:
            warnings.warn(f"[{self.model_name}] too little data for residual correction – using base model only.")
            self.residual_correction_model = None
        else:
            residuals = (y - oof_pred).loc[good_mask]
            X_corr = X.loc[good_mask].select_dtypes(include=[np.number])
            X_corr = X_corr.loc[:, ~X_corr.columns.duplicated()]
            self.residual_correction_model.fit(X_corr, residuals)

        self.trained_base_model = copy.deepcopy(self.base_model).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        base_pred = self.trained_base_model.predict(X)
        if self.residual_correction_model is None:
            return base_pred
        Xf = X.select_dtypes(include=[np.number]).loc[:, lambda d: ~d.columns.duplicated()]
        return base_pred + self.residual_correction_model.predict(Xf)

class GradientBoostingModel(BaseEstimator, RegressorMixin):
    """GradientBoostingRegressor with two modes: early-stopping-like (ES) and randomized search (RS)."""
    def __init__(
        self,
        mode: str = "es",
        n_iter_rs: int = 20,
        cross_validation_splits: int = 5,
        max_boosting_rounds: int = 800,
        validation_fraction: float = 0.1,
        random_seed: int = 42,
        learning_rate: float = 0.015,
        maximum_tree_depth: int = 4,
        subsample_ratio: float = 0.8,
        minimum_samples_per_leaf: int = 15,
    ):
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
            raise ValueError("GBR has 0 columns – no features to train on.")
        if X.shape[0] == 0:
            raise ValueError("GBR has 0 rows – no observations to train on.")

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
            print(f"GBR_ES: best number of trees = {best_iter} (MAE val = {mae_val[best_iter-1]:.2f})")

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
    """XGBRegressor with ES via xgb.cv or RS via sklearn RandomizedSearchCV."""
    def __init__(
        self,
        mode: str = "es",
        n_iter_rs: int = 250,  # sensible default
        cross_validation_splits: int = 5,
        early_stopping_folds: int = 3,
        max_boosting_rounds: int = 5000,
        early_stopping_rounds: int = 50,
        random_seed: int = 42,
    ):
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
            print(f"XGB_ES: best number of trees = {best_iter} (MAE cv = {cv['test-mae-mean'].iloc[-1]:.2f})")
            self.model_ = XGBRegressor(
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
            base = XGBRegressor(objective="reg:squarederror", n_jobs=8, random_state=self.random_seed)
            tscv = TimeSeriesSplit(n_splits=self.cross_validation_splits, test_size=DEFAULT_TEST_HORIZON)
            rs = RandomizedSearchCV(base, param_grid, n_iter=self.n_iter_rs, scoring="neg_mean_absolute_error", cv=tscv, n_jobs=8, verbose=0).fit(X, y)
            print("XGB_RS: best params:", rs.best_params_)
            self.model_ = rs.best_estimator_
        return self

    def predict(self, X: pd.DataFrame):
        return self.model_.predict(X)

def run_holdout_90Train_10Test(
    models: dict[str, BaseEstimator],
    features_dataframe: pd.DataFrame,
    target_column: str,
    sarima_dataset: pd.DataFrame,
    sarimax_dataset: pd.DataFrame | None,
    save_csv_path: str = "results/predictions_holdout_90_10.csv",
    ctx: RunContext | None = None,
) -> pd.DataFrame:
    """
    Train on the first 90% of the sample and forecast the last 10% (hold-out) for every model.
    Returns a merged DataFrame with the common test index and columns:
      - 'y_true'   : observed target on the hold-out
      - one column per model name with its predictions
    Also writes:
      - predictions CSV -> `save_csv_path` (index labeled as 'timestamp')
      - ranking CSV     -> 'results/ranking_90_10.csv' (MAE/RMSE/MAPE)
    """
    if ctx is None:
        ctx = RunContext()

    n = len(features_dataframe)
    if n < 10:
        raise ValueError("Not enough observations for a 90/10 split.")

    split_idx = int(n * 0.9)
    test_index = features_dataframe.index[split_idx:]

    # y_true is always taken from the full features_dataframe to ensure a single reference
    y_true_test = features_dataframe.loc[test_index, target_column].astype(float)

    predictions_dict: dict[str, pd.Series] = {}
    ranking_local: list[dict] = []

    for name, model_object in models.items():
        # Select the appropriate input table
        if name == "SARIMAX":
            input_df = sarimax_dataset
        elif name == "SARIMA":
            input_df = sarima_dataset
        else:
            input_df = features_dataframe

        if input_df is None:
            print(f"[WARN] Skipping {name} – required input table is None.")
            continue

        # Basic column checks
        if target_column not in input_df.columns:
            print(f"[WARN] Skipping {name} – target '{target_column}' missing in its input table.")
            continue

        # Train / test split on the chosen table
        try:
            train_features = input_df.iloc[:split_idx].drop(columns=[target_column])
            train_target   = input_df.iloc[:split_idx][target_column].astype(float)
            test_features  = input_df.loc[test_index].drop(columns=[target_column])

            # Fit & predict
            model_object.fit(train_features, train_target)
            y_pred = pd.Series(model_object.predict(test_features), index=test_index, name=name).astype(float)
            predictions_dict[name] = y_pred

            # Metrics on the common y_true
            mae  = Metrics.mae(y_true_test, y_pred)
            rmse = Metrics.rmse(y_true_test, y_pred)
            mape = Metrics.mape(y_true_test, y_pred)
            print(f"[90/10] {name}: MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")

            ranking_local.append(dict(model=name, MAE=mae, RMSE=rmse, MAPE=mape))
            ctx.preds_by_model[name] = y_pred  # for later plots
        except Exception as e:
            print(f"[ERROR] {name} failed on 90/10 hold-out: {e!s}")

    # Merge into a single table (common test index)
    result_df = pd.DataFrame({"y_true": y_true_test})
    for name, pred in predictions_dict.items():
        result_df[name] = pred.reindex(result_df.index)

    # Save predictions
    Path(save_csv_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(save_csv_path, index_label="timestamp")
    print(f"[INFO] Saved hold-out predictions -> {save_csv_path}")

    # Save ranking
    if ranking_local:
        rank_df = (pd.DataFrame(ranking_local)
                   .sort_values("MAE")
                   .reset_index(drop=True))
        Path("results").mkdir(exist_ok=True)
        rank_df.to_csv("results/ranking_90_10.csv", index=False)
        print("\nRanking 90/10")
        print(rank_df.to_string(index=False,
                                formatters={"MAE": "{:.2f}".format,
                                            "RMSE": "{:.2f}".format,
                                            "MAPE": "{:.2f}".format}))
    else:
        print("[WARN] No 90/10 ranking metrics were produced.")

    # Store last fold (the hold-out) in context for plotting
    if predictions_dict:
        ctx.last_fold = (y_true_test, predictions_dict)

    return result_df


def build_model_registry(exogenous_cols: Optional[List[str]]):
    return {
        "XGB_ES": XGBoostModel(mode="es"),
        "GBR_ES": GradientBoostingModel(mode="es"),
        # "GBR_RS": GradientBoostingModel(mode="rs", n_iter_rs=20, cross_validation_splits=5),
        # "XGB_RS": XGBoostModel(mode="rs", n_iter_rs=50, cross_validation_splits=5),
        "SARIMA": SarimaX(exogenous_columns=None, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24), maximum_observations=20000),
        "SARIMAX": SarimaX(exogenous_columns=exogenous_cols, order=(2, 0, 1), seasonal_order=(1, 1, 0, 24), maximum_observations=8000),
        # "SARIMA_HYB_GBR": HybridResidualModel(
        #     base_model=SarimaX(exogenous_columns=None, order=(1, 0, 1), seasonal_order=(1, 1, 0, 24), maximum_observations=23000),
        #     residual_correction_model=GradientBoostingModel(mode="es"),model_name="SARIMA_HYB_GBR",),
        # "SARIMAX_HYB_XGB": HybridResidualModel(
        #     base_model=SarimaX(exogenous_columns=exogenous_cols, order=(2, 0, 1), seasonal_order=(1, 1, 0, 24), maximum_observations=20000),
        #     residual_correction_model=XGBoostModel(mode="es"),model_name="SARIMAX_HYB_XGB",),
    }


def main() -> None:
    warnings.filterwarnings("ignore")

    # 1) Load features
    features_dataframe = load_features_table(FEATURES_FILE)
    rows_before = len(features_dataframe)
    features_dataframe = features_dataframe.dropna(subset=[TARGET_COLUMN])
    print(f"Dropped {rows_before - len(features_dataframe)} rows with missing '{TARGET_COLUMN}'")

    # 2) Exogenous selection
    exogenous_cols = select_exogenous_columns(features_dataframe, TARGET_COLUMN)

    # 3) Split views for models
    sarima_dataset = features_dataframe[[TARGET_COLUMN]]
    sarimax_dataset = features_dataframe[[TARGET_COLUMN] + exogenous_cols] if exogenous_cols else None

    # 4) Registry & run CV
    ctx = RunContext()
    ctx.full_series = features_dataframe[TARGET_COLUMN]

    models = build_model_registry(exogenous_cols)
    for name, model_obj in models.items():
        if name.startswith("SARIMAX") and "+" not in name:
            input_df = sarimax_dataset
        elif name == "SARIMA":
            input_df = sarima_dataset
        else:
            input_df = features_dataframe

        Metrics.validate_time_series_model(
            input_df,
            TARGET_COLUMN,
            model_obj,
            test_horizon=DEFAULT_TEST_HORIZON,
            model_name=name,
            ctx=ctx,
        )

        if name in ctx.preds_by_model:
            y_pred_last = ctx.preds_by_model[name]
            y_true_last = features_dataframe.loc[y_pred_last.index, TARGET_COLUMN]
            AnalysisTools.plot_forecast_vs_actual_with_metrics(y_true=y_true_last, y_pred=y_pred_last, model_name=name)

    #5 Predictions for holdout 90/10
    _ = run_holdout_90Train_10Test(
        models=models,
        features_dataframe=features_dataframe,
        target_column=TARGET_COLUMN,
        sarima_dataset=sarima_dataset,
        sarimax_dataset=sarimax_dataset,
        save_csv_path="results/predictions_holdout_90_10.csv",
        ctx=ctx,
    )

    # 6) Feature importance & residuals for XGB
    if "XGB_ES" in models:
        xgb_es = models["XGB_ES"]
        AnalysisTools.top_features_by_gain_weight(xgb_es, top_n_features=25)
        AnalysisTools.plot_permutation_importance(xgb_es.model_, features_dataframe, target_column_name=TARGET_COLUMN, top_n_features=25)
        if ctx.last_fold:
            y_true_last, preds_dict = ctx.last_fold
            for model_name, y_pred in preds_dict.items():
                print(f"\nResidual analysis for: {model_name}")
                AnalysisTools.analyze_residuals(y_true_last, y_pred, lags=48)

    # 7) Compare GBR vs XGB top features (optional)
    if models.get("GBR_ES") and models.get("XGB_ES"):
        try:
            AnalysisTools.compare_top_features_gbr_xgb(
                features_dataframe=features_dataframe,
                target_column=TARGET_COLUMN,
                gbr_estimator=models["GBR_ES"],
                xgb_estimator=models["XGB_ES"],
                top_n=20,
                save_csv_path=str(RESULTS_DIR / "top_features_gbr_xgb.csv"),
            )
        except Exception as e:
            print(f"[WARN] Failed to build GBR vs XGB top-20 comparison: {e}")

    # 8) SARIMAX exogenous permutation importance
    if "SARIMAX" in models and sarimax_dataset is not None and exogenous_cols:
        sarimax_model = models["SARIMAX"]
        AnalysisTools.permutation_exogenous_columns(
            fitted_model=sarimax_model,
            full_df=sarimax_dataset,
            target_column=TARGET_COLUMN,
            exogenous_columns=exogenous_cols,
            n_repeats=10,
            random_state=42,
        )

    # 9) Summary
    AnalysisTools.summarize_results(ctx, save_to_csv=True)


if __name__ == "__main__":
    main()
