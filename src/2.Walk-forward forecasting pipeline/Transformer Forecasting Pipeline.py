
from __future__ import annotations


from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


import inspect
import json
import math
import os


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, TensorDataset  # local import

# =========================
# Paths and configuration
# =========================


@dataclass
class ProjectPaths:
    """Resolve and hold all project paths used by this module.

    By default results are saved under "results_transformer" to keep backward
    compatibility with the original script.
    """

    project_root: Path
    data_dir: Path
    processed_dir: Path
    merged_predictions_dir: Path
    results_dir: Path
    plots_dir: Path
    tables_dir: Path
    checkpoints_dir: Path

    @staticmethod
    def bootstrap() -> "ProjectPaths":
        """Discover project root and build standard directories.

        The root is detected by walking up for common markers. If not found, it
        falls back to current working directory.
        """
        try:
            here = Path(__file__).resolve()
        except NameError:
            here = Path.cwd().resolve()

        root: Optional[Path] = None
        for p in [here] + list(here.parents):
            markers = (p / "data", p / "README.md", p / ".git", p / "requirements.txt")
            if any(m.exists() for m in markers):
                root = p
                break
        if root is None:
            env_root = os.environ.get("PROJECT_ROOT")
            root = Path(env_root).resolve() if env_root else here

        data_dir = root / "data"
        processed_dir = data_dir / "processed"
        merged_predictions_dir = data_dir / "merged predictions"
        results_dir = root / "results"
        plots_dir = results_dir / "plots"
        tables_dir = results_dir / "tables"
        checkpoints_dir = results_dir / "checkpoints"

        for d in (
            processed_dir,
            merged_predictions_dir,
            results_dir,
            plots_dir,
            tables_dir,
            checkpoints_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        return ProjectPaths(
            project_root=root,
            data_dir=data_dir,
            processed_dir=processed_dir,
            merged_predictions_dir=merged_predictions_dir,
            results_dir=results_dir,
            plots_dir=plots_dir,
            tables_dir=tables_dir,
            checkpoints_dir=checkpoints_dir,
        )


@dataclass
class RunConfig:
    """Run-time configuration switches for the pipeline.

    All fields are explicit to avoid hidden state. This mirrors original flags
    and constants, only grouped in one place.
    """

    data_path: str
    target_column: str = "fixing_i_price"
    seed: int = 42

    # Tuning
    tune_with_optuna: bool = False
    optuna_trials: int = 45

    # Evaluation
    run_horizons_kfold: bool = True
    n_folds_for_horizons: int = 3
    horizons_hours: Optional[List[int]] = None  # e.g. [24, 168, 336, 744]

    # Recursive blocks
    recursive_block_hours: int = 24 * 7
    recursive_n_blocks: int = 10

    # Model defaults
    lookback: int = 336

    def to_json(self) -> str:
        """Serialize configuration to JSON string for logging."""
        return json.dumps(self.__dict__, indent=2)


# =========================
# Data
# =========================


class DataModule:
    """Load and normalize the feature table for time-series forecasting.

    Responsibilities
    - Read a parquet file from local disk or GCS.
    - Ensure DatetimeIndex and uniform hourly frequency.
    - Forward fill missing values after dropping missing target rows.
    - Drop zero-variance columns for numerical stability.
    """

    def __init__(self, target_column: str) -> None:
        self.target_column = target_column

    def load_dataframe(self, data_uri: str) -> pd.DataFrame:
        """Load a parquet table and prepare it for modeling.

        Parameters
        data_uri: str
            Local path or a GCS URI starting with "gs://".
        """
        is_gcs = str(data_uri).startswith("gs://")
        try:
            if is_gcs:
                # Requires gcsfs installed in the runtime
                df = pd.read_parquet(data_uri, storage_options={"token": "cloud"})
            else:
                df = pd.read_parquet(data_uri)
        except FileNotFoundError as e:
            # Extra diagnostics for GCS paths
            if is_gcs:
                try:
                    import gcsfs  # type: ignore

                    fs = gcsfs.GCSFileSystem()
                    bucket, _, prefix = data_uri[5:].partition("/")
                    base = prefix.rsplit("/", 1)[0]
                    print(f"[DIAG] Listing gs://{bucket}/{base}/")
                    for p in fs.ls(f"{bucket}/{base}"):
                        print(" -", p)
                except Exception as diag_err:  # noqa: BLE001
                    print("[DIAG] Could not list GCS path:", diag_err)
            raise e

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
            else:
                raise ValueError(
                    "Parquet must contain DatetimeIndex or 'datetime'/'timestamp' column"
                )

        # Normalize frequency and fill
        df = df.asfreq("h")
        df = df.dropna(subset=[self.target_column]).fillna(method="ffill")

        # Drop zero-variance columns
        constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
        if constant_cols:
            print("Removing zero-variance columns:", constant_cols)
            df = df.drop(columns=constant_cols)

        return df


# =========================
# Metrics
# =========================


class Metrics:
    """Safe metric helpers that ignore NaNs where appropriate."""

    @staticmethod
    def mae(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> float:
        y_pred_ser = pd.Series(y_pred, index=y_true.index)
        mask = ~np.isnan(y_pred_ser.values)
        if mask.sum() == 0:
            return float("nan")
        return float(np.mean(np.abs(y_true.values[mask] - y_pred_ser.values[mask])))

    @staticmethod
    def rmse(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> float:
        y_pred_ser = pd.Series(y_pred, index=y_true.index)
        mask = ~np.isnan(y_pred_ser.values)
        if mask.sum() == 0:
            return float("nan")
        diff = y_true.values[mask] - y_pred_ser.values[mask]
        return float(np.sqrt(np.mean(diff ** 2)))

    @staticmethod
    def mape(y_true: pd.Series, y_pred: pd.Series | np.ndarray, eps: float = 1e-8) -> float:
        y_pred_ser = pd.Series(y_pred, index=y_true.index)
        mask = (~np.isnan(y_pred_ser.values)) & (np.abs(y_true.values) >= eps)
        if mask.sum() == 0:
            return float("nan")
        rel = np.abs((y_true.values[mask] - y_pred_ser.values[mask]) / y_true.values[mask])
        return float(np.mean(rel) * 100.0)

    @staticmethod
    def rmsle(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> float:
        y_pred_ser = pd.Series(y_pred, index=y_true.index)
        mask = (~np.isnan(y_pred_ser.values)) & (y_true.values >= 0) & (y_pred_ser.values >= 0)
        if mask.sum() == 0:
            return float("nan")
        yt = np.log1p(y_true.values[mask])
        yp = np.log1p(y_pred_ser.values[mask])
        return float(np.sqrt(np.mean((yt - yp) ** 2)))

    @staticmethod
    def r2(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> float:
        y_pred_ser = pd.Series(y_pred, index=y_true.index)
        mask = ~np.isnan(y_pred_ser.values)
        if mask.sum() <= 1:
            return float("nan")
        # Manual R2 to avoid importing sklearn here
        yt = y_true.values[mask]
        yp = y_pred_ser.values[mask]
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


# =========================
# Trainer and context
# =========================




class RunContext:
    """Simple container for last-fold artifacts and aggregated results."""

    def __init__(self) -> None:
        self.last_fold: Optional[Tuple[pd.Series, Dict[str, pd.Series]]] = None
        self.all_results: List[dict] = []
        self.preds_by_model: Dict[str, pd.Series] = {}


class Trainer:
    """Helpers for training and validation."""

    @staticmethod
    def set_global_seed(seed: int = 42) -> None:
        """Set seeds for reproducibility in numpy and torch."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True

    def validate_model(
        self,
        df_input: pd.DataFrame,
        target_col: str,
        model: "SeqTransformerRegressor",
        test_horizon: int,
        n_splits: int,
        model_name: str = "",
        ctx: Optional[RunContext] = None,
        compute_importance: bool = False,
    ) -> float:
        """Time-series CV validation identical to the original helper.

        Splits the series into n_splits with fixed test_size=test_horizon and
        averages MAE, RMSE, MAPE, RMSLE, R2 across folds.
        """
        if ctx is None:
            ctx = RunContext()
        if not model_name:
            model_name = model.__class__.__name__

        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_horizon)
        fold_mae: List[float] = []
        fold_rmse: List[float] = []
        fold_mape: List[float] = []
        fold_rmsle: List[float] = []
        fold_r2: List[float] = []

        for tr_idx, te_idx in tscv.split(df_input):
            train_df = df_input.iloc[tr_idx]
            test_df = df_input.iloc[te_idx]

            X_tr, y_tr = train_df.drop(columns=[target_col]), train_df[target_col]
            X_te, y_te = test_df.drop(columns=[target_col]), test_df[target_col]

            model.fit(X_tr, y_tr)
            print(
                f"[CHECK] model.fitted={getattr(model, 'fitted', False)} | "
                f"epochs_run={getattr(model, '_epochs_run', '?')} | "
                f"best_val_MAE={getattr(model, '_best_val_mae_orig', float('nan')):.3f}"
            )

            lookback = getattr(model, "lookback", 1)
            X_ctx = pd.concat([X_tr.tail(max(0, lookback - 1)), X_te], axis=0)
            y_pred = model.predict(X_ctx).loc[X_te.index]

            fold_mae.append(Metrics.mae(y_te, y_pred))
            fold_rmse.append(Metrics.rmse(y_te, y_pred))
            fold_mape.append(Metrics.mape(y_te, y_pred))
            fold_rmsle.append(Metrics.rmsle(y_te, y_pred))
            fold_r2.append(Metrics.r2(y_te, y_pred))

            ctx.preds_by_model[model_name] = pd.Series(y_pred, index=y_te.index)
            ctx.last_fold = (y_te, {model_name: pd.Series(y_pred, index=y_te.index)})

            print(f"[CHECK] y_pred NaNs: {int(np.isnan(y_pred.values).sum())} / {len(y_pred)}")
            print(
                f"[CHECK] y_pred stats: min={np.nanmin(y_pred.values):.3f}, "
                f"max={np.nanmax(y_pred.values):.3f}, mean={np.nanmean(y_pred.values):.3f}"
            )

            if compute_importance and len(fold_mae) == n_splits:
                try:
                    _ = FeatureImportance.permutation_last_h(
                        model=model,
                        X_tr=X_tr,
                        X_te=X_te,
                        y_te=y_te,
                        metric_fn=Metrics.mae,
                        n_repeats=3,
                        top_k=20,
                    )
                except Exception as e:  # noqa: BLE001
                    print("[WARN] Permutation importance failed:", e)

        mean_mae = float(np.nanmean(fold_mae))
        mean_rmse = float(np.nanmean(fold_rmse))
        mean_mape = float(np.nanmean(fold_mape))
        mean_rmsle = float(np.nanmean(fold_rmsle))
        mean_r2 = float(np.nanmean(fold_r2))
        print(
            f"{model_name}: MAE={mean_mae:.3f} | RMSE={mean_rmse:.3f} | "
            f"MAPE={mean_mape:.2f}% | RMSLE={mean_rmsle:.3f} | R²={mean_r2:.3f}"
        )
        ctx.all_results.append(
            dict(model=model_name, MAE=mean_mae, RMSE=mean_rmse, MAPE=mean_mape, RMSLE=mean_rmsle, R2=mean_r2)
        )
        return mean_mae


# =========================
# Model and adapter
# =========================


class PositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding for sequences."""

    def __init__(self, d_model: int, max_len: int = 20000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        return x + self.pe[:, :length, :]


def make_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """Build an upper-triangular causal mask (L x L)."""
    return torch.triu(torch.ones(length, length, device=device) * float("-inf"), diagonal=1)



class SeqTransformerRegressor(nn.Module):
    """Simple Transformer regressor for univariate target with exogenous features.

    Features
    - Standardization of X and signed_log1p standardization of y.
    - Sliding windows over lookback.
    - Causal attention mask.
    - Aggregation strategy: last or mean over tokens.
    - EMA weights and top-K snapshots (mini-SWA style).
    - Optional MC-Dropout sampling at predict time.
    """

    # Defaults similar to the original script
    SNAPSHOT_TOP_K_DEFAULT: int = 3
    EMA_DECAY_DEFAULT: float = 0.999
    MC_DROPOUT_PASSES_DEFAULT: int = 1
    FEATURE_NOISE_STD_DEFAULT: float = 0.01

    def __init__(
        self,
        embed_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        ff_dim: int = 512,
        dropout: float = 0.10,
        batch_size: int = 32,
        epochs: int = 130,
        lr: float = 5e-4,
        weight_decay: float = 1e-5,
        lookback: int = 336,
        patience: int = 30,
        aggregation: str = "last",
        amp: bool = True,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.lookback = lookback
        self.patience = patience
        self.aggregation = aggregation.lower()
        self.amp = amp
        self._epochs_run: int = 0
        self._best_epoch: int = 0
        self._best_val_mae_orig: float = float("inf")

        if self.aggregation not in {"last", "mean"}:
            raise ValueError("aggregation must be 'last' or 'mean'")

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Will be initialized lazily in _init_network
        self.input_proj: Optional[nn.Linear] = None
        self.pos_enc: Optional[PositionalEncoding] = None
        self.encoder: Optional[nn.TransformerEncoder] = None
        self.final_norm: Optional[nn.LayerNorm] = None
        self.output_proj: Optional[nn.Linear] = None

        # Scalers (fitted on training split only)
        self.scaler_X: Optional["StandardScaler"] = None
        self.scaler_y: Optional["StandardScaler"] = None

        # Fit flag and index cache
        self.fitted: bool = False
        self._index: Optional[pd.DatetimeIndex] = None

        # Cached causal mask
        self._causal_mask_cpu = make_causal_mask(self.lookback, torch.device("cpu"))

        # Time embeddings
        self.use_time_embeddings: bool = True
        self.hour_emb: Optional[nn.Embedding] = (nn.Embedding(24, self.embed_dim).to(self.device)
                                                 if self.use_time_embeddings else None)
        self.dow_emb: Optional[nn.Embedding] = (nn.Embedding(7, self.embed_dim).to(self.device)
                                                if self.use_time_embeddings else None)
        self.time_proj: Optional[nn.Linear] = (nn.Linear(self.embed_dim * 3, self.embed_dim).to(self.device)
                                               if self.use_time_embeddings else None)

        # EMA and snapshots
        self.use_ema: bool = True
        self.ema_decay: float = self.EMA_DECAY_DEFAULT
        self._ema_state: Optional[dict[str, torch.Tensor]] = None
        self.snapshot_top_k: int = self.SNAPSHOT_TOP_K_DEFAULT
        self._snapshots: List[Tuple[float, dict[str, torch.Tensor]]] = []

        # MC-Dropout and light feature noise
        self.mc_dropout_passes: int = self.MC_DROPOUT_PASSES_DEFAULT
        self.feature_noise_std: float = self.FEATURE_NOISE_STD_DEFAULT

    # -------- utilities --------
    @staticmethod
    def signed_log1p(x: np.ndarray | pd.Series) -> np.ndarray:
        """Scale values with sign-preserving log1p."""
        arr = np.asarray(x, dtype=np.float32)
        return np.sign(arr) * np.log1p(np.abs(arr))

    @staticmethod
    def inv_signed_log1p(x: np.ndarray | pd.Series) -> np.ndarray:
        """Inverse of signed_log1p."""
        arr = np.asarray(x, dtype=np.float32)
        return np.sign(arr) * np.expm1(np.abs(arr))

    def _build_windows(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding windows of length lookback over X and aligned target y."""
        L = self.lookback
        if len(X) < L:
            raise ValueError(f"Series shorter than lookback={L}")
        X_win = np.lib.stride_tricks.sliding_window_view(X, window_shape=(L, X.shape[1]))
        X_win = X_win.reshape(-1, L, X.shape[1])
        y_trim = y[L - 1:]
        return X_win, y_trim

    @staticmethod
    def _build_time_indices(index: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
        """Return arrays of hours and day-of-week for time embeddings."""
        hours = index.hour.values.astype(np.int64)
        dows = index.dayofweek.values.astype(np.int64)
        return hours, dows

    def _init_network(self, n_features: int) -> None:
        """Instantiate layers once the feature count is known."""
        self.input_proj = nn.Linear(n_features, self.embed_dim).to(self.device)
        self.pos_enc = PositionalEncoding(self.embed_dim, max_len=max(20000, self.lookback + 1)).to(self.device)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.nhead,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers).to(self.device)
        self.final_norm = nn.LayerNorm(self.embed_dim).to(self.device)
        self.output_proj = nn.Linear(self.embed_dim, 1).to(self.device)

        if self.use_time_embeddings:
            if self.hour_emb is not None:
                self.hour_emb = self.hour_emb.to(self.device)
            if self.dow_emb is not None:
                self.dow_emb = self.dow_emb.to(self.device)
            if self.time_proj is not None:
                self.time_proj = self.time_proj.to(self.device)

    # EMA helpers
    @torch.no_grad()
    def _ema_init_from_current(self) -> None:
        self._ema_state = {k: v.detach().clone().cpu() for k, v in self.state_dict().items()}

    @torch.no_grad()
    def _ema_update(self) -> None:
        if self._ema_state is None:
            self._ema_init_from_current()
            return
        d = self.state_dict()
        for k, v in d.items():
            v_cpu = v.detach().cpu()
            self._ema_state[k].mul_(self.ema_decay).add_(v_cpu, alpha=1.0 - self.ema_decay)

    def _load_state(self, state: dict[str, torch.Tensor]) -> None:
        self.load_state_dict({k: v.to(self.device) for k, v in state.items()}, strict=True)

    # Snapshots
    @staticmethod
    def _avg_states(states: List[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for k in states[0].keys():
            acc = None
            for st in states:
                t = st[k]
                acc = t.clone() if acc is None else acc.add_(t)
            out[k] = acc.div_(len(states))
        return out

    def _maybe_push_snapshot(self, mae_val: float, state: dict[str, torch.Tensor]) -> None:
        self._snapshots.append((mae_val, {k: v.detach().clone().cpu() for k, v in state.items()}))
        self._snapshots.sort(key=lambda x: x[0])
        if len(self._snapshots) > self.snapshot_top_k:
            self._snapshots = self._snapshots[:self.snapshot_top_k]

    # Forward core
    def _forward_core(
        self,
        xb: torch.Tensor,
        hb: Optional[torch.Tensor] = None,
        db: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = self.input_proj(xb)
        if self.use_time_embeddings and (hb is not None) and (db is not None):
            hb = hb.to(self.device, non_blocking=True).long()
            db = db.to(self.device, non_blocking=True).long()
            he = self.hour_emb(hb)  # type: ignore[arg-type]
            de = self.dow_emb(db)   # type: ignore[arg-type]
            z = torch.cat([z, he, de], dim=-1)
            z = self.time_proj(z)   # type: ignore[call-arg]

        z = self.pos_enc(z)  # type: ignore[arg-type]
        mask = self._causal_mask_cpu.to(self.device)
        z = self.encoder(z, mask=mask)  # type: ignore[arg-type]

        if self.aggregation == "mean":
            z = z.mean(dim=1)
        else:
            z = z[:, -1, :]

        z = self.final_norm(z)  # type: ignore[arg-type]
        out = self.output_proj(z)  # type: ignore[arg-type]
        return out

    # Fit / predict
    def fit(self, X_df: pd.DataFrame, y_ser: pd.Series) -> "SeqTransformerRegressor":
        if X_df.isna().any().any():
            raise ValueError("NaN in X")
        if y_ser.isna().any():
            raise ValueError("NaN in y")

        self._index = X_df.index

        # Transform target
        y_log = self.signed_log1p(y_ser.values.astype(np.float32)).reshape(-1, 1)
        from sklearn.preprocessing import StandardScaler  # local import

        self.scaler_y = StandardScaler().fit(y_log)
        y_scaled = self.scaler_y.transform(y_log).astype(np.float32)

        # Scale features on training part
        self.scaler_X = StandardScaler().fit(X_df.values)
        X_scaled = self.scaler_X.transform(X_df.values).astype(np.float32)

        # Windows
        X_win, y_win = self._build_windows(X_scaled, y_scaled)

        # Time windows
        hours_all, dows_all = self._build_time_indices(self._index)
        h_win = np.lib.stride_tricks.sliding_window_view(hours_all, window_shape=self.lookback).astype(np.int64)
        d_win = np.lib.stride_tricks.sliding_window_view(dows_all, window_shape=self.lookback).astype(np.int64)

        # Validation split = last 10% of windows
        M = X_win.shape[0]
        val_len = max(1, int(0.1 * M))
        X_tr = X_win[:-val_len]
        y_tr = y_win[:-val_len]
        X_val = X_win[-val_len:]
        y_val = y_win[-val_len:]
        h_tr = h_win[:-val_len]
        d_tr = d_win[:-val_len]
        h_val = h_win[-val_len:]
        d_val = d_win[-val_len:]

        # Network
        n_features = X_scaled.shape[1]
        self._init_network(n_features)

        # EMA
        if self.use_ema:
            self._ema_init_from_current()

        # Data loaders


        train_ds = TensorDataset(
            torch.from_numpy(X_tr), torch.from_numpy(y_tr),
            torch.from_numpy(h_tr), torch.from_numpy(d_tr)
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(y_val),
            torch.from_numpy(h_val), torch.from_numpy(d_val)
        )

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=2, pin_memory=True, persistent_workers=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=2, pin_memory=True, persistent_workers=False,
        )

        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.L1Loss()
        steps_per_epoch = max(1, math.ceil(len(train_ds) / self.batch_size))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=10.0,
        )

        scaler = torch.cuda.amp.GradScaler(enabled=self.amp and self.device.type == "cuda")

        best_state = {k: v.cpu() for k, v in self.state_dict().items()}
        patience_left = self.patience

        self._epochs_run = 0
        self._best_epoch = 0
        self._best_val_mae_orig = float("inf")
        print(
            f"[TRAIN] device={self.device} | lookback={self.lookback} | "
            f"X.shape={X_df.shape} | y.shape={y_ser.shape}"
        )

        for epoch in range(1, self.epochs + 1):
            self.train()
            train_losses: List[float] = []

            for xb, yb, hb, db in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                hb = hb.to(self.device, non_blocking=True).long()
                db = db.to(self.device, non_blocking=True).long()

                if self.feature_noise_std > 0.0:
                    xb = xb + torch.randn_like(xb) * self.feature_noise_std

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=self.amp and self.device.type == "cuda"):
                    preds = self._forward_core(xb, hb, db)
                    loss = loss_fn(preds, yb)

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()

                if self.use_ema:
                    self._ema_update()

                train_losses.append(loss.detach().item())

            # Validation with EMA weights if enabled
            self.eval()
            saved_state: Optional[dict[str, torch.Tensor]] = None
            if self.use_ema and (self._ema_state is not None):
                saved_state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
                self._load_state(self._ema_state)

            val_losses: List[float] = []
            val_preds_scaled: List[np.ndarray] = []
            with torch.no_grad():
                for xb, yb, hb, db in val_loader:
                    xb = xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)
                    hb = hb.to(self.device, non_blocking=True).long()
                    db = db.to(self.device, non_blocking=True).long()
                    with torch.cuda.amp.autocast(enabled=self.amp and self.device.type == "cuda"):
                        preds = self._forward_core(xb, hb, db)
                        loss = loss_fn(preds, yb)
                    val_losses.append(loss.detach().item())
                    val_preds_scaled.append(preds.detach().cpu().numpy())

            if saved_state is not None:
                self._load_state(saved_state)

            # Metrics on original scale
            val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

            from sklearn.preprocessing import StandardScaler  # local import (types only)

            y_val_scaled = y_val.reshape(-1, 1)
            y_pred_scaled = np.vstack(val_preds_scaled)
            y_val_log = self.scaler_y.inverse_transform(y_val_scaled).ravel()  # type: ignore[arg-type]
            y_pred_log = self.scaler_y.inverse_transform(y_pred_scaled).ravel()  # type: ignore[arg-type]
            y_val_orig = self.inv_signed_log1p(y_val_log)
            y_pred_orig = self.inv_signed_log1p(y_pred_log)

            mae_val_orig = float(np.mean(np.abs(y_val_orig - y_pred_orig)))

            self._epochs_run += 1
            improved = math.isfinite(mae_val_orig) and (mae_val_orig < self._best_val_mae_orig - 1e-9)
            if improved:
                self._best_val_mae_orig = mae_val_orig
                self._best_epoch = epoch
                if self.use_ema and (self._ema_state is not None):
                    best_state = {k: v.clone() for k, v in self._ema_state.items()}
                else:
                    best_state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
                patience_left = self.patience
                self._maybe_push_snapshot(mae_val_orig, best_state)
            else:
                patience_left -= 1
                if patience_left == 0:
                    print("Early stopping")
                    break

            if epoch == 1 or epoch % 5 == 0:
                print(
                    f"epoch {epoch}/{self.epochs}  "
                    f"train_loss={np.mean(train_losses):.4f}  "
                    f"val_loss={val_loss:.4f}  "
                    f"val_MAE_orig={mae_val_orig:.3f}"
                )

        final_state = best_state
        if len(self._snapshots) >= 2:
            avg_state = self._avg_states([st for _, st in self._snapshots])
            final_state = avg_state

        self._load_state(final_state)
        self.eval()
        self.fitted = True
        print(
            f"[TRAIN] done. epochs_run={self._epochs_run} | "
            f"best_val_MAE_orig={self._best_val_mae_orig:.3f} @epoch={self._best_epoch}"
        )
        return self

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        return self._forward_core(xb)

    def predict(self, X_df: pd.DataFrame) -> pd.Series:
        if not self.fitted:
            raise RuntimeError("Call fit() first")

        X_scaled = self.scaler_X.transform(X_df.values).astype(np.float32)  # type: ignore[arg-type]
        N = X_scaled.shape[0]
        L = self.lookback
        if N < L:
            return pd.Series([np.nan] * N, index=X_df.index)

        X_win = np.lib.stride_tricks.sliding_window_view(X_scaled, window_shape=(L, X_scaled.shape[1]))
        X_win = X_win.reshape(-1, L, X_scaled.shape[1])

        hours_all, dows_all = self._build_time_indices(X_df.index)
        h_win = np.lib.stride_tricks.sliding_window_view(hours_all, window_shape=L).astype(np.int64)
        d_win = np.lib.stride_tricks.sliding_window_view(dows_all, window_shape=L).astype(np.int64)

        xb = torch.from_numpy(X_win).to(self.device)
        hb = torch.from_numpy(h_win).long().to(self.device)
        db = torch.from_numpy(d_win).long().to(self.device)

        preds_scaled: List[np.ndarray] = []
        with torch.no_grad():
            saved_training = self.training
            if self.mc_dropout_passes > 1:
                self.train()
            else:
                self.eval()

            for i in range(0, xb.size(0), self.batch_size):
                chunk_x = xb[i:i + self.batch_size]
                chunk_h = hb[i:i + self.batch_size]
                chunk_d = db[i:i + self.batch_size]

                if self.mc_dropout_passes <= 1:
                    with torch.cuda.amp.autocast(enabled=self.amp and self.device.type == "cuda"):
                        out = self._forward_core(chunk_x, chunk_h, chunk_d).squeeze(-1).detach().cpu().numpy()
                    preds_scaled.append(out)
                else:
                    outs = []
                    for _ in range(self.mc_dropout_passes):
                        with torch.cuda.amp.autocast(enabled=self.amp and self.device.type == "cuda"):
                            o = self._forward_core(chunk_x, chunk_h, chunk_d).squeeze(-1).detach().cpu().numpy()
                        outs.append(o)
                    preds_scaled.append(np.mean(np.stack(outs, axis=0), axis=0))

            if saved_training:
                self.train()
            else:
                self.eval()

        y_scaled = np.concatenate(preds_scaled, axis=0)
        y_log = self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()  # type: ignore[arg-type]
        y_hat = self.inv_signed_log1p(y_log).astype(np.float32)

        full = np.full(N, np.nan, dtype=np.float32)
        full[L - 1:] = y_hat
        return pd.Series(full, index=X_df.index)


class ModelConfig:
    """Utility to sanitize configuration dictionaries for the model __init__."""

    @staticmethod
    def filter_for_model(cfg: dict) -> dict:
        sig = inspect.signature(SeqTransformerRegressor.__init__)
        allowed = set(sig.parameters.keys()) - {"self"}
        return {k: v for k, v in cfg.items() if k in allowed}


class TransformerAdapter:
    """Adapter to unify interface with GBR/XGB/SARIMA(X) style.

    It fits a fresh SeqTransformerRegressor for each block and keeps a short
    lookback context to allow the first test prediction to have a full window.
    """

    def __init__(self, cfg: dict, train_extras: Optional[dict] = None, name: str = "Transformer") -> None:
        self.cfg = ModelConfig.filter_for_model(cfg)
        self.name = name
        self.model: Optional[SeqTransformerRegressor] = None
        self._tail_ctx: Optional[pd.DataFrame] = None
        self.train_extras = {
            "use_ema": False,
            "snapshot_top_k": 1,
            "mc_dropout_passes": 1,
        }
        if train_extras:
            self.train_extras.update(train_extras)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model = SeqTransformerRegressor(**self.cfg)
        self.model.use_ema = self.train_extras["use_ema"]
        self.model.snapshot_top_k = self.train_extras["snapshot_top_k"]
        self.model.mc_dropout_passes = self.train_extras["mc_dropout_passes"]
        self.model.fit(X_train, y_train)
        L = getattr(self.model, "lookback", self.cfg.get("lookback", 336))
        self._tail_ctx = X_train.tail(max(0, L - 1))
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        ctx = pd.concat([self._tail_ctx, X_test], axis=0) if self._tail_ctx is not None else X_test
        y_pred_ser = self.model.predict(ctx).loc[X_test.index]
        return y_pred_ser.values.astype(float)


class KFoldEvaluator:
    """Time-series K-fold evaluator for a given horizon."""

    def run_for_horizon(
            self,
            df: pd.DataFrame,
            target_col: str,
            cfg: dict,
            horizon_hours: int,
            n_folds: int,
            label: str,
            results_dir: Path,
    ) -> Dict[str, float]:
        """
        Train and evaluate SeqTransformerRegressor on rolling-origin K-Fold splits.

        Saves a CSV per fold:
          results_dir / f"kfold_h{horizon_hours}_fold{fold_id}.csv"
        with columns: timestamp, Observed, <label>.
        """
        tscv = TimeSeriesSplit(n_splits=n_folds, test_size=horizon_hours)

        fold_mae: List[float] = []
        fold_rmse: List[float] = []
        fold_mape: List[float] = []

        cfg_filtered = ModelConfig.filter_for_model(cfg)

        for fold_id, (tr_idx, te_idx) in enumerate(tscv.split(df), start=1):
            train_df = df.iloc[tr_idx]
            test_df = df.iloc[te_idx]

            X_tr, y_tr = train_df.drop(columns=[target_col]), train_df[target_col]
            X_te, y_te = test_df.drop(columns=[target_col]), test_df[target_col]

            # fresh model per fold (no EMA/SWA/MC at inference for fair comparison)
            model = SeqTransformerRegressor(**cfg_filtered)
            model.use_ema = False
            model.mc_dropout_passes = 1
            model.snapshot_top_k = 1

            print(f"[KFOLD H{horizon_hours}] fold {fold_id}/{n_folds} | train={len(X_tr)} | test={len(X_te)}")
            model.fit(X_tr, y_tr)

            # context to ensure the first test point has full lookback window
            L = model.lookback
            X_ctx = pd.concat([X_tr.tail(max(0, L - 1)), X_te], axis=0)
            y_pred = model.predict(X_ctx).loc[y_te.index]

            # metrics
            mae = Metrics.mae(y_te, y_pred)
            rmse = Metrics.rmse(y_te, y_pred)
            mape = Metrics.mape(y_te, y_pred)

            fold_mae.append(mae)
            fold_rmse.append(rmse)
            fold_mape.append(mape)

            # per-fold CSV (align with walk-forward: include timestamp as index label)
            out = pd.DataFrame({"Observed": y_te, label: y_pred}, index=y_te.index)
            out_path = results_dir / f"kfold_h{horizon_hours}_fold{fold_id}.csv"
            out.to_csv(out_path, index_label="timestamp", float_format="%.6f")
            print(
                f"[KFOLD H{horizon_hours}] fold {fold_id}: "
                f"MAE={mae:.3f} RMSE={rmse:.3f} MAPE={mape:.2f}% | saved {out_path.as_posix()}"
            )

            # easy-to-copy block for logs
            print(f"===KFOLD_H{horizon_hours}_FOLD{fold_id}===")
            print("timestamp,Observed," + label)
            for ts, yt, yp in zip(out.index, out["Observed"].values, out[label].values):
                val = f"{yp:.6f}" if pd.notna(yp) else "nan"
                print(f"{pd.Timestamp(ts).isoformat()},{yt:.6f},{val}")
            print(f"===END_KFOLD_H{horizon_hours}_FOLD{fold_id}===\n")

            # GPU cleanup
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        # fold averages
        mean_mae = float(np.nanmean(fold_mae)) if fold_mae else float("nan")
        mean_rmse = float(np.nanmean(fold_rmse)) if fold_rmse else float("nan")
        mean_mape = float(np.nanmean(fold_mape)) if fold_mape else float("nan")

        print(
            f"[KFOLD H{horizon_hours}] AVG over {n_folds} folds -> "
            f"MAE={mean_mae:.3f} RMSE={mean_rmse:.3f} MAPE={mean_mape:.2f}%"
        )

        # summary JSON (same file shape as wcześniej)
        summary_path = results_dir / f"kfold_summary_h{horizon_hours}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                dict(
                    horizon_hours=horizon_hours,
                    n_folds=n_folds,
                    MAE=mean_mae,
                    RMSE=mean_rmse,
                    MAPE=mean_mape,
                    folds=dict(MAE=fold_mae, RMSE=fold_rmse, MAPE=fold_mape),
                ),
                f,
                indent=2,
            )
        print(f"[SAVE] {summary_path.as_posix()}")

        return dict(MAE=mean_mae, RMSE=mean_rmse, MAPE=mean_mape)



class RecursiveBlocksRunner:
    """Run recursive block forecasts and save a merged CSV plus a ranking.

    Behavior mirrors the original function:
    - Use the last (n_blocks * block_hours) hours as a shared evaluation window.
    - For each block i=1..n_blocks: fit on the prefix and forecast exactly the block span.
    - Concatenate all block predictions, score, save CSV and an overview plot.
    """

    def run(
        self,
        models: Dict[str, object],
        features_dataframe: pd.DataFrame,
        target_column: str,
        block_hours: int,
        n_blocks: int,
        save_csv_path: Path,
        results_dir: Path,
    ) -> pd.DataFrame:
        if len(features_dataframe) <= block_hours * n_blocks:
            raise ValueError(
                f"Not enough data for {n_blocks} blocks of {block_hours}h "
                f"(need > {block_hours * n_blocks} rows)."
            )

        total_hours = block_hours * n_blocks
        eval_index = features_dataframe.index[-total_hours:]
        start_idx = len(features_dataframe) - total_hours
        block_bounds = [(start_idx + i * block_hours, start_idx + (i + 1) * block_hours) for i in range(n_blocks)]
        y_observed_full = features_dataframe.loc[eval_index, target_column].astype(float)

        predictions: Dict[str, pd.Series] = {}
        ranking_rows: List[Dict] = []

        for name, model_object in models.items():
            input_df = features_dataframe
            if input_df is None or input_df.empty or target_column not in input_df.columns:
                print(f"[SKIP] {name} - input table missing/empty or lacks '{target_column}'")
                continue

            preds_full: List[pd.Series] = []
            for bi, (b_start, b_end) in enumerate(block_bounds, start=1):
                train = input_df.iloc[:b_start]
                test = input_df.iloc[b_start:b_end]
                if len(test) != block_hours or len(train) == 0:
                    print(f"[{name}] Skip block {bi}: train={len(train)} test={len(test)}")
                    continue

                X_train = train.drop(columns=[target_column])
                y_train = train[target_column].astype(float)
                X_test = test.drop(columns=[target_column])

                try:
                    model_object.fit(X_train, y_train)
                    y_pred_block = pd.Series(model_object.predict(X_test), index=X_test.index, name=name).astype(float)
                    preds_full.append(y_pred_block)
                except Exception as e:  # noqa: BLE001
                    print(f"[ERROR] {name} failed on block {bi}: {e!s}")
                    preds_full.append(pd.Series([np.nan] * len(X_test), index=X_test.index, name=name))

            if not preds_full:
                print(f"[WARN] {name}: no blocks produced.")
                continue

            pred_series = pd.concat(preds_full).sort_index().reindex(eval_index)
            predictions[name] = pred_series

            yt, yp = y_observed_full.align(pred_series, join="inner")
            mask = yt.notna() & yp.notna()
            if mask.any():
                mae = Metrics.mae(yt[mask], yp[mask])
                rmse = Metrics.rmse(yt[mask], yp[mask])
                mape = Metrics.mape(yt[mask], yp[mask])
                print(f"[RECURSIVE] {name}: MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%  over {mask.sum()} points")
                ranking_rows.append(dict(model=name, MAE=mae, RMSE=rmse, MAPE=mape))
            else:
                print(f"[RECURSIVE] {name}: no valid points to score.")

        out_df = pd.DataFrame({"Observed": y_observed_full})
        for name, pred in predictions.items():
            out_df[name] = pred.reindex(out_df.index)

        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(save_csv_path, index_label="timestamp", float_format="%.6f")
        print(f"[SAVE] Recursive blocks -> {save_csv_path}")

        print("===RECURSIVE_BLOCKS_CSV===")
        print("timestamp," + ",".join(out_df.columns))
        for ts, row in out_df.iterrows():
            vals = [f"{row[col]:.6f}" if pd.notna(row[col]) else "nan" for col in out_df.columns]
            print(f"{pd.Timestamp(ts).isoformat()},{','.join(vals)}")
        print("===END_RECURSIVE_BLOCKS_CSV===")

        if ranking_rows:
            rank_df = pd.DataFrame(ranking_rows).sort_values("MAE").reset_index(drop=True)
            rank_path = results_dir / "ranking_recursive_blocks.csv"
            rank_df.to_csv(rank_path, index=False)
            print("[RANK] Recursive blocks ranking")
            print(
                rank_df.to_string(
                    index=False,
                    formatters={"MAE": "{:.2f}".format, "RMSE": "{:.2f}".format, "MAPE": "{:.2f}".format},
                )
            )
            print(f"[SAVE] {rank_path}")

        try:
            plt.figure(figsize=(12, 4))
            plt.plot(out_df.index, out_df["Observed"].values, label="Observed", linewidth=2)
            for name in predictions.keys():
                plt.plot(out_df.index, out_df[name].values, label=name, alpha=0.9)
            plt.title(f"Recursive block forecasts – {n_blocks}×{block_hours}h")
            plt.legend()
            plt.tight_layout()
            plt.savefig(results_dir / f"plot_recursive_blocks_h{block_hours}_x{n_blocks}.png", dpi=120)
            plt.close()
        except Exception:  # noqa: BLE001
            pass

        return out_df


class FeatureImportance:
    """Permutation importance on the last fold (time-aware)."""

    @staticmethod
    def permutation_importance_csv(
            model: SeqTransformerRegressor,
            features_dataframe: pd.DataFrame,
            target_column: str,
            top_n: int,
            save_csv_path: Path,
            save_plot_path: Optional[Path] = None,
            window_hours: int = 24 * 7,
            n_repeats: int = 3,
            random_seed: int = 42,
    ) -> pd.DataFrame:
        """
        Time-aware permutation importance on the last `window_hours` samples.
        Saves CSV like Walk-forward (columns: feature, importance=ΔMAE) and optional plot.
        """
        rng = np.random.default_rng(random_seed)

        if len(features_dataframe) <= window_hours:
            raise ValueError("Not enough rows for the chosen window_hours.")

        # last window -> validation; the rest -> training context
        train_df = features_dataframe.iloc[:-window_hours]
        val_df = features_dataframe.iloc[-window_hours:]

        X_tr = train_df.drop(columns=[target_column])
        y_tr = train_df[target_column].astype(float)

        X_val = val_df.drop(columns=[target_column])
        y_val = val_df[target_column].astype(float)

        # Build context for Transformer (tail lookback-1 + val)
        lookback = getattr(model, "lookback", 1)
        X_ctx_base = pd.concat([X_tr.tail(max(0, lookback - 1)), X_val], axis=0)

        # Baseline prediction on original X_ctx
        y_pred_base = model.predict(X_ctx_base).loc[X_val.index]
        base_mae = Metrics.mae(y_val, y_pred_base)

        importances: List[Tuple[str, float]] = []
        cols = list(X_val.columns)
        ctx_index = X_ctx_base.index
        test_mask = ctx_index.isin(X_val.index)

        for col in cols:
            deltas: List[float] = []
            for _ in range(n_repeats):
                X_ctx_perm = X_ctx_base.copy()
                # shuffle only the validation part of this column
                perm_vals = X_ctx_perm.loc[test_mask, col].to_numpy()
                rng.shuffle(perm_vals)
                X_ctx_perm.loc[test_mask, col] = perm_vals

                y_pred_perm = model.predict(X_ctx_perm).loc[X_val.index]
                mae_perm = Metrics.mae(y_val, y_pred_perm)
                deltas.append(mae_perm - base_mae)

            importances.append((col, float(np.mean(deltas))))

        imp_df = (
            pd.DataFrame(importances, columns=["feature", "importance"])
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        # Save CSV
        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        imp_df.to_csv(save_csv_path, index=False)

        # Optional plot
        if save_plot_path is not None:
            try:
                top = imp_df.head(top_n)
                plt.figure(figsize=(8, max(4, int(0.35 * len(top)))))
                plt.barh(top["feature"][::-1], top["importance"][::-1])
                plt.xlabel("ΔMAE (higher = more important)")
                plt.title("Permutation importance (last window)")
                plt.tight_layout()
                plt.savefig(save_plot_path, dpi=120)
                plt.close()
            except Exception:
                pass

        # Console summary (like Walk-forward)
        print(f"[Feature importance | permutation] TOP {top_n}:")
        for i, row in imp_df.head(top_n).iterrows():
            print(f" #{i + 1:02d} {row['feature']}: ΔMAE=+{row['importance']:.4f}")

        return imp_df



class OptunaTunerCV:
    """Optuna tuner for CV evaluation (minimize MAE over small CV)."""

    @staticmethod
    def tune(
        df: pd.DataFrame,
        target_col: str,
        trials: int,
        horizon: int,
        splits: int,
        results_dir: Path,
        seed: int = 42,
    ) -> dict:
        try:
            import optuna  # type: ignore
            from optuna.pruners import SuccessiveHalvingPruner  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Optuna is not installed. Please ensure it's available in the environment."
            ) from e

        print(f"[Optuna] Tuning start: n_trials={trials}, test_horizon={horizon}, n_splits={splits}, seed={seed}")
        trainer = Trainer()

        def objective(trial: "optuna.trial.Trial") -> float:  # type: ignore[name-defined]
            cfg = dict(
                embed_dim=trial.suggest_categorical("embed_dim", [128, 192, 256]),
                nhead=trial.suggest_categorical("nhead", [4, 8]),
                num_layers=trial.suggest_int("num_layers", 3, 8),
                ff_dim=trial.suggest_categorical("ff_dim", [256, 384, 512, 768]),
                dropout=trial.suggest_float("dropout", 0.05, 0.25),
                batch_size=trial.suggest_categorical("batch_size", [24, 32, 48]),
                epochs=130,
                lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                weight_decay=trial.suggest_float("weight_decay", 1e-6, 3e-4, log=True),
                lookback=336,
                patience=15,
                aggregation="last",
                amp=True,
            )

            Trainer.set_global_seed(seed)
            model = SeqTransformerRegressor(**cfg)
            mae = trainer.validate_model(
                df_input=df,
                target_col=target_col,
                model=model,
                test_horizon=horizon,
                n_splits=splits,
                model_name=f"TRIAL_{trial.number}",
            )
            trial.set_user_attr("cfg", cfg)
            return mae

        study = optuna.create_study(direction="minimize", pruner=SuccessiveHalvingPruner(min_resource=20, reduction_factor=3))
        study.optimize(objective, n_trials=trials, show_progress_bar=False)

        best_cfg = study.best_trial.user_attrs["cfg"]
        print("Best trial MAE:", study.best_value)
        print("Best config:", json.dumps(best_cfg, indent=2))
        # Optionally save study summary
        try:
            with open(results_dir / "optuna_best_cfg.json", "w", encoding="utf-8") as f:
                json.dump(dict(best_value=study.best_value, best_cfg=best_cfg), f, indent=2)
        except Exception:  # noqa: BLE001
            pass
        return best_cfg


# =========================
# Orchestration
# =========================


class TransformerPipeline:
    """Thin helper that only provides model presets and (optionally) Optuna CV choice.

    Main training/evaluation is now orchestrated directly in main().
    You can override presets from outside by setting the EXTERNAL_* class vars.
    """

    # Built-in fallback presets (kept for backward-compat)
    _DEFAULT_PRESET_BEST_CFG = dict(
        embed_dim=192, nhead=8, num_layers=5, ff_dim=768,
        dropout=0.09643232167672024, batch_size=32, epochs=115,
        lr=0.00014255116408297153, weight_decay=2.3140733126148502e-05,
        lookback=336, patience=20, aggregation="last", amp=True,
    )
    _DEFAULT_PRESET_SECOND_CFG_7D = dict(
        embed_dim=256, nhead=4, num_layers=7, ff_dim=256,
        dropout=0.14692944130959192, batch_size=48, epochs=100,
        lr=6.432262674691829e-05, weight_decay=1.7913659623853466e-04,
        lookback=336, patience=30, aggregation="last", amp=True,
    )

    # Optional external overrides (can be set from main)
    EXTERNAL_PRESET_BEST_CFG: Optional[dict] = None
    EXTERNAL_PRESET_SECOND_CFG_7D: Optional[dict] = None

    def __init__(self, run_cfg: RunConfig, paths: ProjectPaths) -> None:
        self.run_cfg = run_cfg
        self.paths = paths
        self.data = DataModule(target_column=run_cfg.target_column)

    def get_presets(self) -> Tuple[dict, dict]:
        """Return (best_cfg, seven_day_cfg), preferring external overrides."""
        best = (self.EXTERNAL_PRESET_BEST_CFG or self._DEFAULT_PRESET_BEST_CFG).copy()
        seven = (self.EXTERNAL_PRESET_SECOND_CFG_7D or self._DEFAULT_PRESET_SECOND_CFG_7D).copy()
        return best, seven

    def choose_config(self, df: pd.DataFrame) -> dict:
        """Choose base configuration: Optuna CV or preset (best)."""
        best, _ = self.get_presets()
        best["lookback"] = self.run_cfg.lookback
        if self.run_cfg.tune_with_optuna:
            try:
                cfg = OptunaTunerCV.tune(
                    df=df,
                    target_col=self.run_cfg.target_column,
                    trials=self.run_cfg.optuna_trials,
                    horizon=24 * 7,
                    splits=2,
                    results_dir=self.paths.results_dir,
                    seed=self.run_cfg.seed,
                )
                cfg["lookback"] = self.run_cfg.lookback
                cfg_source = "optuna_cv"
            except RuntimeError as e:
                print(f"[WARN] Optuna tuning skipped ({e}). Falling back to preset.")
                cfg = best
                cfg_source = "preset"
        else:
            cfg = best
            cfg_source = "preset"

        print(f"[CFG] Using configuration ({cfg_source}):\n{json.dumps(cfg, indent=2)}")
        return cfg



def main() -> None:
    """Lightweight orchestration done inline for clarity (junior-style)."""
    # 0) Paths and seed
    paths = ProjectPaths.bootstrap()
    Trainer.set_global_seed(42)

    # 1) Hyperparameters (presets) in one place
    PRESET_BEST_CFG = dict(
        embed_dim=192, nhead=8, num_layers=5, ff_dim=768,
        dropout=0.09643232167672024, batch_size=32, epochs=115,
        lr=0.00014255116408297153, weight_decay=2.3140733126148502e-05,
        lookback=336, patience=20, aggregation="last", amp=True,
    )
    PRESET_SECOND_CFG_7D = dict(
        embed_dim=256, nhead=4, num_layers=7, ff_dim=256,
        dropout=0.14692944130959192, batch_size=48, epochs=100,
        lr=6.432262674691829e-05, weight_decay=1.7913659623853466e-04,
        lookback=336, patience=30, aggregation="last", amp=True,
    )
    # Make presets visible to TransformerPipeline (if someone uses it elsewhere)
    TransformerPipeline.EXTERNAL_PRESET_BEST_CFG = PRESET_BEST_CFG
    TransformerPipeline.EXTERNAL_PRESET_SECOND_CFG_7D = PRESET_SECOND_CFG_7D

    # 2) Minimal run config (keeps current horizons 24, 7*24, 14*24, 31*24)
    default_data = paths.project_root / "BAZA_CLIPOFF.parquet"
    run_cfg = RunConfig(
        data_path=str(default_data),
        target_column="fixing_i_price",
        seed=42,
        tune_with_optuna=False,            # set True to run Optuna CV
        optuna_trials=45,
        run_horizons_kfold=True,
        n_folds_for_horizons=3,
        horizons_hours=[24, 24 * 7, 24 * 14, 24 * 31],
        recursive_block_hours=24 * 7,
        recursive_n_blocks=10,
        lookback=336,
    )
    print("[CFG]", run_cfg.to_json())

    # 3) Load data
    data = DataModule(target_column=run_cfg.target_column)
    df = data.load_dataframe(run_cfg.data_path)
    print("[DATA] Columns:", list(df.columns))
    print("[DATA] Shape:", df.shape)
    print("[DATA] NA ratio (top 5):", df.isna().mean().sort_values().tail(5))

    # 4) Choose base config (Optuna CV or preset)
    if run_cfg.tune_with_optuna:
        try:
            base_cfg = OptunaTunerCV.tune(
                df=df,
                target_col=run_cfg.target_column,
                trials=run_cfg.optuna_trials,
                horizon=24 * 7,
                splits=2,
                results_dir=paths.results_dir,
                seed=run_cfg.seed,
            )
            base_cfg["lookback"] = run_cfg.lookback
            cfg_source = "optuna_cv"
        except RuntimeError as e:
            print(f"[WARN] Optuna tuning skipped ({e}). Falling back to preset.")
            base_cfg = PRESET_BEST_CFG.copy()
            base_cfg["lookback"] = run_cfg.lookback
            cfg_source = "preset"
    else:
        base_cfg = PRESET_BEST_CFG.copy()
        base_cfg["lookback"] = run_cfg.lookback
        cfg_source = "preset"
    print(f"[CFG] Using configuration ({cfg_source}):\n{json.dumps(base_cfg, indent=2)}")

    # 5) Time-series K-Fold across horizons
    if run_cfg.run_horizons_kfold and run_cfg.horizons_hours:
        evaluator = KFoldEvaluator()
        for h in run_cfg.horizons_hours:
            if h == 24 * 7:
                cfg_run = PRESET_SECOND_CFG_7D.copy()
                cfg_run["lookback"] = run_cfg.lookback
                label = "Transformer_7d"
            else:
                cfg_run = base_cfg.copy()
                label = f"Transformer_{h}h"

            _ = evaluator.run_for_horizon(
                df=df,
                target_col=run_cfg.target_column,
                cfg=cfg_run,
                horizon_hours=h,
                n_folds=run_cfg.n_folds_for_horizons,
                label=label,
                results_dir=paths.results_dir,
            )
    else:
        print("[SKIP] run_horizons_kfold=False or no horizons provided")

    # 6) Recursive blocks (Transformer only)
    runner = RecursiveBlocksRunner()
    models: Dict[str, object] = {
        "Transformer": TransformerAdapter(
            cfg=PRESET_SECOND_CFG_7D,
            train_extras=dict(use_ema=False, snapshot_top_k=1, mc_dropout_passes=1),
            name="Transformer",
        )
    }
    save_csv = paths.results_dir / f"recursive_blocks_h{run_cfg.recursive_block_hours}_x{run_cfg.recursive_n_blocks}.csv"
    _ = runner.run(
        models=models,
        features_dataframe=df,
        target_column=run_cfg.target_column,
        block_hours=run_cfg.recursive_block_hours,
        n_blocks=run_cfg.recursive_n_blocks,
        save_csv_path=save_csv,
        results_dir=paths.results_dir,
    )

if __name__ == "__main__":
    main()
