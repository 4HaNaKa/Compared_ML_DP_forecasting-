# -*- coding: utf-8 -*-
"""
Transformer Forecasting Pipeline

This module provides a production-style pipeline for training a Transformer-based
regressor on hourly time series, computing permutation feature importance on the
latest horizon, and producing a forward forecast.

Key features:
1) Data loading & preparation from Parquet
2) Optional Optuna hyperparameter tuning with TimeSeriesSplit CV.
3) Transformer model with:
   - signed_log1p target transform + StandardScaler
   - sliding "lookback" windows
   - causal attention mask
   - hour-of-day & day-of-week embeddings
   - early stopping on MAE in the original scale
4) Permutation feature importance on the last H hours (keeps proper lookback context).
5) Refit on the full dataset and forecast the next H hours to CSV.

Outputs (under results directory):
- best_optuna_config.json (if tuning enabled)
- training_config_used.json (config actually used for training/forecast)
- perm_importance_lastH_top20.csv (if feasible)
- forecast_h{H}.csv (timestamp, prediction)
"""

from __future__ import annotations

import os
import math
import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

TARGET_COLUMN = "fixing_i_price"
DEFAULT_DATA = "raw_database.parquet"

# Defaults for Optuna
TUNE_WITH_OPTUNA_DEFAULT = True
OPTUNA_TRIALS_DEFAULT = 40

# Model training enhancements
SNAPSHOT_TOP_K_DEFAULT = 3
EMA_DECAY_DEFAULT = 0.999
MC_DROPOUT_PASSES_DEFAULT = 1
FEATURE_NOISE_STD_DEFAULT = 0.01

# Presets
BEST_CFG = dict(
    embed_dim=256,
    nhead=8,
    num_layers=6,
    ff_dim=512,
    dropout=0.15,
    batch_size=32,
    epochs=130,
    lr=5e-4,
    weight_decay=1e-5,
    lookback=336,            # 14 days of hourly history
    patience=30,
    aggregation="last",
    amp=True
)

TUNED_CFG = dict(
    embed_dim=192, nhead=8, num_layers=5, ff_dim=768,
    dropout=0.09643232167672024, batch_size=32, epochs=130,
    lr=0.00014255116408297153, weight_decay=2.3140733126148502e-05,
    lookback=336, patience=20, aggregation="last", amp=True
)


@dataclass
class RunConfig:
    """High-level pipeline configuration."""
    data_path: str = DEFAULT_DATA
    results_dir: Path = Path("results_transformer")
    forecast_h: int = 7 * 24  # 7 days of hourly forecast
    tune_with_optuna: bool = TUNE_WITH_OPTUNA_DEFAULT
    optuna_trials: int = OPTUNA_TRIALS_DEFAULT
    cv_splits: int = 3
    seed: int = 42

    def ensure_dirs(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)


# Utilities

def _optuna_available() -> bool:
    try:
        import optuna  # noqa: F401
        return True
    except Exception:
        return False


def set_global_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility (keeps CuDNN autotune)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True


def signed_log1p(x: np.ndarray | pd.Series) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.sign(x) * np.log1p(np.abs(x))


def inv_signed_log1p(x: np.ndarray | pd.Series) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.sign(x) * np.expm1(np.abs(x))


class Metrics:
    """NaN-safe metrics for convenience."""
    @staticmethod
    def mae(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> float:
        y_pred = pd.Series(y_pred, index=y_true.index)
        mask = ~np.isnan(y_pred.values)
        if mask.sum() == 0:
            return np.nan
        return float(mean_absolute_error(y_true[mask], y_pred[mask]))

    @staticmethod
    def rmse(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> float:
        y_pred = pd.Series(y_pred, index=y_true.index)
        mask = ~np.isnan(y_pred.values)
        if mask.sum() == 0:
            return np.nan
        return float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))

    @staticmethod
    def mape(y_true: pd.Series, y_pred: pd.Series | np.ndarray, eps: float = 1e-8) -> float:
        y_pred = pd.Series(y_pred, index=y_true.index)
        mask = (~np.isnan(y_pred.values)) & (np.abs(y_true.values) >= eps)
        if mask.sum() == 0:
            return np.nan
        return float(np.mean(np.abs((y_true.values[mask] - y_pred.values[mask]) / y_true.values[mask])) * 100.0)

    @staticmethod
    def r2(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> float:
        y_pred = pd.Series(y_pred, index=y_true.index)
        mask = ~np.isnan(y_pred.values)
        if mask.sum() <= 1:
            return np.nan
        return float(r2_score(y_true.values[mask], y_pred.values[mask]))


# Data Module

class DataModule:
    """Data loading, calendar features, and future feature generation."""

    def __init__(self, target_col: str = TARGET_COLUMN) -> None:
        self.target_col = target_col

    def load_dataframe(self, data_uri: str) -> pd.DataFrame:
        """
        Load Parquet (local). Require DatetimeIndex (or 'datetime'/'timestamp' column).
        Enforce hourly frequency, forward-fill, and drop zero-variance columns.
        """
        df = pd.read_parquet(data_uri)

        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
            else:
                raise ValueError("Parquet must contain DatetimeIndex or 'datetime'/'timestamp' column")

        df = df.asfreq("H")
        df = df.dropna(subset=[self.target_col]).fillna(method="ffill")

        constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
        if constant_cols:
            print("Removing zero-variance columns:", constant_cols)
            df = df.drop(columns=constant_cols)
        return df

    @staticmethod
    def _calendar_from_index(idx: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
        hour = idx.hour.values
        dow = idx.dayofweek.values
        month = idx.month.values
        weekend = (dow >= 5).astype(int)
        rad = 2 * np.pi
        sin24 = np.sin(rad * (hour / 24.0));  cos24 = np.cos(rad * (hour / 24.0))
        sin168 = np.sin(rad * ((dow * 24 + hour) / 168.0));  cos168 = np.cos(rad * ((dow * 24 + hour) / 168.0))
        quarter = ((month - 1) // 3 + 1).astype(int)
        return dict(hour=hour, day_of_week=dow, month=month, weekend=weekend,
                    sin24=sin24, cos24=cos24, sin168=sin168, cos168=cos168, quarter=quarter)

    def ensure_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cal = self._calendar_from_index(df.index)
        for k, v in cal.items():
            if k not in df.columns:
                df[k] = v
        return df

    def make_future_features_like_train(self, train_df: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
        """
        Build a future feature frame with the same columns as training X (train_df without target).
        - Calendar features are recomputed from the future index.
        - All other features are forward-filled with their last observed values.
        """
        last_ts = train_df.index.max()
        idx_future = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=horizon_h, freq="H")
        X_hist = train_df.drop(columns=[self.target_col])
        X_future = pd.DataFrame(index=idx_future, columns=X_hist.columns, dtype=float)

        cal = self._calendar_from_index(idx_future)
        for col in X_future.columns:
            if col in cal:
                X_future[col] = cal[col]
            else:
                X_future[col] = float(X_hist[col].iloc[-1])
        return X_future


# Model

def _make_causal_mask(L: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(L, L, device=device) * float("-inf"), diagonal=1)


class PositionalEncoding(nn.Module):
    """Standard sinus/cos positional encoding."""
    def __init__(self, d_model: int, max_len: int = 20000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class SeqTransformerRegressor(nn.Module):
    """
    Sequence-to-regression model for hourly time series.

    Pipeline:
    - y -> signed_log1p -> StandardScaler
    - X -> StandardScaler
    - sliding lookback windows
    - causal Transformer encoder with positional + temporal embeddings
    - 'last' (default) or 'mean' token aggregation
    """

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
        device: Optional[str] = None
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

        if self.aggregation not in {"last", "mean"}:
            raise ValueError("aggregation must be 'last' or 'mean'")

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # To be initialized in fit()
        self.input_proj: Optional[nn.Linear] = None
        self.pos_enc: Optional[PositionalEncoding] = None
        self.encoder: Optional[nn.TransformerEncoder] = None
        self.final_norm: Optional[nn.LayerNorm] = None
        self.output_proj: Optional[nn.Linear] = None

        self.scaler_X: Optional[StandardScaler] = None
        self.scaler_y: Optional[StandardScaler] = None

        self._epochs_run: int = 0
        self._best_epoch: int = 0
        self._best_val_mae_orig: float = float("inf")
        self.fitted: bool = False

        # Causal mask cache (CPU)
        self._causal_mask_cpu = _make_causal_mask(self.lookback, torch.device("cpu"))

        # Temporal embeddings
        self.hour_emb = nn.Embedding(24, self.embed_dim).to(self.device)
        self.dow_emb = nn.Embedding(7, self.embed_dim).to(self.device)
        self.time_proj = nn.Linear(self.embed_dim * 3, self.embed_dim).to(self.device)

        # EMA + snapshots
        self.use_ema: bool = True
        self.ema_decay: float = EMA_DECAY_DEFAULT
        self._ema_state: Optional[dict[str, torch.Tensor]] = None
        self.snapshot_top_k: int = SNAPSHOT_TOP_K_DEFAULT
        self._snapshots: List[Tuple[float, dict[str, torch.Tensor]]] = []

        # Inference/training helpers
        self.mc_dropout_passes: int = MC_DROPOUT_PASSES_DEFAULT
        self.feature_noise_std: float = FEATURE_NOISE_STD_DEFAULT

    # helpers (windows & temporal indices)
    def _build_windows(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        L = self.lookback
        if len(X) < L:
            raise ValueError(f"Series shorter than lookback={L}")
        X_win = np.lib.stride_tricks.sliding_window_view(X, window_shape=(L, X.shape[1]))
        X_win = X_win.reshape(-1, L, X.shape[1])
        y_trim = y[L - 1:]
        return X_win, y_trim

    @staticmethod
    def _time_indices(index: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
        hours = index.hour.values.astype(np.int64)
        dows = index.dayofweek.values.astype(np.int64)
        return hours, dows

    def _init_network(self, n_features: int) -> None:
        self.input_proj = nn.Linear(n_features, self.embed_dim).to(self.device)
        self.pos_enc = PositionalEncoding(self.embed_dim, max_len=max(20000, self.lookback + 1)).to(self.device)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=self.nhead, dim_feedforward=self.ff_dim,
            dropout=self.dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers).to(self.device)
        self.final_norm = nn.LayerNorm(self.embed_dim).to(self.device)
        self.output_proj = nn.Linear(self.embed_dim, 1).to(self.device)

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

    @staticmethod
    def _avg_states(states: List[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for k in states[0].keys():
            s = None
            for st in states:
                t = st[k]
                s = t.clone() if s is None else s.add_(t)
            out[k] = s.div_(len(states))
        return out

    def _maybe_push_snapshot(self, mae_val: float, state: dict[str, torch.Tensor]) -> None:
        self._snapshots.append((mae_val, {k: v.detach().clone().cpu() for k, v in state.items()}))
        self._snapshots.sort(key=lambda x: x[0])
        if len(self._snapshots) > self.snapshot_top_k:
            self._snapshots = self._snapshots[:self.snapshot_top_k]

    # forward core
    def _forward_core(self, xb: torch.Tensor, hb: Optional[torch.Tensor] = None, db: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.input_proj(xb)
        if (hb is not None) and (db is not None):
            hb = hb.to(self.device, non_blocking=True).long()
            db = db.to(self.device, non_blocking=True).long()
            he = self.hour_emb(hb)
            de = self.dow_emb(db)
            z = torch.cat([z, he, de], dim=-1)
            z = self.time_proj(z)
        z = self.pos_enc(z)
        mask = self._causal_mask_cpu.to(self.device)
        z = self.encoder(z, mask=mask)
        z = z[:, -1, :] if self.aggregation == "last" else z.mean(dim=1)
        z = self.final_norm(z)
        out = self.output_proj(z)
        return out

    # ---- fit/predict ----
    def fit(self, X_df: pd.DataFrame, y_ser: pd.Series) -> "SeqTransformerRegressor":
        if X_df.isna().any().any():
            raise ValueError("NaN in X")
        if y_ser.isna().any():
            raise ValueError("NaN in y")

        index = X_df.index
        y_log = signed_log1p(y_ser.values.astype(np.float32)).reshape(-1, 1)
        self.scaler_y = StandardScaler().fit(y_log)
        y_scaled = self.scaler_y.transform(y_log).astype(np.float32)

        self.scaler_X = StandardScaler().fit(X_df.values)
        X_scaled = self.scaler_X.transform(X_df.values).astype(np.float32)

        X_win, y_win = self._build_windows(X_scaled, y_scaled)

        hours_all, dows_all = self._time_indices(index)
        h_win = np.lib.stride_tricks.sliding_window_view(hours_all, window_shape=self.lookback).astype(np.int64)
        d_win = np.lib.stride_tricks.sliding_window_view(dows_all,  window_shape=self.lookback).astype(np.int64)

        M = X_win.shape[0]
        val_len = max(1, int(0.1 * M))
        X_tr, y_tr = X_win[:-val_len], y_win[:-val_len]
        X_val, y_val = X_win[-val_len:], y_win[-val_len:]
        h_tr, d_tr = h_win[:-val_len], d_win[:-val_len]
        h_val, d_val = h_win[-val_len:], d_win[-val_len:]

        self._init_network(X_scaled.shape[1])

        if self.use_ema:
            self._ema_init_from_current()

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr),
                          torch.from_numpy(h_tr), torch.from_numpy(d_tr)),
            batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=False
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val),
                          torch.from_numpy(h_val), torch.from_numpy(d_val)),
            batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=False
        )

        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.L1Loss()
        steps_per_epoch = max(1, math.ceil(len(X_tr) / self.batch_size))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, epochs=self.epochs, steps_per_epoch=steps_per_epoch,
            pct_start=0.1, div_factor=10.0, final_div_factor=10.0
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self.amp and self.device.type == "cuda")

        best_state = {k: v.cpu() for k, v in self.state_dict().items()}
        patience_left = self.patience
        self._epochs_run = 0
        self._best_epoch = 0
        self._best_val_mae_orig = float("inf")

        print(f"[TRAIN] device={self.device} | lookback={self.lookback} | epochs={self.epochs}")

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
                    scaler.step(optimizer); scaler.update(); scheduler.step()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step(); scheduler.step()

                if self.use_ema:
                    self._ema_update()
                train_losses.append(float(loss.detach().item()))

            # Validation (optionally with EMA weights)
            self.eval()
            saved_state = None
            if self.use_ema and (self._ema_state is not None):
                saved_state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
                self._load_state(self._ema_state)

            val_losses, val_preds_scaled = [], []
            with torch.no_grad():
                for xb, yb, hb, db in val_loader:
                    xb = xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)
                    hb = hb.to(self.device, non_blocking=True).long()
                    db = db.to(self.device, non_blocking=True).long()
                    with torch.cuda.amp.autocast(enabled=self.amp and self.device.type == "cuda"):
                        preds = self._forward_core(xb, hb, db)
                        loss = loss_fn(preds, yb)
                    val_losses.append(float(loss.detach().item()))
                    val_preds_scaled.append(preds.detach().cpu().numpy())

            if saved_state is not None:
                self._load_state(saved_state)

            # Early stopping on MAE in original scale
            y_val_scaled = y_val.reshape(-1, 1)
            y_pred_scaled = np.vstack(val_preds_scaled)
            y_val_log = self.scaler_y.inverse_transform(y_val_scaled).ravel()
            y_pred_log = self.scaler_y.inverse_transform(y_pred_scaled).ravel()
            y_val_orig = inv_signed_log1p(y_val_log)
            y_pred_orig = inv_signed_log1p(y_pred_log)
            mae_val_orig = float(np.mean(np.abs(y_val_orig - y_pred_orig)))

            self._epochs_run = epoch
            if math.isfinite(mae_val_orig) and mae_val_orig < self._best_val_mae_orig - 1e-9:
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
                    print("[EARLY STOP] No improvement.")
                    break

            if epoch == 1 or epoch % 5 == 0:
                print(f"[EPOCH {epoch}/{self.epochs}] "
                      f"train_loss={np.mean(train_losses):.4f}  "
                      f"val_loss={np.mean(val_losses):.4f}  "
                      f"val_MAE_orig={mae_val_orig:.3f}")

        final_state = best_state
        if len(self._snapshots) >= 2:
            final_state = self._avg_states([st for _, st in self._snapshots])

        self._load_state(final_state)
        self.eval()
        self.fitted = True
        print(f"[TRAIN] done. epochs_run={self._epochs_run} | "
              f"best_val_MAE_orig={self._best_val_mae_orig:.3f} @epoch={self._best_epoch}")
        return self

    def predict(self, X_df: pd.DataFrame) -> pd.Series:
        if not self.fitted:
            raise RuntimeError("Call fit() first")

        X_scaled = self.scaler_X.transform(X_df.values).astype(np.float32)
        N, L = X_scaled.shape[0], self.lookback
        if N < L:
            return pd.Series([np.nan] * N, index=X_df.index)

        X_win = np.lib.stride_tricks.sliding_window_view(X_scaled, window_shape=(L, X_scaled.shape[1]))
        X_win = X_win.reshape(-1, L, X_scaled.shape[1])

        hours_all = X_df.index.hour.values.astype(np.int64)
        dows_all = X_df.index.dayofweek.values.astype(np.int64)
        h_win = np.lib.stride_tricks.sliding_window_view(hours_all, window_shape=L).astype(np.int64)
        d_win = np.lib.stride_tricks.sliding_window_view(dows_all,  window_shape=L).astype(np.int64)

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

            mask = self._causal_mask_cpu.to(self.device)
            for i in range(0, xb.size(0), self.batch_size):
                chunk_x = xb[i:i + self.batch_size]
                chunk_h = hb[i:i + self.batch_size]
                chunk_d = db[i:i + self.batch_size]
                with torch.cuda.amp.autocast(enabled=self.amp and self.device.type == "cuda"):
                    z = self.input_proj(chunk_x)
                    z = torch.cat([z, self.hour_emb(chunk_h), self.dow_emb(chunk_d)], dim=-1)
                    z = self.time_proj(z)
                    z = self.pos_enc(z)
                    z = self.encoder(z, mask=mask)
                    z = z[:, -1, :] if self.aggregation == "last" else z.mean(dim=1)
                    z = self.final_norm(z)
                    out = self.output_proj(z).squeeze(-1).detach().cpu().numpy()
                preds_scaled.append(out)

            if saved_training:
                self.train()
            else:
                self.eval()

        y_scaled = np.concatenate(preds_scaled, axis=0)
        y_log = self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
        y_hat = inv_signed_log1p(y_log).astype(np.float32)

        full = np.full(N, np.nan, dtype=np.float32)
        full[L - 1:] = y_hat
        return pd.Series(full, index=X_df.index)


# Feature Importance

class FeatureImportance:
    """Permutation feature importance for time series with lookback context."""

    @staticmethod
    def permutation_last_h(
        model: SeqTransformerRegressor,
        X_tr: pd.DataFrame,
        X_te: pd.DataFrame,
        y_te: pd.Series,
        metric_fn = Metrics.mae,
        n_repeats: int = 3,
        top_k: int = 20
    ) -> pd.DataFrame:
        """
        Compute permutation importance by permuting ONLY the test region.
        The 'context' (tail of training needed for the first test prediction) is kept intact.
        """
        lookback = getattr(model, "lookback", 1)
        X_ctx = pd.concat([X_tr.tail(max(0, lookback - 1)), X_te], axis=0)

        y_pred_base = model.predict(X_ctx).loc[X_te.index]
        base_err = metric_fn(y_te, y_pred_base)

        importances: List[Tuple[str, float]] = []
        feat_cols = list(X_te.columns)

        test_index = X_te.index
        ctx_index = X_ctx.index
        test_mask = ctx_index.isin(test_index)

        for col in feat_cols:
            deltas = []
            for _ in range(n_repeats):
                X_perm = X_ctx.copy()
                shuffled = X_perm.loc[test_mask, col].sample(frac=1.0, replace=False).values
                X_perm.loc[test_mask, col] = shuffled
                y_pred_perm = model.predict(X_perm).loc[X_te.index]
                err = metric_fn(y_te, y_pred_perm)
                deltas.append(err - base_err)
            importances.append((col, float(np.mean(deltas))))

        imp_df = (pd.DataFrame(importances, columns=["feature", "delta_mae"])
                    .sort_values("delta_mae", ascending=False)
                    .reset_index(drop=True))

        print(f"\n[Permutation importance] TOP {top_k}:")
        for i, row in imp_df.head(top_k).iterrows():
            print(f" #{i+1:02d} {row['feature']}: ΔMAE=+{row['delta_mae']:.4f}")
        return imp_df


# Optuna Tuner (CV)

class OptunaTuner:
    """Optuna-based hyperparameter tuner using TimeSeriesSplit CV."""

    @staticmethod
    def _cv_mae_for_cfg(
        df: pd.DataFrame,
        target_col: str,
        cfg: dict,
        test_horizon: int,
        n_splits: int,
        seed: int = 42
    ) -> float:
        set_global_seed(seed)
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_horizon)
        maes: List[float] = []

        for fold, (tr_idx, te_idx) in enumerate(tscv.split(df), start=1):
            train_df = df.iloc[tr_idx]
            test_df = df.iloc[te_idx]
            X_tr, y_tr = train_df.drop(columns=[target_col]), train_df[target_col]
            X_te, y_te = test_df.drop(columns=[target_col]), test_df[target_col]

            model = SeqTransformerRegressor(**cfg)
            # lightweight training for tuning
            model.use_ema = False
            model.feature_noise_std = 0.0
            model.snapshot_top_k = 1
            model.mc_dropout_passes = 1

            model.fit(X_tr, y_tr)
            L = model.lookback
            X_ctx = pd.concat([X_tr.tail(max(0, L - 1)), X_te], axis=0)
            y_pred = model.predict(X_ctx).loc[y_te.index]
            mae = Metrics.mae(y_te, y_pred)
            maes.append(mae)
            print(f"[CV] fold={fold} MAE={mae:.3f}")

        return float(np.nanmean(maes))

    @staticmethod
    def tune(
        df: pd.DataFrame,
        target_col: str,
        trials: int,
        horizon: int,
        splits: int,
        results_dir: Path,
        seed: int = 42
    ) -> dict:
        if not _optuna_available():
            raise RuntimeError("Optuna is not installed.")

        import optuna
        from optuna.pruners import SuccessiveHalvingPruner

        print(f"[Optuna] CV tuning: trials={trials}, horizon={horizon}, splits={splits}")

        def objective(trial: "optuna.trial.Trial") -> float:
            cfg = dict(
                embed_dim     = trial.suggest_categorical("embed_dim", [128, 192, 256]),
                nhead         = trial.suggest_categorical("nhead", [4, 8]),
                num_layers    = trial.suggest_int("num_layers", 3, 7),
                ff_dim        = trial.suggest_categorical("ff_dim", [384, 512, 768]),
                dropout       = trial.suggest_float("dropout", 0.05, 0.25),
                batch_size    = trial.suggest_categorical("batch_size", [24, 32, 48]),
                epochs        = 130,
                lr            = trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                weight_decay  = trial.suggest_float("weight_decay", 1e-6, 3e-4, log=True),
                lookback      = trial.suggest_categorical("lookback", [336]),
                patience      = 20,
                aggregation   = "last",
                amp           = True,
            )
            print(f"[Optuna][trial {trial.number}] cfg={cfg}")
            mae = OptunaTuner._cv_mae_for_cfg(df, target_col, cfg, horizon, splits, seed)
            trial.set_user_attr("cfg", cfg)
            return mae

        study = optuna.create_study(
            direction="minimize",
            pruner=SuccessiveHalvingPruner(min_resource=20, reduction_factor=3),
        )
        study.optimize(objective, n_trials=trials, show_progress_bar=False)

        best_cfg = study.best_trial.user_attrs["cfg"]
        print("[Optuna] Best MAE:", study.best_value)
        print("[Optuna] Best config:\n" + json.dumps(best_cfg, indent=2))
        (results_dir / "best_optuna_config.json").write_text(json.dumps(best_cfg, indent=2), encoding="utf-8")
        return best_cfg


# Config Factory

class ConfigFactory:
    """Select a good starting config based on dataset name."""
    @staticmethod
    def pick_for_data(data_uri: str) -> dict:
        name = os.path.basename(str(data_uri)).upper()
        if "CLIPOFF" in name:
            print("[CFG] Detected dataset=CLIPOFF -> using TUNED_CFG")
            return TUNED_CFG.copy()
        print("[CFG] Dataset not recognized -> using BEST_CFG")
        return BEST_CFG.copy()


# Pipeline

class ForecasterPipeline:
    """End-to-end training → importance → refit → forecast exporter."""

    def __init__(self, run_cfg: RunConfig, target_col: str = TARGET_COLUMN) -> None:
        self.run_cfg = run_cfg
        self.target_col = target_col
        self.data = DataModule(target_col=target_col)

    def run(self) -> None:
        set_global_seed(self.run_cfg.seed)
        self.run_cfg.ensure_dirs()

        # 1) Load & enrich data
        df = self.data.load_dataframe(self.run_cfg.data_path)
        df = self.data.ensure_calendar_features(df)

        print("[DATA] shape =", df.shape)
        print("[DATA] head columns =", list(df.columns)[:10], "..." if df.shape[1] > 10 else "")

        assert df[self.target_col].notna().all(), "TARGET contains NaN"
        na_cols = [c for c in df.columns if df[c].isna().any()]
        assert not na_cols, f"NaN in columns: {na_cols[:5]}"

        # 2) Config selection: Optuna CV or preset
        used_tuning = False
        if self.run_cfg.tune_with_optuna:
            try:
                cfg = OptunaTuner.tune(
                    df=df,
                    target_col=self.target_col,
                    trials=self.run_cfg.optuna_trials,
                    horizon=self.run_cfg.forecast_h,
                    splits=self.run_cfg.cv_splits,
                    results_dir=self.run_cfg.results_dir,
                    seed=self.run_cfg.seed,
                )
                used_tuning = True
            except RuntimeError as e:
                print(f"[WARN] Optuna tuning skipped ({e}). Falling back to preset.")
                cfg = ConfigFactory.pick_for_data(self.run_cfg.data_path)
        else:
            cfg = ConfigFactory.pick_for_data(self.run_cfg.data_path)

        print("[CFG] Using configuration:\n" + json.dumps(cfg, indent=2))
        (self.run_cfg.results_dir / "training_config_used.json").write_text(
            json.dumps(cfg, indent=2), encoding="utf-8"
        )

        # 3) Permutation importance on the LAST H hours (if feasible)
        h_imp = min(self.run_cfg.forecast_h, len(df) // 3)
        if h_imp >= cfg["lookback"] + 1:
            print(f"[IMPORTANCE] Computing on last {h_imp} hours.")
            train_df = df.iloc[:-h_imp]
            test_df = df.iloc[-h_imp:]
            X_tr, y_tr = train_df.drop(columns=[self.target_col]), train_df[self.target_col]
            X_te, y_te = test_df.drop(columns=[self.target_col]), test_df[self.target_col]

            model_imp = SeqTransformerRegressor(**cfg)
            # keep importance training light and deterministic
            model_imp.use_ema = False
            model_imp.feature_noise_std = 0.0
            model_imp.snapshot_top_k = 1
            model_imp.mc_dropout_passes = 1

            model_imp.fit(X_tr, y_tr)
            try:
                imp_df = FeatureImportance.permutation_last_h(
                    model=model_imp,
                    X_tr=X_tr,
                    X_te=X_te,
                    y_te=y_te,
                    metric_fn=Metrics.mae,
                    n_repeats=3,
                    top_k=20
                )
                imp_csv = self.run_cfg.results_dir / "perm_importance_lastH_top20.csv"
                imp_df.to_csv(imp_csv, index=False, float_format="%.6f")
                print(f"[SAVE] Permutation importance -> {imp_csv.as_posix()}")
            except Exception as e:
                print("[WARN] Permutation importance failed:", e)
        else:
            print("[IMPORTANCE] Skipped (history shorter than lookback + 1).")

        # 4) Refit on FULL data and forecast next H hours
        print("[REFIT] Training on the full dataset and forecasting...")
        X_full, y_full = df.drop(columns=[self.target_col]), df[self.target_col]

        model_full = SeqTransformerRegressor(**cfg)
        # same light settings as before (adjust if you want EMA on final model)
        model_full.use_ema = False
        model_full.feature_noise_std = 0.0
        model_full.snapshot_top_k = 1
        model_full.mc_dropout_passes = 1

        model_full.fit(X_full, y_full)

        X_future = self.data.make_future_features_like_train(df, self.run_cfg.forecast_h)
        X_all = pd.concat([X_full, X_future], axis=0)
        y_all = model_full.predict(X_all)
        y_forecast = y_all.loc[X_future.index]

        out = pd.DataFrame({"timestamp": y_forecast.index, "prediction": y_forecast.values})
        out_path = self.run_cfg.results_dir / f"forecast_h{self.run_cfg.forecast_h}.csv"
        out.to_csv(out_path, index=False, float_format="%.6f")
        print(f"[SAVE] {self.run_cfg.forecast_h}h forecast -> {out_path.as_posix()}")
        print(f"[INFO] Config source: {'optuna(CV)' if used_tuning else 'preset'}")


# CLI

def _bool_env_or_flag(v: str | int | None, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, int):
        return bool(v)
    v = str(v).strip().lower()
    if v in {"1", "true", "yes", "y"}:
        return True
    if v in {"0", "false", "no", "n"}:
        return False
    return default


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Transformer forecasting pipeline (hourly).")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Path to Parquet with DatetimeIndex.")
    parser.add_argument("--h", type=int, default=7 * 24, help="Forecast horizon in hours.")
    parser.add_argument("--tune", type=str, default=os.getenv("TUNE_WITH_OPTUNA", str(int(TUNE_WITH_OPTUNA_DEFAULT))),
                        help="Enable Optuna tuning (1/0 or true/false).")
    parser.add_argument("--trials", type=int, default=int(os.getenv("OPTUNA_TRIALS", OPTUNA_TRIALS_DEFAULT)),
                        help="Number of Optuna trials.")
    parser.add_argument("--splits", type=int, default=3, help="CV splits for Optuna tuning.")
    parser.add_argument("--out", type=str, default="results_transformer", help="Output directory.")
    args = parser.parse_args()

    run_cfg = RunConfig(
        data_path=args.data,
        results_dir=Path(args.out),
        forecast_h=int(args.h),
        tune_with_optuna=_bool_env_or_flag(args.tune, TUNE_WITH_OPTUNA_DEFAULT),
        optuna_trials=int(args.trials),
        cv_splits=int(args.splits),
        seed=42,
    )

    print("[RUN CONFIG]\n" + json.dumps({**asdict(run_cfg), "results_dir": str(run_cfg.results_dir)}, indent=2))
    ForecasterPipeline(run_cfg).run()


if __name__ == "__main__":
    main()
