# -*- coding: utf-8 -*-
"""
Aggregate and visualize feature rankings from multiple models.

Loads feature lists from a merged feature-importance file with '*_feature' columns.
Builds an aggregated ranking using Borda and RRF, optionally weighted by model MAE.
Model weights are read from 'ranking_models.csv' filtered to HorizonHours == 168.

Inputs
- feature_importance.csv
- ranking_models.csv

Outputs
- results/tables/features/feature_ranking_aggregated.csv
- results/plots/features/heatmap_ranks.png
- results/plots/features/bar_final_score.png
"""

from types import SimpleNamespace
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
# ========================= PATHS BOOTSTRAP =========================

def bootstrap_paths_models() -> SimpleNamespace:
    """Locate project root and build common directories."""
    try:
        here = Path(__file__).resolve()
    except NameError:
        here = Path.cwd().resolve()

    root = None
    for p in [here] + list(here.parents):
        if (p / "data" / "merged predictions").exists() or (p / "data" / "processed").exists():
            root = p
            break

    if root is None:
        env = os.environ.get("PROJECT_ROOT")
        if env and ((Path(env) / "data" / "merged predictions").exists() or (Path(env) / "data" / "processed").exists()):
            root = Path(env).resolve()

    if root is None:
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

# ========================= SETTINGS =========================

RANKS_PATH = "feature_importance.csv"
USE_METRIC_WEIGHTS = True
MODEL_METRICS_PATH = "ranking_models.csv"
MODEL_METRIC_NAME = "MAE"
HORIZON_FOR_WEIGHTS = 168
DECIMAL_PLACES_METRICS = 6
K_RRF = 60
TOP_N_PLOT = 15

SHOW_PLOTS = True           # show windows with plots after saving
SHOW_PLOTS_BLOCK = False    # whether to block the window (True) or let it go (False)
PRINT_FULL_TABLE = True     # print the entire table in logs
PRINT_ROWS_LIMIT = None     # None = entire table; or, for example, 200
ROUND_LOG_DECIMALS = 4      # rounding in logs

OUT_TABLES = Path("results/tables/features"); OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_PLOTS  = Path("results/plots/features");  OUT_PLOTS.mkdir(parents=True, exist_ok=True)

# Optional feature-name normalization
ALIASES: Dict[str, str] = {}

# Map column prefixes to canonical model names
MODEL_ALIASES: Dict[str, str] = {
    "xgb": "XGB",
    "gbr": "GBR",
    "sarimax": "SARIMAX",
    "sarima": "SARIMA",
    "tide": "TiDE",
    "seq2seq": "Seq2Seq",
    "transformer": "Transformer",
    "automl": "AutoML",
}

# ========================= HELPERS =========================

def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def read_csv_smart(path: Path) -> pd.DataFrame:
    """Try auto-sep, then common fallbacks."""
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        for sep in [",", ";", "\t", "|"]:
            try:
                return pd.read_csv(path, sep=sep)
            except Exception:
                continue
        raise

def safe_to_csv(df: pd.DataFrame, path: Path, **kwargs) -> Path:
    """Save DataFrame with a fallback name if locked."""
    path = Path(path)
    try:
        df.to_csv(path, **kwargs)
        print(f"[OK] Saved: {path}")
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{_ts()}{path.suffix}")
        df.to_csv(alt, **kwargs)
        print(f"[WARN] Locked? Saved to: {alt}")
        return alt

def safe_savefig(path: Path, **kwargs) -> Path:
    """Save figure with a fallback name if locked."""
    path = Path(path)
    try:
        plt.savefig(path, **kwargs)
        print(f"[OK] Saved: {path}")
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{_ts()}{path.suffix}")
        plt.savefig(alt, **kwargs)
        print(f"[WARN] Locked? Saved to: {alt}")
        return alt

def _canonical_model_name(col: str) -> str:
    base = col.lower().replace("_feature", "")
    return MODEL_ALIASES.get(base, col.replace("_feature", ""))

# ========================= LOADERS =========================

def load_ranks_from_feature_importance(path: Path) -> pd.DataFrame:
    """Read only '*_feature' columns and rename them to model names."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = read_csv_smart(path)

    keep_cols = [c for c in df.columns if str(c).lower().endswith("_feature")]
    if not keep_cols:
        raise RuntimeError("No '*_feature' columns found.")

    df = df[keep_cols].copy()
    rename_map = {c: _canonical_model_name(c) for c in keep_cols}
    df = df.rename(columns=rename_map)
    df = df.dropna(how="all").dropna(axis=1, how="all")

    if df.shape[1] < 2:
        raise RuntimeError(f"Too few model columns after filtering: {list(df.columns)}")

    return df

def load_metrics_ranking_models(path: Path,
                                metric_col: str,
                                horizon_hours: int) -> pd.DataFrame:
    """Load ranking_models.csv, keep HorizonHours == given, index by model, round metrics."""
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    df = read_csv_smart(path)
    required = {"model", "HorizonHours", metric_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}")

    df = df[df["HorizonHours"] == horizon_hours].copy()
    if df.empty:
        raise RuntimeError(f"No rows for HorizonHours == {horizon_hours} in {path}")

    df["model"] = df["model"].astype(str)
    for col in ["MAE", "RMSE", "MAPE", "RMSLE", "R2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(DECIMAL_PLACES_METRICS)

    return df.set_index("model")

# ========================= CORE LOGIC =========================

def compute_long_table(ranks_wide: pd.DataFrame,
                       aliases: Dict[str, str]) -> Tuple[pd.DataFrame, int, List[str]]:
    """Convert wide ranks to long format (model, feature, rank)."""
    model_cols = list(ranks_wide.columns)
    rows: List[Tuple[str, str, int]] = []

    for model in model_cols:
        col = (
            ranks_wide[model]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .map(lambda x: aliases.get(x, x))
            .drop_duplicates(keep="first")
            .tolist()
        )
        for rank, feature in enumerate(col, start=1):
            rows.append((model, feature, rank))

    if not rows:
        raise RuntimeError("No ranking data after preprocessing.")

    long_df = pd.DataFrame(rows, columns=["model", "feature", "rank"]).astype({"rank": int})
    top_n = int(long_df["rank"].max())
    return long_df, top_n, model_cols

def build_weights(model_cols: List[str],
                  metrics_path: Path,
                  metric_name: str,
                  horizon_hours: int,
                  use_weights: bool) -> Tuple[pd.Series, pd.DataFrame | None]:
    """Return per-model weights and the metrics frame (if used)."""
    if not use_weights or not metrics_path.exists():
        weights = pd.Series(1.0 / len(model_cols), index=model_cols)
        return weights, None

    metrics_df = load_metrics_ranking_models(metrics_path, metric_name, horizon_hours)
    common = [m for m in model_cols if m in metrics_df.index]
    if not common:
        print("[WARN] No matching models in metrics file -> equal weights.")
        weights = pd.Series(1.0 / len(model_cols), index=model_cols)
        return weights, metrics_df

    inv = 1.0 / metrics_df.loc[common, metric_name].astype(float)
    inv = inv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    w_norm = inv / inv.sum() if inv.sum() > 0 else pd.Series(1.0 / len(common), index=common)

    weights = pd.Series(1.0 / len(model_cols), index=model_cols)
    weights.loc[common] = w_norm
    return weights, metrics_df

def aggregate_scores(long_df: pd.DataFrame,
                     model_cols: List[str],
                     top_n: int) -> pd.DataFrame:
    """Compute Borda, weighted Borda, RRF and summary stats."""
    freq = long_df.groupby("feature")["model"].nunique().rename("models_count").sort_values(ascending=False)
    freq_frac = (freq / len(model_cols)).rename("models_frac")

    long_df["borda_points"] = (top_n + 1 - long_df["rank"])
    long_df["wborda"] = long_df["w"] * long_df["borda_points"]
    long_df["rrf"] = long_df["w"] / (K_RRF + long_df["rank"])

    ranks_stats = (
        long_df.groupby("feature")["rank"]
        .agg(["mean", "median"])
        .rename(columns={"mean": "mean_rank", "median": "median_rank"})
    )
    borda = long_df.groupby("feature")["borda_points"].sum().rename("borda")
    wborda = long_df.groupby("feature")["wborda"].sum().rename("wborda")
    rrf = long_df.groupby("feature")["rrf"].sum().rename("rrf")

    out = pd.concat([borda, wborda, rrf, ranks_stats, freq, freq_frac], axis=1).fillna(0.0)

    for col in ["borda", "wborda", "rrf"]:
        mn, mx = out[col].min(), out[col].max()
        out[col + "_norm"] = 0.0 if mx == mn else (out[col] - mn) / (mx - mn)

    out["final_score"] = (out["wborda_norm"] + out["rrf_norm"]) / 2.0
    out = out.sort_values("final_score", ascending=False)
    return out

# ========================= PLOTS =========================

def plot_heatmap(long_df: pd.DataFrame,
                 top_features: List[str],
                 model_cols: List[str],
                 top_n: int,
                 out_path: Path,
                 show: bool = False) -> None:
    heat = (
        long_df[long_df["feature"].isin(top_features)]
        .pivot_table(index="feature", columns="model", values="rank", aggfunc="min")
        .reindex(columns=model_cols)
        .loc[top_features]
    )
    data = heat.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)

    fig_w = 2 + 0.8 * max(1, len(model_cols))
    fig_h = 0.55 * max(1, len(top_features)) + 1
    plt.figure(figsize=(fig_w, fig_h))

    cmap = plt.cm.viridis_r.copy()
    cmap.set_bad(color="#f2f2f2")

    im = plt.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=1, vmax=top_n)
    cbar = plt.colorbar(im, pad=0.01)
    cbar.set_label("Rank (1 = best)", rotation=270, labelpad=12)
    ticks = np.unique([1, top_n] + [int(x) for x in np.linspace(1, top_n, 5)])
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([str(t) for t in ticks])

    plt.yticks(range(len(heat.index)), heat.index, fontsize=9)
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=30, ha="right", fontsize=9)
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, len(heat.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(heat.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    safe_savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show(block=SHOW_PLOTS_BLOCK)
    plt.close()


def plot_final_score_bars(out_table: pd.DataFrame,
                          out_path: Path,
                          show: bool = False) -> None:
    top_bar = out_table.head(TOP_N_PLOT).iloc[::-1]
    plt.figure(figsize=(10, 0.4 * max(1, len(top_bar))))
    plt.barh(top_bar.index, top_bar["final_score"])
    plt.xlabel("Final score (avg of WBorda_norm & RRF_norm)")
    plt.tight_layout()
    safe_savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show(block=SHOW_PLOTS_BLOCK)
    plt.close()

# ========================= MAIN =========================

def main() -> None:
    # Paths
    paths = bootstrap_paths_models()
    print(f"[INFO] Project root: {paths.project_root}")
    print(f"[INFO] Data dir: {paths.data_dir}")
    print(f"[INFO] Results dir: {paths.results_dir}")

    # Inputs
    ranks_path = paths.merged_predictions_dir / RANKS_PATH
    metrics_path = paths.merged_predictions_dir / MODEL_METRICS_PATH
    if not metrics_path.exists():
        alt = paths.results_dir / MODEL_METRICS_PATH
        if alt.exists():
            metrics_path = alt

    # Outputs
    out_tables_dir = paths.results_dir / "tables" / "features"
    out_plots_dir = paths.results_dir / "plots" / "features"
    out_tables_dir.mkdir(parents=True, exist_ok=True)
    out_plots_dir.mkdir(parents=True, exist_ok=True)

    # Load ranks
    rank_df = load_ranks_from_feature_importance(ranks_path)
    print(f"[INFO] Loaded ranks: {ranks_path} -> shape {rank_df.shape}")

    long_df, top_n, model_cols = compute_long_table(rank_df, ALIASES)

    # Weights
    weights, metrics_df = build_weights(
        model_cols=model_cols,
        metrics_path=metrics_path,
        metric_name=MODEL_METRIC_NAME,
        horizon_hours=HORIZON_FOR_WEIGHTS,
        use_weights=USE_METRIC_WEIGHTS,
    )
    long_df = long_df.merge(weights.rename("w"), left_on="model", right_index=True, how="left")
    long_df["w"] = long_df["w"].fillna(1.0 / len(model_cols))

    # Aggregate
    out_table = aggregate_scores(long_df, model_cols, top_n)

    # Save table
    out_path = out_tables_dir / "feature_ranking_aggregated.csv"
    safe_to_csv(out_table, out_path, encoding="utf-8-sig", index_label="feature")

    if PRINT_FULL_TABLE:
        with pd.option_context(
                "display.max_rows", None if PRINT_ROWS_LIMIT is None else PRINT_ROWS_LIMIT,
                "display.max_columns", None,
                "display.width", 2000,
                "display.max_colwidth", None,
                "display.float_format", (lambda x: f"{x:.{ROUND_LOG_DECIMALS}f}")
        ):
            print("[INFO] Aggregated table:")
            if PRINT_ROWS_LIMIT is None:
                print(out_table.to_string(index=True))
            else:
                print(out_table.head(PRINT_ROWS_LIMIT).to_string(index=True))
    else:
        print("[INFO] Top 5:")
        print(out_table.head(5).round(ROUND_LOG_DECIMALS))

    # Plots
    top_features = out_table.head(TOP_N_PLOT).index.tolist()
    plot_heatmap(
        long_df=long_df,
        top_features=top_features,
        model_cols=model_cols,
        top_n=top_n,
        out_path=out_plots_dir / "heatmap_ranks.png",
        show=SHOW_PLOTS,
    )
    plot_final_score_bars(out_table, out_plots_dir / "bar_final_score.png", show=SHOW_PLOTS)

    if USE_METRIC_WEIGHTS and metrics_df is not None:
        used_models = [m for m in model_cols if m in metrics_df.index]
        print("[DEBUG] Weighted models:", used_models)
        print("[DEBUG] MAE:", metrics_df.loc[used_models, "MAE"].to_dict())
        print("[DEBUG] Weights:", weights.loc[used_models].round(4).to_dict())

if __name__ == "__main__":
    main()
