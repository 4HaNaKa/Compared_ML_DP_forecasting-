"""
Aggregate and visualize feature rankings from multiple models.

    Implements a fusion of Borda count and Reciprocal Rank Fusion (RRF),
    optionally weighted by model performance metrics (e.g. MAE).

Input:
- top20_ranks.csv
- general_metrics_7d.csv: model metrics file

Output:
- CSV with aggregated ranking
- Heatmap of feature ranks across models
- Bar chart of final aggregated scores
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# === SETTINGS ===
RANKS_PATH = "top20_ranks.csv"          # input file with feature ranks
RANKS_SHEET = 0                         # Excel sheet index/name (if Excel)
USE_METRIC_WEIGHTS = True               # if False -> equal weights
MODEL_METRICS_PATH = "general_metrics_7d.csv"   # file with model metrics
MODEL_METRIC_NAME = "MAE"               # metric to define model weights
K_RRF = 60                              # RRF constant
OUT_DIR = Path("feature_rankings")      # output directory
TOP_N_PLOT = 15                         # number of features in plots


OUT_DIR.mkdir(parents=True, exist_ok=True)

# Helpers
def _ts() -> str:
    """Return timestamp string for unique filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_to_csv(df: pd.DataFrame, path: Path, **kwargs) -> Path:
    """Save DataFrame to CSV; if file is locked, append timestamp to filename."""
    path = Path(path)
    try:
        df.to_csv(path, **kwargs)
        print(f"[OK] Saved: {path}")
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{_ts()}{path.suffix}")
        df.to_csv(alt, **kwargs)
        print(f"[WARN] Could not overwrite (locked?) -> saved to: {alt}")
        return alt


def safe_savefig(path: Path, **kwargs) -> Path:
    """Save matplotlib figure; if file is locked, append timestamp."""
    path = Path(path)
    try:
        plt.savefig(path, **kwargs)
        print(f"[OK] Saved: {path}")
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{_ts()}{path.suffix}")
        plt.savefig(alt, **kwargs)
        print(f"[WARN] Could not overwrite (locked?) -> saved to: {alt}")
        return alt


def read_csv_smart(path: Path) -> pd.DataFrame:
    """Load CSV with auto-detected separator; fallback to common delimiters."""
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        for sep in [",", ";", "\t", "|"]:
            try:
                return pd.read_csv(path, sep=sep)
            except Exception:
                continue
        raise


def load_ranks(path: str, sheet=0) -> pd.DataFrame:
    """Load feature ranking table (Excel/CSV) with fallbacks and sanity checks."""
    p = Path(path)
    ext = p.suffix.lower()

    if ext in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        try:
            df = pd.read_excel(p, sheet_name=sheet)  # requires openpyxl
        except ImportError as e:
            alt = p.with_suffix(".csv")
            if alt.exists():
                print(f"[WARN] Missing openpyxl – using CSV: {alt.name}")
                df = read_csv_smart(alt)
            else:
                raise RuntimeError(
                    "Missing 'openpyxl' for .xlsx. Install via: pip install openpyxl\n"
                    f"Or save as CSV: {alt.name}"
                ) from e
    elif ext == ".xls":
        try:
            df = pd.read_excel(p, sheet_name=sheet)  # requires xlrd
        except ImportError as e:
            raise RuntimeError("Missing 'xlrd' for .xls (pip install xlrd).") from e
    else:
        df = read_csv_smart(p)

    # Drop technical columns like "LP."
    df = df.loc[:, [c for c in df.columns if str(c).strip().lower() not in {"lp.", "lp", "rank"}]]

    # Fix if wrong separator produced a single column
    if df.shape[1] == 1:
        sample = str(df.iloc[0, 0])
        for sep in [";", "\t", "|"]:
            if sep in sample:
                print(f"[WARN] Detected embedded separator '{sep}' – splitting column.")
                df = df.iloc[:, 0].str.split(sep, expand=True)
                break

    df = df.dropna(how="all").dropna(axis=1, how="all")

    if df.shape[1] < 2:
        raise RuntimeError(f"Ranking looks wrong (columns={df.shape[1]}). Check file: {p.name}")

    return df


def load_metrics(path: str, metric_col: str) -> pd.DataFrame:
    """Load model metrics file (CSV/Excel) and return DF with index=model."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    if p.suffix.lower() in {".xlsx", ".xlsm", ".xltx", ".xltm", ".xls"}:
        try:
            df = pd.read_excel(p)
        except ImportError:
            df = read_csv_smart(p.with_suffix(".csv"))
    else:
        df = read_csv_smart(p)

    if "model" in df.columns:
        df = df.set_index("model")

    if metric_col not in df.columns:
        raise KeyError(f"Metric '{metric_col}' not found in {path}. Available: {list(df.columns)}")

    return df


# 1) Load ranking
rank_df = load_ranks(RANKS_PATH, sheet=RANKS_SHEET)
print(f"[INFO] Loaded ranks: {RANKS_PATH}  -> shape {rank_df.shape}")
print(f"[INFO] Columns (models): {list(rank_df.columns)}")

model_cols = list(rank_df.columns)

# Convert to long format: (model, feature, rank)
rows = []
for m in model_cols:
    col = (
        rank_df[m]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .map(lambda x: ALIASES.get(x, x))
        .drop_duplicates(keep="first")
        .tolist()
    )
    for r, feat in enumerate(col, start=1):
        rows.append((m, feat, r))

if not rows:
    raise RuntimeError("No ranking data after preprocessing.")

long = pd.DataFrame(rows, columns=["model", "feature", "rank"]).astype({"rank": int})
TOP_N = int(long["rank"].max())
print(f"[INFO] Long-format: {long.shape[0]} rows, TOP_N={TOP_N}")


# 2) Model weights
if USE_METRIC_WEIGHTS and Path(MODEL_METRICS_PATH).exists():
    metrics = load_metrics(MODEL_METRICS_PATH, MODEL_METRIC_NAME)
    common = [m for m in model_cols if m in metrics.index]
    if not common:
        print("[WARN] No matching models in metrics file – using equal weights.")
        weights = pd.Series(1.0 / len(model_cols), index=model_cols)
    else:
        inv = 1.0 / metrics.loc[common, MODEL_METRIC_NAME].astype(float)
        inv = inv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        w_norm = inv / inv.sum() if inv.sum() > 0 else pd.Series(1.0 / len(common), index=common)
        weights = pd.Series(1.0 / len(model_cols), index=model_cols)
        weights.loc[common] = w_norm
else:
    weights = pd.Series(1.0 / len(model_cols), index=model_cols)

long = long.merge(weights.rename("w"), left_on="model", right_index=True, how="left")
long["w"] = long["w"].fillna(1.0 / len(model_cols))


# 3) Aggregations
long["borda_points"] = (TOP_N + 1 - long["rank"])
long["wborda"] = long["w"] * long["borda_points"]
long["rrf"] = long["w"] / (K_RRF + long["rank"])

freq = (long.groupby("feature")["model"].nunique().rename("models_count").sort_values(ascending=False))
freq_frac = (freq / len(model_cols)).rename("models_frac")
ranks_stats = (long.groupby("feature")["rank"].agg(["mean", "median"])
               .rename(columns={"mean": "mean_rank", "median": "median_rank"}))

borda = long.groupby("feature")["borda_points"].sum().rename("borda")
wborda = long.groupby("feature")["wborda"].sum().rename("wborda")
rrf = long.groupby("feature")["rrf"].sum().rename("rrf")

out = pd.concat([borda, wborda, rrf, ranks_stats, freq, freq_frac], axis=1).fillna(0.0)

# Normalize 0–1
for col in ["borda", "wborda", "rrf"]:
    mn, mx = out[col].min(), out[col].max()
    out[col + "_norm"] = 0.0 if mx == mn else (out[col] - mn) / (mx - mn)

# Final score: average of normalized WBorda and RRF
out["final_score"] = (out["wborda_norm"] + out["rrf_norm"]) / 2.0
out = out.sort_values("final_score", ascending=False)


# 4) Save results
out_path = OUT_DIR / "feature_ranking_aggregated.csv"
safe_to_csv(out, out_path, encoding="utf-8-sig", index_label="feature")
print("[INFO] Top 5 (preview):")
print(out.head(5).round(4))


# 5) Heatmap of top features
top_feats = out.head(TOP_N_PLOT).index.tolist()
heat = (
    long[long["feature"].isin(top_feats)]
    .pivot_table(index="feature", columns="model", values="rank", aggfunc="min")
    .reindex(columns=model_cols)
    .loc[top_feats]
)

data = heat.to_numpy(dtype=float)
masked = np.ma.masked_invalid(data)

fig_w = 2 + 0.8 * max(1, len(model_cols))
fig_h = 0.55 * max(1, len(top_feats)) + 1
plt.figure(figsize=(fig_w, fig_h))

cmap = plt.cm.viridis_r.copy()
cmap.set_bad(color="#f2f2f2")

im = plt.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=1, vmax=TOP_N)

cbar = plt.colorbar(im, pad=0.01)
cbar.set_label("Rank (1 = best)", rotation=270, labelpad=12)
ticks = np.unique([1, TOP_N] + [int(x) for x in np.linspace(1, TOP_N, 5)])
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
heat_path = OUT_DIR / "heatmap_ranks.png"
safe_savefig(heat_path, dpi=200, bbox_inches="tight")
plt.close()


# 6) Barplot of final_score
top_bar = out.head(TOP_N_PLOT).iloc[::-1]
plt.figure(figsize=(10, 0.4 * max(1, len(top_bar))))
plt.barh(top_bar.index, top_bar["final_score"])
plt.xlabel("Final score (avg of WBorda_norm & RRF_norm)")
plt.tight_layout()
bar_path = OUT_DIR / "bar_final_score.png"
safe_savefig(bar_path, dpi=150, bbox_inches="tight")
plt.close()
