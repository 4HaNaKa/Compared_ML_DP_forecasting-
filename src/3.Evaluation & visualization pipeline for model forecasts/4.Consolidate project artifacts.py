# -*- coding: utf-8 -*-
"""
Purpose
- Collect and merge outputs produced by other parts of the project: forecasts, feature importance tables,
  recursive-block predictions, and model rankings.
- Save clean CSVs to a single place.

Inputs (looked up in results/)
- forecasts_ML_h{H}_hours.csv and forecasts_DP_h{H}_hours.csv for H in {24, 168, 336, 744}
- feature_importance_Tide,Seq2Seq,AutoML.csv
- feature_importance_transformer.csv
- feature_importance_xgb.csv
- feature_importance_sarimax.csv
- feature_importance_gbr.csv
- recursive_blocks_h168x10_ML.csv, recursive_blocks_h168x10_DP.csv
- ranking_models_ML.csv, ranking_models_DP.csv

Outputs (saved under data/merged predictions/)
- forecasts_24h.csv, forecasts_168h.csv, forecasts_336h.csv, forecasts_744h.csv
- feature_importance.csv (wide table with all available models side by side)
- recursive_blocks_h168x10.csv (ML and DP aligned on timestamp)
- ranking_models.csv (concatenated and sorted by HorizonHours then MAE)
- For repository requirement, any merged forecasts follow the column order:
  timestamp, XGB, GBR, Observed (Observed when available).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional
import os
import sys

import numpy as np
import pandas as pd


# ========================= PATHS =========================

def bootstrap_paths() -> SimpleNamespace:
    """Locate project root"""
    try:
        here = Path(__file__).resolve()
    except NameError:
        here = Path.cwd().resolve()

    root: Optional[Path] = None
    for probe in [here] + list(here.parents):
        if (probe / "data" / "processed").exists():
            root = probe
            break

    if root is None:
        env = os.environ.get("PROJECT_ROOT")
        if env and (Path(env) / "data" / "processed").exists():
            root = Path(env).resolve()

    if root is None:
        root = here.parent.parent

    data_dir = root / "data"
    processed_dir = data_dir / "processed"
    merged_dir = data_dir / "merged predictions"

    results_dir = root / "results"

    merged_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    return SimpleNamespace(
        project_root=root,
        processed_dir=processed_dir,
        merged_dir=merged_dir,
        results_dir=results_dir,
    )


PATHS = bootstrap_paths()
PROJECT_ROOT: Path = PATHS.project_root
RESULTS_DIR: Path = PATHS.results_dir
OUTPUT_DIR: Path = PATHS.merged_dir


# ========================= HELPERS =========================

def list_candidates_text(folder: Path, hint: str) -> str:
    if not folder.exists():
        return f"Folder does not exist: {folder}"
    names = sorted(p.name for p in folder.glob("*.csv"))
    near = [n for n in names if hint.lower().split(".csv")[0] in n.lower()]
    head = "\n  - ".join(names[:40]) if names else "(no *.csv files)"
    near_text = "\n  - ".join(near) if near else "(no similar names)"
    return f"Available CSVs in {folder}:\n  - {head}\nSimilar to '{hint}':\n  - {near_text}"


def read_csv_smart(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing file: {file_path}\n{list_candidates_text(file_path.parent, file_path.name)}"
        )
    try:
        data_frame = pd.read_csv(file_path, sep=None, engine="python", dtype=str)
    except Exception:
        # Fallback to common separators
        for sep in [",", ";", "\t", "|"]:
            try:
                data_frame = pd.read_csv(file_path, sep=sep, dtype=str)
                break
            except Exception:
                continue
        else:
            raise

    # trim whitespace and remove BOM from headers
    clean_cols = [str(c).strip().lstrip("\ufeff").rstrip("\ufeff") for c in data_frame.columns]
    data_frame.columns = clean_cols
    for col in data_frame.columns:
        data_frame[col] = data_frame[col].astype(str).str.strip()
    return data_frame


def parse_and_sort_timestamp(data_frame: pd.DataFrame, column_name: str = "timestamp") -> pd.DataFrame:
    if column_name not in data_frame.columns:
        return data_frame
    out = data_frame.copy()
    out[column_name] = pd.to_datetime(out[column_name], errors="coerce", utc=False)
    out = out.dropna(subset=[column_name]).sort_values(column_name).reset_index(drop=True)
    return out


def coerce_numeric_columns(data_frame: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
    out = data_frame.copy()
    for column in column_names:
        if column in out.columns:
            out[column] = (
                out[column]
                .astype(str)
                .str.replace(" ", "", regex=False)
                .str.replace("\xa0", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def safe_merge_on_timestamp(left: pd.DataFrame, right: pd.DataFrame, how: str = "outer") -> pd.DataFrame:
    left_sorted = parse_and_sort_timestamp(left, "timestamp")
    right_sorted = parse_and_sort_timestamp(right, "timestamp")
    if "timestamp" in left_sorted.columns and "timestamp" in right_sorted.columns:
        merged = left_sorted.merge(right_sorted, on="timestamp", how=how)
        return parse_and_sort_timestamp(merged, "timestamp")
    return pd.concat([left_sorted.reset_index(drop=True), right_sorted.reset_index(drop=True)], axis=1)


# ========================= MERGE FORECASTS =========================

def merge_forecasts_pair(ml_filename: str, dp_filename: str, output_name: str) -> Path:
    ml_frame = read_csv_smart(RESULTS_DIR / ml_filename)
    dp_frame = read_csv_smart(RESULTS_DIR / dp_filename)

    numeric_candidates = [
        "Observed",
        "XGB",
        "GBR",
        "SARIMA",
        "SARIMAX",
        "Seq2Seq",
        "TiDE",
        "Transformer",
        "AutoML",
    ]
    ml_frame = coerce_numeric_columns(ml_frame, numeric_candidates)
    dp_frame = coerce_numeric_columns(dp_frame, numeric_candidates)

    merged = safe_merge_on_timestamp(ml_frame, dp_frame, how="outer")

    has_observed = "Observed" in merged.columns
    has_observed_x = "Observed_x" in merged.columns
    has_observed_y = "Observed_y" in merged.columns

    if not has_observed and (has_observed_x or has_observed_y):
        if has_observed_x and has_observed_y:
            obs_x = pd.to_numeric(merged["Observed_x"], errors="coerce")
            obs_y = pd.to_numeric(merged["Observed_y"], errors="coerce")
            merged["Observed"] = obs_x.where(obs_y.isna(), obs_x)
        elif has_observed_x:
            merged["Observed"] = pd.to_numeric(merged["Observed_x"], errors="coerce")
        else:
            merged["Observed"] = pd.to_numeric(merged["Observed_y"], errors="coerce")
        merged = merged.drop(columns=[c for c in ["Observed_x", "Observed_y"] if c in merged.columns])

    all_cols = list(merged.columns)
    model_cols = [c for c in all_cols if c not in ["timestamp", "Observed"]]
    ordered = ["timestamp", "Observed"] + sorted(model_cols)
    ordered = [c for c in ordered if c in merged.columns]
    merged = merged[ordered]

    output_path = OUTPUT_DIR / f"{output_name}.csv"
    merged.to_csv(output_path, index=False)
    print(f"[SAVE] forecasts -> {output_path}  rows={len(merged)}  cols={len(merged.columns)}")
    return output_path


# ========================= MERGE FEATURE IMPORTANCE  =========================

def _pad_rows(data_frame: pd.DataFrame, target_len: int) -> pd.DataFrame:
    if len(data_frame) < target_len:
        to_add = target_len - len(data_frame)
        pad = pd.DataFrame({col: pd.NA for col in data_frame.columns}, index=range(to_add))
        data_frame = pd.concat([data_frame, pad], ignore_index=True)
    return data_frame


def build_feature_importance_wide(output_name: str = "feature_importance") -> Path:
    """Read up to 5 FI files and concatenate them side by side. Missing files are skipped."""
    def read_optional(name: str) -> Optional[pd.DataFrame]:
        file_path = RESULTS_DIR / name
        if not file_path.exists():
            print(f"[WARN] missing FI file, skipped: {name}")
            return None
        frame = read_csv_smart(file_path)
        frame.columns = [c.lstrip("\ufeff").rstrip("\ufeff") for c in frame.columns]
        return frame

    source_frames: Dict[str, pd.DataFrame] = {
        "tide_seq2seq_automl": read_optional("feature_importance_Tide,Seq2Seq,AutoML.csv"),
        "transformer": read_optional("feature_importance_transformer.csv"),
        "xgb": read_optional("feature_importance_xgb.csv"),
        "sarimax": read_optional("feature_importance_sarimax.csv"),
        "gbr": read_optional("feature_importance_gbr.csv"),
    }
    source_frames = {k: v for k, v in source_frames.items() if v is not None}
    if not source_frames:
        raise FileNotFoundError("No feature importance files found in results/ directory.")

    numeric_columns_map = {
        "tide_seq2seq_automl": ["Seq2Seq_importance", "TiDE_importance", "AutoML_importance"],
        "transformer": ["transformer_importance"],
        "xgb": ["xgb_importance"],
        "sarimax": ["sarimax_importance"],
        "gbr": ["gbr_importance"],
    }

    fixed_frames: Dict[str, pd.DataFrame] = {}
    for key, frame in source_frames.items():
        numeric_cols = numeric_columns_map.get(key, [])
        fixed_frames[key] = coerce_numeric_columns(frame, numeric_cols)

    max_len = max(len(f) for f in fixed_frames.values())
    fixed_frames = {k: _pad_rows(v, max_len) for k, v in fixed_frames.items()}

    ordered_blocks: List[pd.DataFrame] = []

    if "tide_seq2seq_automl" in fixed_frames:
        f = fixed_frames["tide_seq2seq_automl"]
        cols = [
            "Seq2Seq_feature",
            "Seq2Seq_importance",
            "TiDE_feature",
            "TiDE_importance",
            "AutoML_feature",
            "AutoML_importance",
        ]
        ordered_blocks.append(f[[c for c in cols if c in f.columns]])

    if "transformer" in fixed_frames:
        f = fixed_frames["transformer"]
        cols = ["transformer_feature", "transformer_importance"]
        ordered_blocks.append(f[[c for c in cols if c in f.columns]])

    if "xgb" in fixed_frames:
        f = fixed_frames["xgb"]
        cols = ["xgb_feature", "xgb_importance"]
        ordered_blocks.append(f[[c for c in cols if c in f.columns]])

    if "sarimax" in fixed_frames:
        f = fixed_frames["sarimax"]
        cols = ["sarimax_feature", "sarimax_importance"]
        ordered_blocks.append(f[[c for c in cols if c in f.columns]])

    if "gbr" in fixed_frames:
        f = fixed_frames["gbr"]
        cols = ["gbr_feature", "gbr_importance"]
        ordered_blocks.append(f[[c for c in cols if c in f.columns]])

    output_frame = pd.concat(ordered_blocks, axis=1)
    output_path = OUTPUT_DIR / f"{output_name}.csv"
    output_frame.to_csv(output_path, index=False)
    print(f"[SAVE] feature importance (wide) -> {output_path}  rows={len(output_frame)}  cols={len(output_frame.columns)}")
    return output_path


# ========================= MERGE RECURSIVE BLOCKS =========================

def merge_recursive_blocks_side_by_side(
    ml_filename: str, dp_filename: str, output_name: str
) -> Path:

    ml_frame = read_csv_smart(RESULTS_DIR / ml_filename)
    dp_frame = read_csv_smart(RESULTS_DIR / dp_filename)

    ml_frame = parse_and_sort_timestamp(ml_frame)
    dp_frame = parse_and_sort_timestamp(dp_frame)

    ml_frame = coerce_numeric_columns(ml_frame, [c for c in ml_frame.columns if c != "timestamp"])
    dp_frame = coerce_numeric_columns(dp_frame, [c for c in dp_frame.columns if c != "timestamp"])

    merged = safe_merge_on_timestamp(ml_frame, dp_frame, how="inner")

    if "Observed" not in merged.columns:
        observed_x = pd.to_numeric(merged.get("Observed_x"), errors="coerce")
        observed_y = pd.to_numeric(merged.get("Observed_y"), errors="coerce")
        merged["Observed"] = observed_x.where(observed_x.notna(), observed_y)
        merged = merged.drop(columns=[c for c in ["Observed_x", "Observed_y"] if c in merged.columns])

    def adopt(base: str) -> None:
        nonlocal merged
        if f"{base}_x" in merged.columns and base not in merged.columns:
            merged[base] = merged[f"{base}_x"]
            merged = merged.drop(columns=[f"{base}_x"])
        if f"{base}_y" in merged.columns and base not in merged.columns:
            merged[base] = merged[f"{base}_y"]
            merged = merged.drop(columns=[f"{base}_y"])

    for base_name in [
        "XGB",
        "GBR",
        "SARIMA",
        "SARIMAX",
        "Transformer",
        "TiDE",
        "Seq2Seq",
        "AutoML",
    ]:
        adopt(base_name)

    final_order = [
        "timestamp",
        "Observed",
        "XGB",
        "GBR",
        "SARIMA",
        "SARIMAX",
        "Transformer",
        "TiDE",
        "Seq2Seq",
        "AutoML",
    ]
    final_order = [c for c in final_order if c in merged.columns]
    merged = merged[final_order]

    output_path = OUTPUT_DIR / f"{output_name}.csv"
    merged.to_csv(output_path, index=False)
    print(
        f"[SAVE] recursive blocks (ML|DP) -> {output_path}  rows={len(merged)}  cols={len(merged.columns)}"
    )
    return output_path


# ========================= MERGE RANKINGS =========================

def _to_numeric(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace("\xa0", "", regex=False)
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )


def concat_rankings_no_source(input_filenames: List[str], output_name: str) -> Path:
    frames: List[pd.DataFrame] = []
    for name in input_filenames:
        frame = read_csv_smart(RESULTS_DIR / name)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError("No ranking files to concatenate.")

    output_frame = pd.concat(frames, ignore_index=True)

    if "HorizonHours" in output_frame.columns:
        output_frame["HorizonHours"] = _to_numeric(output_frame["HorizonHours"]).astype("Int64")
    else:
        print("[WARN] ranking: missing 'HorizonHours' column; horizon sorting skipped.")

    if "MAE" in output_frame.columns:
        output_frame["MAE"] = _to_numeric(output_frame["MAE"])
    else:
        print("[WARN] ranking: missing 'MAE' column; MAE sorting skipped.")

    sort_columns = [c for c in ["HorizonHours", "MAE"] if c in output_frame.columns]
    if sort_columns:
        output_frame = output_frame.sort_values(by=sort_columns, kind="stable").reset_index(drop=True)

    output_path = OUTPUT_DIR / f"{output_name}.csv"
    output_frame.to_csv(output_path, index=False)
    print(
        f"[SAVE] rankings (sorted by HorizonHours then MAE) -> {output_path}  rows={len(output_frame)}  cols={len(output_frame.columns)}"
    )
    return output_path


# ========================= MAIN =========================

def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Results dir : {RESULTS_DIR}")
    print(f"Output dir  : {OUTPUT_DIR}")

    # 1) Forecasts (ML + DP)
    for horizon, label in [(24, "24h"), (168, "168h"), (336, "336h"), (744, "744h")]:
        ml_name = f"forecasts_ML_h{horizon}_hours.csv"
        dp_name = f"forecasts_DP_h{horizon}_hours.csv"
        try:
            merge_forecasts_pair(ml_name, dp_name, f"forecasts_{label}")
        except FileNotFoundError as e:
            print(f"[WARN] forecasts (h={horizon}) skipped: {e}")

    # 2) Feature importance (wide)
    try:
        build_feature_importance_wide(output_name="feature_importance")
    except FileNotFoundError as e:
        print(f"[WARN] feature importance skipped: {e}")

    # 3) Recursive blocks (ML vs DP, 168x10)
    try:
        merge_recursive_blocks_side_by_side(
            ml_filename="recursive_blocks_h168x10_ML.csv",
            dp_filename="recursive_blocks_h168x10_DP.csv",
            output_name="recursive_blocks_h168x10",
        )
    except FileNotFoundError as e:
        print(f"[WARN] recursive blocks skipped: {e}")

    # 4) Rankings (concat without 'source')
    try:
        concat_rankings_no_source(
            input_filenames=["ranking_models_ML.csv", "ranking_models_DP.csv"],
            output_name="ranking_models",
        )
    except FileNotFoundError as e:
        print(f"[WARN] ranking merge skipped: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(130)
