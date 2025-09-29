"Energy Price Forecasting"

1) Clean preprocessing pipeline for energy-price time series
  Loads hourly parquet with a datetime column, builds a leak-safe feature set, drops zero-variance and highly correlated columns, imputes with IterativeImputer(RandomForest), optional stationarity tests, and STL plot. Saves a clean table.
  Input:
     -data/raw/raw_database.parquet
  Outputs:
    -data/processed/ready_database.parquet
    -results/plots/stl_decomposition.png

2) Walk-forward forecasting pipeline
  Runs walk-forward time series CV (1d,7d,14d,31d,62d) for XGB, GBR, SARIMA, SARIMAX; reports MAE/RMSE/MAPE/RMSLE/R²; saves model ranking and last-fold plots. Includes simple feature importance helpers for XGB, GBR, SARIMAX and residual diagnostics.
   Outputs
    -results/ranking_models.csv
    -results/plot_LAST_FOLD_ALL.png and results/plot_LAST_FOLD_<MODEL>.png
    -results/xgb_top_features.csv, results/gbr_top_features.csv, results/top_features_gbr_xgb.csv

3) Transformer Forecasting Pipeline
   Trains a Transformer regressor on hourly features, computes permutation importance on the latest window, runs multi-horizon walk-forward (1d, 7d, 14d, 31d, 62d), and a 90/10 holdout. Saves forecasts and metrics tables.
   Outputs
      -results/forecasts_walkforward_multi_horizon.csv
      -results/predictions_holdout_90_10.csv
      -results/tables/transformer/perm_importance_lastH_top20.csv

4) Aggregate and visualize feature rankings
  Fuses feature rankings from multiple models using Borda count and Reciprocal Rank Fusion (RRF). Can weight models by a metric (for example MAE). Produces an aggregated table and simple plots.
  Inputs:
      -top20_ranks.csv
      -general_metrics_7d.csv (optional, for weights)
  Outputs:
      -results/tables/features/feature_ranking_aggregated.csv
      -results/plots/features/heatmap_ranks.png, results/plots/features/bar_final_score.png

5) Evaluation & visualization pipeline for model forecasts (single file)
  Takes one predictions table (CSV/Parquet) with a timestamp column, the observed target, and any model columns. Cleans and validates, auto-detects models, computes general metrics on full sample and standard windows (24h, 7d, 14d, 31d), computes peak-aware metrics (counts, MAE on peaks/non-peaks, WMAE, peak precision/recall/F1), runs Diebold–Mariano tests (MAE/MSE losses, HAC small-sample correction), and draws clear plots (sliding windows, Observed vs selected models with peak markers).
  Inputs: merged_predictions.csv (with columns like timestamp,Observed,<Model1>,<Model2>,...)
  Outputs: metrics CSVs, DM tests, and plots saved under src/3.tests/results/
