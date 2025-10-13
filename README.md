"Energy Price Forecasting"

1) Clean preprocessing pipeline for energy-price time series
  Loads hourly parquet with a datetime column, builds a leak-safe feature set, drops zero-variance and highly correlated columns, imputes with IterativeImputer(RandomForest), optional stationarity tests, and STL plot. Saves a clean table.


2) Walk-forward forecasting pipeline
  Runs walk-forward time series CV (1d,7d,14d,31d,62d) for XGB, GBR, SARIMA, SARIMAX; reports MAE/RMSE/MAPE/RMSLE/R²; saves model ranking and last-fold plots. Includes simple feature importance helpers for XGB, GBR, SARIMAX and residual diagnostics.

3) Transformer Forecasting Pipeline
   Trains a Transformer regressor on hourly features, computes permutation importance on the latest window, runs multi-horizon walk-forward (1d, 7d, 14d, 31d, 62d), and a 90/10 holdout. Saves forecasts and metrics tables.


4) Aggregate and visualize feature rankings
  Fuses feature rankings from multiple models using Borda count and Reciprocal Rank Fusion (RRF). Can weight models by a metric (for example MAE). Produces an aggregated table and simple plots.


5) Evaluation & visualization pipeline for model forecasts (single file)
  Takes one predictions table (CSV/Parquet) with a timestamp column, the observed target, and any model columns. Cleans and validates, auto-detects models, computes general metrics on full sample and standard windows (24h, 7d, 14d, 31d), computes peak-aware metrics (counts, MAE on peaks/non-peaks, WMAE, peak precision/recall/F1), runs Diebold–Mariano tests (MAE/MSE losses, HAC small-sample correction), and draws clear plots (sliding windows, Observed vs selected models with peak markers).
