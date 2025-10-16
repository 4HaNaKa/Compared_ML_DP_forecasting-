"Energy Price Forecasting"


/src:
1) Clean preprocessing pipeline for energy-price time series
This script loads an hourly energy-price dataset (Parquet format) and prepares a clean, leak-safe feature table for forecasting.
It removes invalid or redundant columns, imputes missing values using an IterativeImputer with a RandomForest estimator, and builds lagged and rolling statistical features that depend only on past data.
The pipeline also adds calendar, volatility, imbalance, renewable-share, and interaction features, then optionally performs stationarity and linearity tests.
Finally, it saves the processed dataset to data/processed/ready_database.parquet and produces diagnostic plots and summaries.


2) Walk-forward forecasting pipeline
This module runs a complete model evaluation workflow with classical and machine-learning regressors (SARIMA, SARIMAX, XGB, GBR).
It loads the cleaned feature table, selects exogenous variables, and performs walk-forward time-series cross-validation for multiple horizons (1d, 7d, 14d, 31d).
For each model and horizon, it computes MAE, RMSE, MAPE, RMSLE, and R², saving per-horizon forecasts and overall rankings.
It also performs a recursive-block forecast benchmark (10 sequential 7-day blocks), computes permutation feature importance for each model, and plots last-fold predictions.

3) Transformer Forecasting Pipeline
Pipeline that trains and evaluates a Transformer-based regressor for hourly electricity price prediction.
It features automatic directory setup and configurable runtime parameters, a model architecture with causal attention masking, positional encoding, and temporal embeddings, as well as EMA averaging and Monte Carlo dropout for stability and uncertainty estimation.
The pipeline performs K-Fold evaluation across multiple forecasting horizons (1 d, 7 d, 14 d, 31 d), supports hyperparameter optimization with Optuna, and provides recursive block forecasts along with permutation-based feature importance for interpretability.


4) Consolidate project artifacts
Merges outputs from different stages of the project: predictions, feature-importance tables, and model rankings into unified summary files.
concatenates them with consistent timestamps and column naming, and prepares combined datasets for downstream evaluation or visualization.
The goal is to keep all forecasting outputs synchronized and ready for meta-analysis.


5) Aggregate and visualize feature rankings
This program collects and aggregates feature-importance results from all models. It reads individual *_feature.csv files and model metrics , then computes combined rankings using Borda and RRF (Reciprocal Rank Fusion) methods

6) Evaluation pipeline
It automatically detects valid model columns, applies coverage filtering, and computes peak-aware metrics (e.g., MAE_peak, F1_peak) using quantile-based thresholds.
It also runs Diebold–Mariano tests (with Newey–West correction) to statistically compare forecasting accuracy between model pairs.
Finally, it prints all summaries to the console, saves CSV tables of metrics and DM test results, and plots the last-month performance highlighting price spikes and top-performing models.

INFO*: Forecasts and benchmark results from the Seq2Seq, TiDE, and AutoML models were additionally incorporated from experiments conducted on Google Cloud Platform (GCP).
