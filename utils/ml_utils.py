"""Shared helpers for the Assignment 2 ML pipeline (training, inference, monitoring)."""
import glob
import os

import numpy as np
import pandas as pd

# ---------- paths (relative to the project root, /opt/airflow inside the container) ----------
MODEL_BANK = "model_bank/"
GOLD_FEATURES = "datamart/gold/feature_store/"
GOLD_LABELS = "datamart/gold/label_store/"
GOLD_PREDICTIONS = "datamart/gold/model_predictions/"
GOLD_MONITORING = "datamart/gold/model_monitoring/"
GOLD_PLOTS = "datamart/gold/monitoring_plots/"

NON_FEATURE_COLS = ["Customer_ID", "loan_id", "snapshot_date"]

# ---------- governance thresholds (documented in the deck) ----------
MIN_OOT_AUC = 0.65      # a model must reach this out-of-time AUC to be promoted / to count as healthy
PSI_WARN = 0.10         # PSI below this = stable
PSI_ALERT = 0.25        # PSI above this = unstable, investigate / retrain


def suffix(date_str):
    return date_str.replace("-", "_")


def get_spark(app="ml_pipeline"):
    import pyspark
    spark = (pyspark.sql.SparkSession.builder.appName(app).master("local[*]")
             .config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "4")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def read_gold_folder(spark, folder):
    """Read every partition in a gold folder into one pandas DataFrame (None if the folder is empty)."""
    files = sorted(glob.glob(os.path.join(folder, "*")))
    if not files:
        return None
    return spark.read.parquet(*files).toPandas()


def coerce_types(df, num_cols, cat_cols):
    """Make sure numeric columns are numeric and categorical columns are clean strings."""
    df = df.copy()
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown").astype(str)
    return df


# ---------- PSI (population stability index) ----------
def psi_numeric(ref, cur, bins=10, eps=1e-4):
    ref = pd.to_numeric(pd.Series(ref), errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(pd.Series(cur), errors="coerce").dropna().to_numpy(dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        return np.nan
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref, cur, eps=1e-4):
    ref = pd.Series(ref).fillna("Unknown").astype(str)
    cur = pd.Series(cur).fillna("Unknown").astype(str)
    if len(ref) == 0 or len(cur) == 0:
        return np.nan
    cats = sorted(set(ref) | set(cur))
    r = ref.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    c = cur.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def feature_psi(ref_df, cur_df, num_cols, cat_cols):
    out = {}
    for c in num_cols:
        out[c] = psi_numeric(ref_df[c], cur_df[c])
    for c in cat_cols:
        out[c] = psi_categorical(ref_df[c], cur_df[c])
    return out


def psi_status(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if value < PSI_WARN:
        return "stable"
    if value < PSI_ALERT:
        return "warning"
    return "unstable"
