"""Monitor model stability and performance for one snapshot month.

python scripts/model_monitor.py --snapshotdate 2024-09-01
Output: datamart/gold/model_monitoring/monitoring_YYYY_MM_DD.parquet (one row per month)

- Stability (month T): PSI of the score and of every feature versus the reference data kept in the model artefact
- Performance: labels only mature 6 months after application, so at month T we evaluate the cohort that
  applied in month T-6 (its labels arrive in the label store at snapshot T)
"""
import argparse
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import roc_auc_score

from utils import ml_utils as mu

MIN_EVAL_ROWS = 100


def evaluate_cohort(preds, labels):
    """AUC / Gini / bad rate of a scored cohort against its matured labels."""
    m = preds.merge(labels[["Customer_ID", "label"]], on="Customer_ID", how="inner")
    out = {"n_labelled": int(len(m)), "bad_rate": np.nan, "auc": np.nan, "gini": np.nan}
    if len(m):
        out["bad_rate"] = float(m["label"].mean())
    if len(m) >= MIN_EVAL_ROWS and m["label"].nunique() == 2:
        out["auc"] = float(roc_auc_score(m["label"], m["model_score"]))
        out["gini"] = 2 * out["auc"] - 1
    return out


def stability(art, preds, feats):
    score_psi = mu.psi_numeric(art["reference_scores"], preds["model_score"])
    f = mu.coerce_types(feats[art["feature_cols"]], art["num_cols"], art["cat_cols"])
    per_feature = mu.feature_psi(art["reference_features"], f, art["num_cols"], art["cat_cols"])
    series = pd.Series(per_feature).dropna()
    return score_psi, float(series.mean()), float(series.max()), str(series.idxmax())


def main(snapshotdate):
    sfx = mu.suffix(snapshotdate)
    pred_path = mu.GOLD_PREDICTIONS + "predictions_" + sfx + ".parquet"
    if not os.path.exists(pred_path):
        print("no predictions for", snapshotdate, "- skipping monitoring")
        return
    spark = mu.get_spark("model_monitor")
    preds = spark.read.parquet(pred_path).toPandas()
    if len(preds) == 0:
        spark.stop()
        return
    version = preds["model_version"].iloc[0]
    with open(mu.MODEL_BANK + version + ".pkl", "rb") as f:
        art = pickle.load(f)
    feats = spark.read.parquet(mu.GOLD_FEATURES + "gold_feature_store_" + sfx + ".parquet").toPandas()
    score_psi, feat_mean, feat_max, feat_max_name = stability(art, preds, feats)

    # performance of the cohort that applied 6 months ago
    snap = datetime.strptime(snapshotdate, "%Y-%m-%d")
    cohort = (snap - relativedelta(months=6)).strftime("%Y-%m-%d")
    cohort_path = mu.GOLD_PREDICTIONS + "predictions_" + mu.suffix(cohort) + ".parquet"
    label_path = mu.GOLD_LABELS + "gold_label_store_" + sfx + ".parquet"
    perf = {"n_labelled": 0, "bad_rate": np.nan, "auc": np.nan, "gini": np.nan}
    perf_version = ""
    if os.path.exists(cohort_path) and os.path.exists(label_path):
        cohort_preds = spark.read.parquet(cohort_path).toPandas()
        labels = spark.read.parquet(label_path).toPandas()
        if len(cohort_preds) and len(labels):
            perf = evaluate_cohort(cohort_preds, labels)
            perf_version = cohort_preds["model_version"].iloc[0]

    psi_state = mu.psi_status(score_psi)
    perf_state = "n/a" if np.isnan(perf["auc"]) else ("ok" if perf["auc"] >= mu.MIN_OOT_AUC else "degraded")
    action = "retrain / investigate" if (psi_state == "unstable" or perf_state == "degraded") else \
             ("watch" if psi_state == "warning" else "none")

    row = pd.DataFrame([{
        "snapshot_date": snapshotdate, "model_version": version, "n_scored": int(len(preds)),
        "mean_score": float(preds["model_score"].mean()), "score_psi": float(score_psi),
        "feature_psi_mean": feat_mean, "feature_psi_max": feat_max, "feature_psi_max_name": feat_max_name,
        "psi_status": psi_state, "perf_cohort_date": cohort if perf_version else "",
        "perf_model_version": perf_version, "n_labelled": perf["n_labelled"], "bad_rate": perf["bad_rate"],
        "auc": perf["auc"], "gini": perf["gini"], "perf_status": perf_state,
        "reference_auc_oot": float(art["selected_results"]["auc_oot"]), "recommended_action": action}])
    os.makedirs(mu.GOLD_MONITORING, exist_ok=True)
    outpath = mu.GOLD_MONITORING + "monitoring_" + sfx + ".parquet"
    spark.createDataFrame(row).write.mode("overwrite").parquet(outpath)
    print(row.T)
    spark.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=True)
    main(p.parse_args().snapshotdate)
