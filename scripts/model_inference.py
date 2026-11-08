"""Score the loans applied for in one snapshot month with the latest approved model.

python scripts/model_inference.py --snapshotdate 2024-09-01
Output: datamart/gold/model_predictions/predictions_YYYY_MM_DD.parquet
"""
import argparse
import glob
import os
import pickle
from datetime import datetime

from utils import ml_utils as mu


def pick_model(snapshotdate):
    """Latest model trained on or before the snapshot date that passed the promotion gate."""
    target = datetime.strptime(snapshotdate, "%Y-%m-%d")
    found = []
    for f in glob.glob(mu.MODEL_BANK + "credit_model_*.pkl"):
        d = datetime.strptime(os.path.basename(f)[len("credit_model_"):-4], "%Y_%m_%d")
        if d <= target:
            found.append((d, f))
    for _, f in sorted(found, reverse=True):
        with open(f, "rb") as fh:
            art = pickle.load(fh)
        if art["passed_gate"]:
            return art
    return None


def score(artefact, feats):
    X = mu.coerce_types(feats[artefact["feature_cols"]], artefact["num_cols"], artefact["cat_cols"])
    Xp = artefact["preprocessing_transformers"]["preprocessor"].transform(X)
    return artefact["model"].predict_proba(Xp)[:, 1]


def main(snapshotdate):
    path = mu.GOLD_FEATURES + "gold_feature_store_" + mu.suffix(snapshotdate) + ".parquet"
    art = pick_model(snapshotdate)
    if art is None or not os.path.exists(path):
        print("no approved model or no features for", snapshotdate, "- skipping inference")
        return
    spark = mu.get_spark("model_inference")
    feats = spark.read.parquet(path).toPandas()
    if len(feats) == 0:
        print("no loan applications in", snapshotdate)
        spark.stop()
        return
    out = feats[["Customer_ID", "loan_id", "snapshot_date"]].copy()
    out["model_version"] = art["model_version"]
    out["model_score"] = score(art, feats).astype(float)
    os.makedirs(mu.GOLD_PREDICTIONS, exist_ok=True)
    outpath = mu.GOLD_PREDICTIONS + "predictions_" + mu.suffix(snapshotdate) + ".parquet"
    spark.createDataFrame(out).write.mode("overwrite").parquet(outpath)
    print("scored", len(out), "loans with", art["model_version"], "->", outpath)
    spark.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=True)
    main(p.parse_args().snapshotdate)
