"""Train candidate models for one snapshot date, select the best and store it in the model bank.

python scripts/model_train.py --snapshotdate 2024-09-01

Leakage controls
- only labels already observed BEFORE the snapshot date are used (label snapshot_date < snapshot date)
- features are taken at the loan application date (loan_start_date), never at the label date
- imputation / scaling / encoding are fitted on the training split only
- the out-of-time (OOT) set is the most recent labelled months and is never used for fitting or selection
"""
import argparse
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import ml_utils as mu

try:
    import xgboost as xgb
except ImportError:  # xgboost is in requirements.txt; this only keeps the script importable elsewhere
    xgb = None

TRAIN_TEST_MONTHS = 12      # labelled history used for train + test
OOT_MONTHS = 2              # most recent labelled months held out as out-of-time test
TRAIN_TEST_RATIO = 0.8
MIN_TRAIN_ROWS = 1500       # skip training until enough labels have matured
MIN_OOT_ROWS = 300


def build_preprocessor(num_cols, cat_cols):
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])


def make_dataset(labels, features):
    """Join labels to application-time features (NOT to features at the label date)."""
    labels = labels.copy()
    features = features.copy()
    labels["label_snapshot_date"] = pd.to_datetime(labels.pop("snapshot_date"))
    labels["loan_start_date"] = pd.to_datetime(labels["loan_start_date"])
    features["snapshot_date"] = pd.to_datetime(features["snapshot_date"])
    feature_cols = [c for c in features.columns if c not in mu.NON_FEATURE_COLS]
    data = labels[["Customer_ID", "loan_start_date", "label", "label_snapshot_date"]].merge(
        features.drop(columns=["loan_id"], errors="ignore"),
        left_on=["Customer_ID", "loan_start_date"], right_on=["Customer_ID", "snapshot_date"], how="inner")
    return data, feature_cols


def split_windows(data, train_date):
    oot_end = train_date - relativedelta(days=1)
    oot_start = train_date - relativedelta(months=OOT_MONTHS)
    tt_end = oot_start - relativedelta(days=1)
    tt_start = oot_start - relativedelta(months=TRAIN_TEST_MONTHS)
    d = data["label_snapshot_date"]
    oot = data[(d >= oot_start) & (d <= oot_end)]
    tt = data[(d >= tt_start) & (d <= tt_end)]
    cfg = {"train_date": train_date.strftime("%Y-%m-%d"), "oot_start": oot_start.strftime("%Y-%m-%d"),
           "oot_end": oot_end.strftime("%Y-%m-%d"), "train_test_start": tt_start.strftime("%Y-%m-%d"),
           "train_test_end": tt_end.strftime("%Y-%m-%d")}
    return tt, oot, cfg


def fit_and_select(tt, oot, feature_cols, seed=88, n_iter=10):
    """Fit the candidate models, pick the best on the test split, report OOT. Returns an artefact dict."""
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(tt[c])]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    tt = tt.copy()
    oot = oot.copy()
    tt[feature_cols] = mu.coerce_types(tt[feature_cols], num_cols, cat_cols)
    oot[feature_cols] = mu.coerce_types(oot[feature_cols], num_cols, cat_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        tt[feature_cols], tt["label"], test_size=1 - TRAIN_TEST_RATIO,
        random_state=seed, shuffle=True, stratify=tt["label"])
    X_oot, y_oot = oot[feature_cols], oot["label"]

    pre = build_preprocessor(num_cols, cat_cols).fit(X_train)          # fitted on TRAIN only
    Xtr, Xte, Xoot = pre.transform(X_train), pre.transform(X_test), pre.transform(X_oot)

    candidates = {}
    lr = LogisticRegression(max_iter=2000).fit(Xtr, y_train)
    candidates["logistic_regression"] = (lr, {})
    if xgb is not None:
        param_dist = {"n_estimators": [25, 50, 100], "max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1],
                      "subsample": [0.6, 0.8], "colsample_bytree": [0.6, 0.8], "min_child_weight": [1, 3, 5],
                      "reg_lambda": [1, 2]}
        search = RandomizedSearchCV(xgb.XGBClassifier(eval_metric="logloss", random_state=seed),
                                    param_distributions=param_dist, scoring="roc_auc", n_iter=n_iter,
                                    cv=3, random_state=42, n_jobs=2)
        search.fit(Xtr, y_train)
        candidates["xgboost"] = (search.best_estimator_, search.best_params_)

    results = {}
    for name, (model, _) in candidates.items():
        results[name] = {
            "auc_train": roc_auc_score(y_train, model.predict_proba(Xtr)[:, 1]),
            "auc_test": roc_auc_score(y_test, model.predict_proba(Xte)[:, 1]),
            "auc_oot": roc_auc_score(y_oot, model.predict_proba(Xoot)[:, 1]),
        }
        for k in ("train", "test", "oot"):
            results[name]["gini_" + k] = round(2 * results[name]["auc_" + k] - 1, 4)

    best = max(results, key=lambda n: results[n]["auc_test"])          # selection uses TEST, never OOT
    model, hp = candidates[best]
    return {
        "model": model, "model_type": best, "hp_params": hp,
        "preprocessing_transformers": {"preprocessor": pre},
        "feature_cols": feature_cols, "num_cols": num_cols, "cat_cols": cat_cols,
        "results": results, "selected_results": results[best],
        "passed_gate": bool(results[best]["auc_oot"] >= mu.MIN_OOT_AUC),
        "data_stats": {"n_train": len(X_train), "n_test": len(X_test), "n_oot": len(X_oot),
                       "bad_rate_train": float(y_train.mean()), "bad_rate_test": float(y_test.mean()),
                       "bad_rate_oot": float(y_oot.mean())},
        # reference distributions for monitoring (test split = data the model was NOT fitted on)
        "reference_scores": model.predict_proba(Xte)[:, 1],
        "reference_features": X_test.reset_index(drop=True),
    }


def update_registry(artefact):
    path = mu.MODEL_BANK + "model_registry.csv"
    row = {"model_version": artefact["model_version"], "model_type": artefact["model_type"],
           "train_date": artefact["data_dates"]["train_date"], "passed_gate": artefact["passed_gate"],
           "auc_train": artefact["selected_results"]["auc_train"], "auc_test": artefact["selected_results"]["auc_test"],
           "auc_oot": artefact["selected_results"]["auc_oot"], "gini_oot": artefact["selected_results"]["gini_oot"],
           "n_train": artefact["data_stats"]["n_train"], "n_oot": artefact["data_stats"]["n_oot"]}
    reg = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    if len(reg):
        reg = reg[reg["model_version"] != row["model_version"]]
    pd.concat([reg, pd.DataFrame([row])], ignore_index=True).sort_values("train_date").to_csv(path, index=False)


def main(snapshotdate):
    spark = mu.get_spark("model_train")
    train_date = datetime.strptime(snapshotdate, "%Y-%m-%d")
    labels = mu.read_gold_folder(spark, mu.GOLD_LABELS)
    features = mu.read_gold_folder(spark, mu.GOLD_FEATURES)
    spark.stop()
    if labels is None or features is None or len(labels) == 0:
        print("no gold data yet - skipping training")
        return
    data, feature_cols = make_dataset(labels, features)
    tt, oot, cfg = split_windows(data, train_date)
    print("train/test rows:", len(tt), "| oot rows:", len(oot), cfg)
    if len(tt) < MIN_TRAIN_ROWS or len(oot) < MIN_OOT_ROWS or tt["label"].nunique() < 2 or oot["label"].nunique() < 2:
        print("not enough matured labels yet - skipping training for", snapshotdate)
        return

    artefact = fit_and_select(tt, oot, feature_cols)
    artefact["model_version"] = "credit_model_" + mu.suffix(snapshotdate)
    artefact["data_dates"] = cfg
    os.makedirs(mu.MODEL_BANK, exist_ok=True)
    with open(mu.MODEL_BANK + artefact["model_version"] + ".pkl", "wb") as f:
        pickle.dump(artefact, f)
    update_registry(artefact)
    print("saved", artefact["model_version"], artefact["model_type"], artefact["selected_results"],
          "passed_gate:", artefact["passed_gate"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=True)
    main(p.parse_args().snapshotdate)
