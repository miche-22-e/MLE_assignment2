"""Runs one layer of the Assignment 1 medallion pipeline for ONE snapshot month.

python scripts/data_pipeline.py --snapshotdate 2024-03-01 --layer bronze|silver|gold_labels|gold_features
"""
import argparse
import os
from datetime import datetime

import utils.data_processing_bronze_table as bronze
import utils.data_processing_silver_table as silver
import utils.data_processing_gold_table as gold
from utils.ml_utils import get_spark

# each source only has data up to its own last snapshot date
SOURCES = {
    "loan_daily":  {"csv": "data/lms_loan_daily.csv",      "dir": "datamart/bronze/lms/",         "end": "2025-11-01"},
    "attributes":  {"csv": "data/features_attributes.csv", "dir": "datamart/bronze/attributes/",  "end": "2025-01-01"},
    "financials":  {"csv": "data/features_financials.csv", "dir": "datamart/bronze/financials/",  "end": "2025-01-01"},
    "clickstream": {"csv": "data/feature_clickstream.csv", "dir": "datamart/bronze/clickstream/", "end": "2024-12-01"},
}
SILVER_LOAN_DIR = "datamart/silver/loan_daily/"
GOLD_LABEL_DIR = "datamart/gold/label_store/"
GOLD_FEATURE_DIR = "datamart/gold/feature_store/"


def available(prefix, date_str):
    return datetime.strptime(date_str, "%Y-%m-%d") <= datetime.strptime(SOURCES[prefix]["end"], "%Y-%m-%d")


def main(date_str, layer):
    spark = get_spark("data_pipeline")
    if layer == "bronze":
        for prefix, cfg in SOURCES.items():
            if available(prefix, date_str):
                os.makedirs(cfg["dir"], exist_ok=True)
                bronze.process_bronze_table(date_str, cfg["csv"], cfg["dir"], prefix, spark)
    elif layer == "silver":
        os.makedirs(SILVER_LOAN_DIR, exist_ok=True)
        silver.process_silver_table(date_str, SOURCES["loan_daily"]["dir"], SILVER_LOAN_DIR, spark)
        funcs = {"attributes": silver.process_silver_attributes,
                 "financials": silver.process_silver_financials,
                 "clickstream": silver.process_silver_clickstream}
        for prefix, func in funcs.items():
            if available(prefix, date_str):
                out_dir = f"datamart/silver/{prefix}/"
                os.makedirs(out_dir, exist_ok=True)
                func(date_str, SOURCES[prefix]["dir"], out_dir, spark)
    elif layer == "gold_labels":
        os.makedirs(GOLD_LABEL_DIR, exist_ok=True)
        gold.process_labels_gold_table(date_str, SILVER_LOAN_DIR, GOLD_LABEL_DIR, spark, dpd=30, mob=6)
    elif layer == "gold_features":
        if available("attributes", date_str):
            os.makedirs(GOLD_FEATURE_DIR, exist_ok=True)
            gold.process_features_gold_table(date_str, "datamart/silver/attributes/", "datamart/silver/financials/",
                                             "datamart/silver/clickstream/", GOLD_FEATURE_DIR, spark)
    else:
        raise ValueError("unknown layer " + layer)
    spark.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=True)
    p.add_argument("--layer", required=True)
    a = p.parse_args()
    main(a.snapshotdate, a.layer)
