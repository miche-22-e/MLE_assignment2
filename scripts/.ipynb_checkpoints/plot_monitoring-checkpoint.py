"""Draw the monitoring dashboard from the monitoring gold table.

python scripts/plot_monitoring.py --snapshotdate 2025-01-01   (the date is only used to make the task re-runnable)
Output: datamart/gold/monitoring_plots/model_monitoring.png and monitoring_summary.csv
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import ml_utils as mu


def draw(df):
    df = df.sort_values("snapshot_date").copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))

    a = ax[0, 0]
    a.plot(df["snapshot_date"], df["score_psi"], marker="o", color="#12263F", label="score PSI")
    a.plot(df["snapshot_date"], df["feature_psi_max"], marker="s", color="#B9713A", label="max feature PSI")
    a.axhline(mu.PSI_WARN, color="#E0B03A", ls="--", label="warning 0.10")
    a.axhline(mu.PSI_ALERT, color="#C0392B", ls="--", label="alert 0.25")
    a.set_title("Stability: PSI vs training reference"); a.legend(fontsize=8); a.set_ylabel("PSI")

    perf = df[df["auc"].notna()].copy()
    a = ax[0, 1]
    if len(perf):
        perf["cohort"] = pd.to_datetime(perf["perf_cohort_date"])
        a.plot(perf["cohort"], perf["auc"], marker="o", color="#12263F", label="AUC (monthly cohort)")
        a.plot(perf["cohort"], perf["reference_auc_oot"], ls=":", color="#B9713A", label="AUC at training (OOT)")
        a.axhline(mu.MIN_OOT_AUC, color="#C0392B", ls="--", label="minimum 0.65")
        a.legend(fontsize=8)
    a.set_title("Performance by application cohort"); a.set_ylabel("AUC")

    a = ax[1, 0]
    a.plot(df["snapshot_date"], df["mean_score"], marker="o", color="#1FA5A0", label="mean predicted score")
    if len(perf):
        a.plot(pd.to_datetime(perf["perf_cohort_date"]), perf["bad_rate"], marker="s", color="#B9713A",
               label="observed bad rate (cohort)")
    a.set_title("Predicted score vs observed bad rate"); a.legend(fontsize=8)

    a = ax[1, 1]
    a.bar(df["snapshot_date"], df["n_scored"], width=20, color="#B4BEC9")
    a.set_title("Loans scored per month"); a.set_ylabel("count")

    for x in ax.flat:
        x.tick_params(axis="x", rotation=30); x.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    spark = mu.get_spark("plot_monitoring")
    df = mu.read_gold_folder(spark, mu.GOLD_MONITORING)
    spark.stop()
    if df is None or len(df) == 0:
        print("no monitoring results yet")
        return
    os.makedirs(mu.GOLD_PLOTS, exist_ok=True)
    df.sort_values("snapshot_date").to_csv(mu.GOLD_PLOTS + "monitoring_summary.csv", index=False)
    draw(df).savefig(mu.GOLD_PLOTS + "model_monitoring.png", dpi=150)
    print("saved", mu.GOLD_PLOTS + "model_monitoring.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=False)
    p.parse_args()
    main()
