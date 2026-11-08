from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def job(task_id, script, extra=""):
    # {{ ds }} is the run's logical date (first of the month); PYTHONPATH lets the scripts import utils/
    return BashOperator(
        task_id=task_id,
        bash_command=(f'cd /opt/airflow && PYTHONPATH=/opt/airflow '
                      f'python3 scripts/{script} --snapshotdate "{{{{ ds }}}}" {extra}'),
    )


with DAG(
    "ml_pipeline",
    default_args=default_args,
    description="Monthly loan-default pipeline: data (medallion) -> train -> inference -> monitoring",
    schedule_interval="0 0 1 * *",          # 00:00 on the 1st of every month
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2025, 1, 1),          # last month with application-time features
    catchup=True,                           # backfills every month from start_date
    max_active_runs=1,                      # months run in order: later months need earlier labels / models
    tags=["cs611", "ml-pipeline"],
) as dag:

    start = EmptyOperator(task_id="start")

    # --- data pipeline (medallion) ---
    bronze = job("bronze_all_sources", "data_pipeline.py", "--layer bronze")
    silver = job("silver_all_sources", "data_pipeline.py", "--layer silver")
    gold_labels = job("gold_label_store", "data_pipeline.py", "--layer gold_labels")
    gold_features = job("gold_feature_store", "data_pipeline.py", "--layer gold_features")
    data_completed = EmptyOperator(task_id="data_completed")

    # --- ML ---
    train = job("model_train", "model_train.py")
    infer = job("model_inference", "model_inference.py")
    monitor = job("model_monitor", "model_monitor.py")
    plot = job("plot_monitoring", "plot_monitoring.py")

    end = EmptyOperator(task_id="end")

    start >> bronze >> silver >> [gold_labels, gold_features] >> data_completed
    data_completed >> train >> infer >> monitor >> plot >> end
