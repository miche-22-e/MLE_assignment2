import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

# For feature-table silver processing

def _clean_numeric(df, columns, dtype="double"):
    """Strip junk characters (e.g. '52312.68_') and cast to a number; unparseable -> null."""
    for c in columns:
        stripped = F.regexp_replace(col(c).cast("string"), r"[^0-9.\-]", "")
        df = df.withColumn(c, F.when(stripped == "", F.lit(None)).otherwise(stripped).cast(dtype))
    return df


def _null_out_of_range(df, bounds):
    """Set values outside plausible business ranges to null (fixed rules, no dataset statistics)."""
    for c, (lo, hi) in bounds.items():
        df = df.withColumn(c, F.when((col(c) >= lo) & (col(c) <= hi), col(c)))
    return df


def _null_junk_values(df, junk_map):
    """Replace placeholder junk values in categorical columns with null."""
    for c, junk in junk_map.items():
        df = df.withColumn(c, F.when(col(c).isin(junk), F.lit(None)).otherwise(col(c)))
    return df


# silver: attributes

def process_silver_attributes(snapshot_date_str, bronze_dir, silver_dir, spark):
    filepath = bronze_dir + "bronze_attributes_" + snapshot_date_str.replace('-', '_') + '.csv'
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # drop personal identifiers (no predictive value, privacy risk)
    df = df.drop("Name", "SSN")

    # clean age: strip junk, cast, null out implausible values (thresholds are assumptions - adjust after EDA)
    df = _clean_numeric(df, ["Age"], "int")
    df = _null_out_of_range(df, {"Age": (18, 100)})

    # clean categorical placeholder
    df = df.withColumn("Occupation", F.when(col("Occupation").rlike("^_+$"), F.lit(None)).otherwise(col("Occupation")))

    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType())) \
           .withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    outpath = silver_dir + "silver_attributes_" + snapshot_date_str.replace('-', '_') + '.parquet'
    df.write.mode("overwrite").parquet(outpath)
    print('saved to:', outpath)
    return df


# silver: financials

def process_silver_financials(snapshot_date_str, bronze_dir, silver_dir, spark):
    filepath = bronze_dir + "bronze_financials_" + snapshot_date_str.replace('-', '_') + '.csv'
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    numeric_cols = [
        "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
        "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt",
        "Credit_Utilization_Ratio", "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
    ]
    df = _clean_numeric(df, numeric_cols, "double")

    # null out implausible counts / rates (thresholds are assumptions - adjust after EDA)
    df = _null_out_of_range(df, {
        "Num_Bank_Accounts": (0, 20),
        "Num_Credit_Card": (0, 20),
        "Interest_Rate": (0, 100),
        "Num_of_Loan": (0, 20),
        "Num_of_Delayed_Payment": (0, 50),
        "Num_Credit_Inquiries": (0, 50),
    })

    # count columns -> integer
    for c in ["Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
              "Delay_from_due_date", "Num_of_Delayed_Payment", "Num_Credit_Inquiries"]:
        df = df.withColumn(c, col(c).cast(IntegerType()))

    # categorical placeholders -> null
    df = _null_junk_values(df, {
        "Credit_Mix": ["_"],
        "Payment_of_Min_Amount": ["NM"],
        "Payment_Behaviour": ["!@9#%8"],
    })

    # "10 Years and 9 Months" -> total months
    years = F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Years", 1)
    months = F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Months", 1)
    df = df.withColumn("Credit_History_Months", years.cast("int") * 12 + months.cast("int")).drop("Credit_History_Age")

    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType())) \
           .withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    outpath = silver_dir + "silver_financials_" + snapshot_date_str.replace('-', '_') + '.parquet'
    df.write.mode("overwrite").parquet(outpath)
    print('saved to:', outpath)
    return df


# silver: clickstream

def process_silver_clickstream(snapshot_date_str, bronze_dir, silver_dir, spark):
    filepath = bronze_dir + "bronze_clickstream_" + snapshot_date_str.replace('-', '_') + '.csv'
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    for i in range(1, 21):
        df = df.withColumn(f"fe_{i}", col(f"fe_{i}").cast(IntegerType()))

    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType())) \
           .withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    outpath = silver_dir + "silver_clickstream_" + snapshot_date_str.replace('-', '_') + '.parquet'
    df.write.mode("overwrite").parquet(outpath)
    print('saved to:', outpath)
    return df