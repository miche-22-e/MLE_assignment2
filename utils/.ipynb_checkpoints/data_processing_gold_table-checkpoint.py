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


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save (loan_start_date kept so labels can be joined to application-time features)
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "loan_start_date", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df


def process_features_gold_table(snapshot_date_str, silver_attributes_directory, silver_financials_directory,
                                silver_clickstream_directory, gold_feature_store_directory, spark,
                                clickstream_max_date="2024-12-01"):
    suffix = snapshot_date_str.replace('-', '_')

    # clickstream: latest snapshot on or before the application date (ISO date strings compare correctly)
    click_date_str = min(snapshot_date_str, clickstream_max_date)
    click_suffix = click_date_str.replace('-', '_')

    fin = spark.read.parquet(silver_financials_directory + "silver_financials_" + suffix + ".parquet")
    att = spark.read.parquet(silver_attributes_directory + "silver_attributes_" + suffix + ".parquet").drop("snapshot_date")
    click = spark.read.parquet(silver_clickstream_directory + "silver_clickstream_" + click_suffix + ".parquet") \
                 .drop("snapshot_date") \
                 .withColumn("has_clickstream", F.lit(1))

    df = fin.join(att, on="Customer_ID", how="left").join(click, on="Customer_ID", how="left")
    df = df.withColumn("has_clickstream", F.coalesce(col("has_clickstream"), F.lit(0)))

    # engineered ratio features (application-time data only)
    df = df.withColumn("debt_to_income", F.when(col("Annual_Income") > 0, col("Outstanding_Debt") / col("Annual_Income"))) \
           .withColumn("emi_to_salary", F.when(col("Monthly_Inhand_Salary") > 0, col("Total_EMI_per_month") / col("Monthly_Inhand_Salary"))) \
           .withColumn("balance_to_salary", F.when(col("Monthly_Inhand_Salary") > 0, col("Monthly_Balance") / col("Monthly_Inhand_Salary"))) \
           .withColumn("invested_to_salary", F.when(col("Monthly_Inhand_Salary") > 0, col("Amount_invested_monthly") / col("Monthly_Inhand_Salary")))

    # missing categoricals -> fixed "Unknown" label (no dataset statistics used)
    df = df.fillna("Unknown", subset=["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"])

    # loan_id in the same format as the loan table (one loan per customer)
    df = df.withColumn("loan_id", F.concat_ws("_", col("Customer_ID"), F.date_format(col("snapshot_date"), "yyyy_MM_dd")))

    df = df.drop("Type_of_Loan")

    outpath = gold_feature_store_directory + "gold_feature_store_" + suffix + ".parquet"
    df.write.mode("overwrite").parquet(outpath)
    print('saved to:', outpath, 'row count:', df.count())
    return df
