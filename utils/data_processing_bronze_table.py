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


def process_bronze_table(snapshot_date_str, csv_file_path, bronze_directory, prefix, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True) \
              .filter(col("snapshot_date") == snapshot_date)
    print(prefix, snapshot_date_str, "row count:", df.count())

    # save bronze table to datamart
    partition_name = "bronze_" + prefix + "_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = bronze_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print("saved to:", filepath)

    return df
