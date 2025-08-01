import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.sql.functions import col, avg, round, lag, when
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType
from awsglue.context import GlueContext
from pyspark.context import SparkContext

# ----------------------------------------------------------------------------------------
# ASSUMPTIONS:
# 1. Use ONLY the 'close' price to compute returns.
# 2. If a closing price is missing for a date, use the closest available previous date.
# 3. Return is computed as % difference between two closing prices:
#       (current_close - previous_close) / previous_close
# ----------------------------------------------------------------------------------------

# --- 1. Get Required Glue Job Arguments ---
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'input_path', 'output_path'])
input_path = args['input_path']
output_path = args['output_path']

# --- 2. Initialize Glue and Spark Contexts ---
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# --- 3. Define Schema for CSV ---
schema = StructType([
    StructField("Date", StringType(), True),
    StructField("open", DoubleType(), True),
    StructField("high", DoubleType(), True),
    StructField("low", DoubleType(), True),
    StructField("close", DoubleType(), True),
    StructField("volume", DoubleType(), True),
    StructField("ticker", StringType(), True)
])

# --- 4. Read Input from S3 ---
try:
    print(f"Reading data from: {input_path}")
    df = spark.read.csv(
        input_path,
        header=True,
        schema=schema
    )
except Exception as e:
    print(f"An error occurred while reading the data: {e}")
    sys.exit(1)

# --- 5. Compute Daily Return Using Closing Price ---
df = df.withColumn("Date", col("Date").cast(DateType()))

# Define the correct window to partition by 'ticker' and order by 'Date'.
# This is crucial for calculating the return per stock correctly.
window_spec = Window.partitionBy("ticker").orderBy("Date")

# Use lag to get previous close price within each ticker's partition
df_with_lag = df.withColumn("prev_close", lag("close").over(window_spec))

# Compute daily return: (close - prev_close) / prev_close
returns_df = df_with_lag.withColumn(
    "daily_return",
    when(col("prev_close").isNotNull(), (col("close") - col("prev_close")) / col("prev_close"))
)

# Average daily return per date across all stocks
avg_daily_return_df = returns_df.groupBy("Date").agg(
    round(avg("daily_return"), 6).alias("average_return")
).orderBy("Date")

# --- 6. Write Result to S3 as Parquet ---
print(f"Writing results to: {output_path}")
avg_daily_return_df.write.mode("overwrite").parquet(output_path)