import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.sql.functions import col, lag, stddev, sqrt, when, lit
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from awsglue.job import Job

# --- 1. Get Required Glue Job Arguments ---
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'input_path', 'output_path'])
input_path = args['input_path']
output_path = args['output_path']

# --- 2. Initialize Glue and Spark Contexts ---
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# --- 3. Define Schema for CSV ---
# This schema ensures the correct data types are inferred for calculations.
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

# --- 5. Find the stock with the highest annualized volatility ---
# Cast the Date column to DateType for proper ordering
df_with_date = df.withColumn("Date", col("Date").cast(DateType()))

# Define the window to partition by 'ticker' and order by 'Date'
window_spec = Window.partitionBy("ticker").orderBy("Date")

# Use lag to get the previous day's close price for each stock
df_with_lag = df_with_date.withColumn("previous_close", lag("close").over(window_spec))

# Compute the daily return: (current_close - previous_close) / previous_close
df_with_return = df_with_lag.withColumn(
    "daily_return",
    when(col("previous_close").isNotNull(), (col("close") - col("previous_close")) / col("previous_close"))
)

# Group by ticker and compute the annualized volatility
# Annualized volatility = standard deviation of daily returns * sqrt(252 trading days)
volatility_df = df_with_return.groupBy("ticker").agg(
    (stddev("daily_return") * sqrt(lit(252))).alias("annualized_volatility")
)

# Get the most volatile stock by ordering in descending order and taking the top result
most_volatile_df = volatility_df.orderBy(col("annualized_volatility").desc()).limit(1)

# --- 6. Write Result to S3 as Parquet ---
print(f"Writing results to: {output_path}")
most_volatile_df.write.mode("overwrite").parquet(output_path)
