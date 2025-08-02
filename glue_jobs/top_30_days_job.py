import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.sql.functions import col, lag, when
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
# This schema is defined to ensure the correct data types are used
# for the calculations, especially 'close' as a DoubleType.
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
    # Read the data from the specified S3 path using the defined schema
    df = spark.read.csv(
        input_path,
        header=True,
        schema=schema
    )
except Exception as e:
    print(f"An error occurred while reading the data: {e}")
    sys.exit(1)

# --- 5. Find the top three 30-day returns ---
# Cast the Date column to DateType for proper ordering
df_with_date = df.withColumn("Date", col("Date").cast(DateType()))

# Define the window to partition by 'ticker' and order by 'Date'
window_spec = Window.partitionBy("ticker").orderBy("Date")

# Use lag to get the close price from 30 trading days prior
df_with_lag = df_with_date.withColumn("close_30d_ago", lag("close", 30).over(window_spec))

# Compute the 30-day return: (current_close - close_30d_ago) / close_30d_ago
df_with_return = df_with_lag.withColumn(
    "30_day_return",
    when(col("close_30d_ago").isNotNull(), (col("close") - col("close_30d_ago")) / col("close_30d_ago"))
)

# Filter for valid returns, sort in descending order, and take the top 3
top_returns = df_with_return.select("ticker", "Date", "30_day_return") \
    .where(col("30_day_return").isNotNull()) \
    .orderBy(col("30_day_return").desc()) \
    .limit(3)

# --- 6. Write Result to S3 as Parquet ---
print(f"Writing results to: {output_path}")
top_returns.write.mode("overwrite").parquet(output_path)