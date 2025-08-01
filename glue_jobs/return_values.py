import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.sql.functions import col, avg, round, when
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from pyspark.sql import SparkSession

# This script is designed to be run as an AWS Glue job.
# It computes the average daily return for all stocks from a specified
# CSV file in S3 and writes the results to another S3 location.

# --- 1. Get Glue Job Arguments with Default Values ---
# We define a list of expected parameters, including the ones with default values.
# The getResolvedOptions function will still raise an error if a parameter
# is missing and is not handled. A better approach is to provide a fallback
# value using the dictionary's get() method, as shown below.
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'input_path', 'output_path'])

# Define default paths for input and output.
# These will be used if the parameters are not provided in the job configuration.
default_input_path = "s3://your-default-bucket/input-data/"
default_output_path = "s3://your-default-bucket/output-data/"

# Use the .get() method to safely retrieve arguments with a fallback to the default values.
input_path = args.get('input_path', default_input_path)
output_path = args.get('output_path', default_output_path)

# --- 2. Initialize Glue and Spark Contexts ---
# In a Glue job, you don't create a new SparkSession.
# Instead, you get the existing SparkContext and GlueContext.
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# --- 3. Define Schema for Robustness ---
# Explicitly defining the schema is a best practice for performance and
# ensuring data types are correctly interpreted by Spark.
schema = StructType([
    StructField("Date", StringType(), True),
    StructField("open", DoubleType(), True),
    StructField("high", DoubleType(), True),
    StructField("low", DoubleType(), True),
    StructField("close", DoubleType(), True),
    StructField("volume", DoubleType(), True),
    StructField("ticker", StringType(), True)
])

# --- 4. Read Data from S3 ---
# Use the input_path variable (which now has a default) to read the data.
try:
    print(f"Reading data from: {input_path}")
    df = spark.read.csv(
        input_path,
        header=True,
        schema=schema
    )
except Exception as e:
    print(f"An error occurred while reading the data: {e}")
    sys.exit(1) # Exit the job with an error

# --- 5. PySpark Transformation Logic ---
# The core logic remains largely the same, but it's executed on the
# distributed data read from S3.
avg_daily_return_df = df \
    .withColumn("Date", col("Date").cast(DateType())) \
    .withColumn(
        "daily_return",
        when(col("open") != 0, (col("close") - col("open")) / col("open")).otherwise(None)
    ) \
    .groupBy("Date") \
    .agg(round(avg("daily_return"), 6).alias("average_return")) \
    .orderBy("Date")

# --- 6. Write Data to S3 ---
# Write the final DataFrame to the S3 location specified by the output_path variable.
# Parquet is a highly recommended format for its columnar storage and compression.
print(f"Writing results to: {output_path}")
avg_daily_return_df.write.mode("overwrite").parquet(output_path)