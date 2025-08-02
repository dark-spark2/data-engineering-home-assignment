import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.sql.functions import col, avg
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

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

# --- 5. Find the stock with the highest average traded worth ---
# Calculate the total worth of shares traded for each day (close price * volume)
df_with_worth = df.withColumn("traded_worth", col("close") * col("volume"))

# Group by ticker and calculate the average traded worth
avg_worth_df = df_with_worth.groupBy("ticker").agg(
    avg("traded_worth").alias("avg_traded_worth")
)

# Find the ticker with the highest average traded worth
top_stock_df = avg_worth_df.orderBy(col("avg_traded_worth").desc()).limit(1)

# --- 6. Write Result to S3 as Parquet ---
print(f"Writing results to: {output_path}")
# Write the final DataFrame to the specified S3 path in Parquet format
top_stock_df.write.mode("overwrite").parquet(output_path)