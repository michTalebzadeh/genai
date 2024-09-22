import os
import sys
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_extract, sum as spark_sum, isnan, isnull, cast, udf
from pyspark.sql.types import IntegerType
import matplotlib.pyplot as plt
import pandas as pd
import re

# Global variable for PNG directory
PNG_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
def create_png_dir():
    global PNG_DIR
    # Define the directory path relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    PNG_DIR = os.path.join(script_dir, "designs", "png_files")
    # Create the directory if it does not exist
    if not os.path.exists(PNG_DIR):
        os.makedirs(PNG_DIR)

# Define a UDF for manual conversion
def to_int(size_mb):
    return int(size_mb) if size_mb is not None else None
to_int_udf = udf(to_int, IntegerType())


spark = SparkSession.builder.appName("FileSizeAnalysis").getOrCreate()
# Set the log level to ERROR to reduce verbosity
sc = spark.sparkContext
sc.setLogLevel("ERROR")

# Read the datafile
DIRECTORY="/home/hduser/dba/bin/python/genai/src/"
file_path = f"file:///{DIRECTORY}/size.txt"
print(f"\n\n file path is {file_path}")
# Read the data as a text file
df = spark.read.text(file_path)
# Extract the size and release version from the data
df = df.withColumn("size", F.regexp_extract(F.col("value"), r"^(\d+\.?\d*)(M|G)", 1).cast("double")) \
           .withColumn("unit", F.regexp_extract(F.col("value"), r"^(\d+\.?\d*)(M|G)", 2)) \
           .withColumn("release", F.regexp_extract(F.col("value"), r"^\d+(M|G) (.+)", 2)) \
           .withColumn("major_release", F.regexp_extract(F.col("release"), r"(\d+)\.", 1))

df = df.withColumn("size_in_mb", F.when(F.col("unit") == "G", F.col("size") * 1024).otherwise(F.col("size")))

# Convert size to MB (already in your code)
print(f"\n\n CPCN\n\n")


# Filter out rows with null or non-numeric size_MB
df_filtered = df.filter(~(isnan(col('size_MB')) | isnull(col('size_MB'))))

# Debugging step: Ensure size_MB is present and correct type
df_filtered.printSchema()
df_filtered.select("size_MB").show(5)

# Check the `major_release` column before filtering
df_filtered.select("major_release").distinct().show()
# Ensure `major_release` does not contain empty strings
df_filtered = df_filtered.filter(col("major_release") != "")

# Ensure `size_MB` is numeric and not null
df_filtered = df_filtered.filter(col("size_MB").cast("int").isNotNull())

# Cast `size_MB` to integer
df_filtered = df_filtered.withColumn("size_MB", col("size_MB").cast("int"))
df_filtered_sort = df_filtered.orderBy(col("size_MB").desc())
#df_filtered_sort.show(10)
df_filtered_sort_pd = df_filtered_sort.toPandas()
print(df_filtered_sort_pd)

# Confirm schema and values before aggregation
df_filtered.printSchema()
df_filtered.select("major_release", "size_MB").show(100)

# Assuming df_filtered_release is your DataFrame
df_filtered.show(100)

# Group and aggregate the data
grouped_data = df_filtered.groupBy("major_release").agg(
    F.sum(F.col("size_MB")).alias("total_size_MB")
)

# Show the results in Spark DataFrame
grouped_data.show(100)

# Convert Spark DataFrame to Pandas DataFrame
grouped_data_pd = grouped_data.toPandas()
print(grouped_data_pd)

# Create a bar chart using matplotlib
plt.figure(figsize=(14, 10))
plt.bar(grouped_data_pd['major_release'], grouped_data_pd['total_size_MB'])
plt.xlabel('Major Release')
plt.ylabel('Total Size (MB)')
plt.title('Total Size by Major Release')
plt.xticks(rotation=45)
plt.grid(True)
# Save the figure
plt.savefig(os.path.join(PNG_DIR, 'releaseSizes.png'))
plt.close()

# For scatter plot, filter null values and convert to Pandas
df_filtered = df.filter(col("size_MB").isNotNull()).toPandas()
plt.scatter(df_filtered['release'], df_filtered['size_MB'])
plt.xlabel('Release')
plt.ylabel('Size (MB)')
plt.title('Release Size Distribution (Non-Null Values)')
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig(os.path.join(PNG_DIR, 'release_size_scatter_nonull.png'))
plt.close()

spark.stop()
