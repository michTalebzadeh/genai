import sys
import os
import pandas as pd
import numpy as np
from faker import Faker
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import time

start_time = time.time()
# Global variable for DATA directory
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Initialize Faker
fake = Faker()


def create_data_dir():
    global DATA_DIR
    # Define the directory path relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(script_dir, "data", "data_files")
    # Create the directory if it does not exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


# Function to generate synthetic data
def create_large_table(num_rows):
    data = {
        'id1': np.arange(1, num_rows + 1),
        'feature_1': np.random.randint(0, 100, num_rows),
        'feature_2': np.random.rand(num_rows) * 100,
        'dt': pd.date_range(start='2024-01-01', periods=num_rows, freq='H').strftime('%Y-%m-%d').tolist(),
        'hr': pd.date_range(start='2024-01-01', periods=num_rows, freq='H').strftime('%H').tolist()
    }
    return pd.DataFrame(data)

def create_small_table(num_rows, large_table_ids):
    data = {
        'id2': np.random.choice(large_table_ids, num_rows, replace=False),
        'small_feature_1': [fake.city() for _ in range(num_rows)],
        'small_feature_2': np.random.rand(num_rows) * 50,
        'dt': pd.date_range(start='2024-01-01', periods=num_rows, freq='H').strftime('%Y-%m-%d').tolist(),
        'hr': pd.date_range(start='2024-01-01', periods=num_rows, freq='H').strftime('%H').tolist()
    }
    return pd.DataFrame(data)

# Generate synthetic data
large_table = create_large_table(num_rows=1000000)
small_table = create_small_table(num_rows=10000, large_table_ids=large_table['id1'])

# Assuming you have already generated and saved synthetic data
large_table = create_large_table(num_rows=1000000)
small_table = create_small_table(num_rows=10000, large_table_ids=large_table['id1'])

# Save tables as Parquet files
large_table.to_parquet(os.path.join(DATA_DIR, 'large_table.parquet'), engine='pyarrow')
small_table.to_parquet(os.path.join(DATA_DIR, 'small_table.parquet'), engine='pyarrow')

spark = SparkSession.builder \
        .appName("SyntheticDataExample") \
        .enableHiveSupport() \
        .getOrCreate()

# Set the log level to ERROR to reduce verbosity
sc = spark.sparkContext
sc.setLogLevel("ERROR")

num_buckets = 160 


# Load large_df and small_df from previously saved Parquet files
large_df_path = "file://" + os.path.abspath(os.path.join(DATA_DIR, 'large_table.parquet'))
small_df_path = "file://" + os.path.abspath(os.path.join(DATA_DIR, 'small_table.parquet'))

large_df = spark.read.parquet(large_df_path)
small_df = spark.read.parquet(small_df_path)

# Write large_df as a bucketed table
bucketed_table_path = os.path.join(DATA_DIR, 'bucketed_large_table')
large_df.write \
    .bucketBy(num_buckets, 'id1') \
    .sortBy('id1') \
    .format('parquet') \
    .mode('overwrite') \
    .saveAsTable('bucketed_large_table')
    # Bucketing and saving small_df

small_df.write \
    .bucketBy(16000, "id2") \
    .partitionBy("dt", "hr") \
    .sortBy("id2") \
    .format("parquet") \
    .mode("overwrite") \
    .option("path", os.path.join(DATA_DIR, "bucketed_small_table")) \
    .saveAsTable("bucketed_small_table")

    # Perform the join and measure performance


spark.sql("SELECT * FROM bucketed_large_table LIMIT 10").show()
spark.sql("SELECT * FROM bucketed_small_table LIMIT 10").show()
bucketed_large_df = spark.sql("SELECT * FROM bucketed_large_table WHERE dt = '2024-01-14'")
bucketed_small_df = spark.sql("SELECT * FROM bucketed_small_table WHERE dt = '2024-01-04'")
bucketed_large_df.printSchema()
bucketed_small_df.printSchema()

bucketed_large_df_count = bucketed_large_df.count()
bucketed_small_df_count = bucketed_small_df.count()


print(f"\n\n bucketed_large_df_count =  {bucketed_large_df_count}\n")
print(f"\n bucketed_small_df_count =  {bucketed_small_df_count}\n")


# Join the tables
spark.conf.set("spark.sql.legacy.bucketedTableScan.outputOrdering", True)
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1) # Elapsed time in seconds is 1421.746469259262
# Elapsed time in seconds is 1418.315283536911

# BroadcastHashJoin, Elapsed time in seconds is 1406.8763236999512
result = bucketed_large_df.join(bucketed_small_df, bucketed_large_df.id1 == bucketed_small_df.id2, "left")
result.explain('extended')
end_time = time.time()
time_elapsed = (end_time - start_time)
print(f"\n\n Elapsed time in seconds is {time_elapsed}")


