#  Initialise imports
import sys
import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, datediff, expr, when, format_number
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, FloatType, DoubleType, BooleanType
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.functions import udf, rand
from pyspark.sql.types import DoubleType
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.functions import struct
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType
from tensorflow.keras import layers, Model
import numpy as np

class DataFrameProcessor:
    def __init__(self, spark):
        self.spark = spark

    # Load data from Hive table
    def load_data(self):
        DSDB = "DS"
        tableName = "ukhouseprices"
        fullyQualifiedTableName = f"{DSDB}.{tableName}"
        if self.spark.sql(f"SHOW TABLES IN {DSDB} LIKE '{tableName}'").count() == 1:
            self.spark.sql(f"ANALYZE TABLE {fullyQualifiedTableName} COMPUTE STATISTICS")
            rows = self.spark.sql(f"SELECT COUNT(1) FROM {fullyQualifiedTableName}").collect()[0][0]
            print(f"Total number of rows in table {fullyQualifiedTableName} is {rows}")
        else:
            print(f"No such table {fullyQualifiedTableName}")
            sys.exit(1)
        
        # create a dataframe from the loaded data
        house_df = self.spark.sql(f"SELECT * FROM {fullyQualifiedTableName}")
        return house_df
   
    def write_data(self, df, tableName) -> None:
        tableSchema = StructType([
            StructField("Datetaken", DateType(), nullable=True),
            StructField("RegionName", StringType(), nullable=True),
            StructField("AreaCode", StringType(), nullable=True),
            StructField("AveragePrice", DoubleType(), nullable=True),
            StructField("Index", DoubleType(), nullable=True),
            StructField("IndexSA", DoubleType(), nullable=True),
            StructField("oneMonthPercentChange", DoubleType(), nullable=True),
            StructField("twelveMonthPercentChange", DoubleType(), nullable=True),
            StructField("AveragePriceSA", DoubleType(), nullable=True),
            StructField("SalesVolume", DoubleType(), nullable=True),
            
            # Detached Property Prices and Indices
            StructField("DetachedPrice", DoubleType(), nullable=True),
            StructField("DetachedIndex", DoubleType(), nullable=True),
            StructField("Detached1mPercentChange", DoubleType(), nullable=True),
            StructField("Detached12mPercentChange", DoubleType(), nullable=True),
            
            # Semi-Detached Property Prices and Indices
            StructField("SemiDetachedPrice", DoubleType(), nullable=True),
            StructField("SemiDetachedIndex", DoubleType(), nullable=True),
            StructField("SemiDetached1mPercentChange", DoubleType(), nullable=True),
            StructField("SemiDetached12mPercentChange", DoubleType(), nullable=True),
            
            # Terraced Property Prices and Indices
            StructField("TerracedPrice", DoubleType(), nullable=True),
            StructField("TerracedIndex", DoubleType(), nullable=True),
            StructField("Terraced1mPercentChange", DoubleType(), nullable=True),
            StructField("Terraced12mPercentChange", DoubleType(), nullable=True),
            
            # Flat Property Prices and Indices
            StructField("FlatPrice", DoubleType(), nullable=True),
            StructField("FlatIndex", DoubleType(), nullable=True),
            StructField("Flat1mPercentChange", DoubleType(), nullable=True),
            StructField("Flat12mPercentChange", DoubleType(), nullable=True),
            
            # Cash Purchase Prices and Indices
            StructField("CashPrice", DoubleType(), nullable=True),
            StructField("CashIndex", DoubleType(), nullable=True),
            StructField("Cash1mPercentChange", DoubleType(), nullable=True),
            StructField("Cash12mPercentChange", DoubleType(), nullable=True),
            
            # Mortgage Purchase Prices and Indices
            StructField("MortgagePrice", DoubleType(), nullable=True),
            StructField("MortgageIndex", DoubleType(), nullable=True),
            StructField("Mortgage1mPercentChange", DoubleType(), nullable=True),
            StructField("Mortgage12mPercentChange", DoubleType(), nullable=True),
            
            # First Time Buyer Prices and Indices
            StructField("FTBPrice", DoubleType(), nullable=True),
            StructField("FTBIndex", DoubleType(), nullable=True),
            StructField("FTB1mPercentChange", DoubleType(), nullable=True),
            StructField("FTB12mPercentChange", DoubleType(), nullable=True),
            
            # Full Ownership Occupier Prices and Indices
            StructField("FOOPrice", DoubleType(), nullable=True),
            StructField("FOOIndex", DoubleType(), nullable=True),
            StructField("FOO1mPercentChange", DoubleType(), nullable=True),
            StructField("FOO12mPercentChange", DoubleType(), nullable=True),
            
            # New Build Prices and Indices
            StructField("NewPrice", DoubleType(), nullable=True),
            StructField("NewIndex", DoubleType(), nullable=True),
            StructField("New1mPercentChange", DoubleType(), nullable=True),
            StructField("New12mPercentChange", DoubleType(), nullable=True),
            
            # Existing Property Prices and Indices
            StructField("OldPrice", DoubleType(), nullable=True),
            StructField("OldIndex", DoubleType(), nullable=True),
            StructField("Old1mPercentChange", DoubleType(), nullable=True),
            StructField("Old12mPercentChange", DoubleType(), nullable=True),
            StructField("features", VectorUDT(), nullable=True), 
            StructField("is_fraud", BooleanType(), nullable=False),
        ])
        DSDB = "DS"
        fullyQualifiedTableName = f"{DSDB}.{tableName}"
        try:
            df.write.mode("overwrite").option("schema", tableSchema).saveAsTable(fullyQualifiedTableName)
            print(f"Dataframe data written to table: {fullyQualifiedTableName}")
        except Exception as e:
            print(f"Error writing data: {e}")

    # Cleans up loaded data and retun back dynamic_schema
    def cleanup_data(self, house_df):
        label_column = "is_fraud"
        
        # Call the method to find columns with null values
        columns_with_null, null_percentage_dict = self.find_columns_with_null_values(house_df)

        if len(columns_with_null) > 0:
            print("Columns with null values:")
            sorted_columns = sorted(null_percentage_dict.items(), key=lambda x: x[1], reverse=True)
            for col_name, null_percentage in sorted_columns:
                print(f"{col_name}:\t{null_percentage}%")  # Add \t for tab
            # Convert the sorted_columns list to a DataFrame
            null_percentage_df = self.spark.createDataFrame(sorted_columns, ["Column", "Null_Percentage"])
            # Add tab between column name and percentage in the DataFrame
            null_percentage_df = null_percentage_df.withColumn("Column_Null_Percentage", expr("concat(Column, '\t', cast(Null_Percentage as string), '%')"))
            # Write the DataFrame to a CSV file in the specified directory with overwrite
            DIRECTORY="/d4T/hduser/genai" 
            output_file_path = f"file:///{DIRECTORY}/null_percentage_list_ukhouseprices.csv"
            null_percentage_df.select("Column_Null_Percentage").coalesce(1).write.mode("overwrite").option("header", "true").csv(output_file_path)          
        else:
            print("No columns have null values.")

        # Calculate the summary percentage of columns that have no values (all null)
        total_columns = len(null_percentage_dict)
        null_columns_count = sum(1 for percentage in null_percentage_dict.values() if percentage == 100.0)
        null_columns_percentage = (null_columns_count / total_columns) * 100

        # Print summary percentage
        print(f"\nSummary percentage of columns with no values (all null): {null_columns_percentage:.1f}%")
        # Exclude columns with 100% null values
        filtered_null_percentage_dict = {col_name: percentage for col_name, percentage in null_percentage_dict.items() if percentage < 100}

        # Generate dynamic schema
        dynamic_schema = self.generate_dynamic_schema(house_df, filtered_null_percentage_dict)
        return house_df

    def find_columns_with_null_values(self, df):
        null_value_percentages = {}

        total_rows = df.count()

        for col_name in df.columns:
            null_count = df.filter(col(col_name).isNull()).count()
            null_percentage = (null_count / total_rows) * 100
            null_value_percentages[col_name] = round(null_percentage, 1)

        columns_with_null = [col_name for col_name, percentage in null_value_percentages.items() if percentage > 0]

        return columns_with_null, null_value_percentages

    def generate_dynamic_schema(self, house_df, filtered_null_percentage_dict):
        # Define the schema for the CSV file based on filtered columns
        schema = StructType([
            StructField(col_name, StringType() if percentage > 0 else IntegerType(), True)  # Assuming non-null columns are IntegerType
            for col_name, percentage in filtered_null_percentage_dict.items()
        ])

        # Print the schema
        #print("\nSchema for the CSV file:")
        #print(schema)

        # Get the DataFrame's schema
        df_schema = house_df.schema

        # Filter out columns with 100% null values
        non_null_columns = [col_name for col_name, null_percentage in filtered_null_percentage_dict.items() if null_percentage < 100]

        # Initialize an empty list to store StructFields for non-null columns
        fields = []

        # Iterate over each field in the DataFrame's schema
        for field in df_schema.fields:
            if field.name in non_null_columns:
                # Create a new StructField with the same name and data type for non-null columns
                new_field = StructField(field.name, field.dataType, nullable=True)
                # Append the new StructField to the list
                fields.append(new_field)

        # Create a new StructType schema using the list of StructFields
        dynamic_schema = StructType(fields)

        # Print the dynamically generated schema
        #print("\ndynamic schema:")
        #print(dynamic_schema)

        return dynamic_schema
    
    def process_non_fraud_data(self, house_df):
     
        # 1. Encode Categorical Variables
        # Identify categorical columns
        categorical_cols = ["RegionName", "AreaCode"]
     
        # Apply StringIndexer transformations
        indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="keep") for col in categorical_cols]
        indexer_models = [indexer.fit(house_df) for indexer in indexers]
        house_df_indexed = house_df
        for indexer_model in indexer_models:
            house_df_indexed = indexer_model.transform(house_df_indexed)

        # Print the schema after applying transformations
        #print("\nSchema after applying StringIndexer transformations:")
     
        
        print("\nSchema before dropping categorical columns:")
        house_df_indexed.printSchema()
        # Drop categorical columns
        # house_df_indexed = house_df_indexed.drop(*[col+"_index" for col in categorical_cols])
      
        house_df_indexed = house_df_indexed.drop( \
                      col("RegionName_index")
                    , col("AreaCode_index") \
                    ) 
        

        # Print the schema after dropping categorical columns
        print("\nSchema after dropping categorical columns:")
        house_df_indexed.printSchema()
     
        # 3. Apply VectorAssembler
        numerical_cols = ["AveragePrice"]
        assembler_input = ["AveragePrice"]
        assembler = VectorAssembler(inputCols=assembler_input, outputCol="features", handleInvalid="skip")
        house_df_final = assembler.transform(house_df_indexed)

        # Print the schema of the final DataFrame
        #print("\nSchema of the final DataFrame:")
        #house_df_final.printSchema()
        #house_df_final.show(5,False)

        # Apply additional transformations and add the is_fraud column
        high_price_threshold = 1000000
     
        house_df_final = house_df_final.withColumn(
            "is_fraud",
            when(
                col("AveragePrice") < high_price_threshold,  # Condition
                0  # Value if condition is true (not fraud)
            ).otherwise(
                1  # Value if condition is false (fraud)
            )
        )

        return house_df_final
        
def main():
    appName = "Generative Fraud Detection"
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName(appName) \
        .enableHiveSupport() \
        .getOrCreate()
    # Set the log level to ERROR to reduce verbosity
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    
    # Create DataFrameProcessor instance
    df_processor = DataFrameProcessor(spark)

    # Load data
    house_df = df_processor.load_data()
    house_df_cleaned = df_processor.cleanup_data(house_df)

    # Process the non_fraud data
    house_df_processed = df_processor.process_non_fraud_data(house_df_cleaned)
    print(f"printing house_df_processed")
    house_df_processed.printSchema()
    
    
    total_data_count = house_df.count()
    print(f"Total data count: {total_data_count}")
    good_data = house_df_processed.filter(col("is_fraud") == 0)
    good_data_count = house_df_processed.filter(col("is_fraud") == 0).count()
    print(f"good data count: {good_data_count}")
    # Calculate the percentage of good rows
    good_data_percentage = (good_data_count / total_data_count) * 100
    print(f"Percentage of good data: {good_data_percentage:.2f}%")

    # Now use generative AI to create synthetic data based on bad rows
    # Select only the bad rows (fraudulent transactions)
    good_rows = house_df_processed.filter(col("is_fraud") == 0)
  
   # Calculate the fraction of good rows
    good_data_fraction = good_data_count / total_data_count

   # Sample from the good rows to create good synthetic data
    synthetic_good_data = good_data.sample(withReplacement=True, fraction=good_data_fraction, seed=42)

    # Union the synthetic data with the original bad rows DataFrame
    new_good_data = good_data.union(synthetic_good_data)

    # Optionally, shuffle the DataFrame to randomize the order of rows
    new_good_data = new_good_data.orderBy(rand())

    # Get the count of synthetic rows created
    synthetic_good_rows_count = synthetic_good_data.count()
    print(f"synthetic good data row count: {synthetic_good_rows_count}")
    total_good_data_count = new_good_data.count()
    print(f"combined good data row count: {total_good_data_count}\n")
 
    df_processor.write_data(good_data, 'good_data') # from csv origin 
    df_processor.write_data(synthetic_good_data, 'synthetic_good_data')
    df_processor.write_data(new_good_data, 'good_fraud_data') # new_good_data = good_data.union(synthetic_good_data)

   
    # Stop the SparkSession
    spark.stop()

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print("PySpark code started at:", start_time)
    print("Working on fraud detection...")
    main()
    # Calculate and print the execution time
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    print("PySpark code finished at:", end_time)
    print("Execution time:", execution_time)


