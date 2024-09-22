import sys
import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, datediff, expr, when, format_number
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, FloatType
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.functions import struct


# UDF to extract a value from a vector
def extract_vector_element(vector, index):
    try:
        return float(vector[index])
    except:
        return None
# Register UDF
extract_vector_element_udf = udf(extract_vector_element, DoubleType())

class DataFrameProcessor:
    @staticmethod
    def find_columns_with_null_values(df):
        # Initialize dictionary to store column-wise null value percentages
        null_value_percentages = {}

        # Calculate total number of rows
        total_rows = df.count()

        # Iterate over columns
        for col_name in df.columns:
            # Count null values in the column
            null_count = df.filter(col(col_name).isNull()).count()

            # Calculate percentage of rows with null values in the column
            null_percentage = (null_count / total_rows) * 100

            # Store column-wise null value percentage rounded to one decimal point
            null_value_percentages[col_name] = round(null_percentage, 1)

        # Filter columns with null values
        columns_with_null = [col_name for col_name, percentage in null_value_percentages.items() if percentage > 0]

        return columns_with_null, null_value_percentages

def main():
    appName = "Fraud Detection"
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName(appName) \
        .enableHiveSupport() \
        .getOrCreate()
    # Set the log level to ERROR to reduce verbosity
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    DIRECTORY="/d4T/hduser/genai"
    # Get data from Hive table
    DSDB = "DS"
    tableName = "ocod_full_2020_12"
    fullyQualifiedTableName = f"{DSDB}.{tableName}"
    if spark.sql(f"SHOW TABLES IN {DSDB} LIKE '{tableName}'").count() == 1:
        spark.sql(f"ANALYZE TABLE {fullyQualifiedTableName} COMPUTE STATISTICS")
        rows = spark.sql(f"SELECT COUNT(1) FROM {fullyQualifiedTableName}").collect()[0][0]
        print(f"Total number of rows in table {fullyQualifiedTableName} is {rows}")
    else:
        print(f"No such table {fullyQualifiedTableName}")
        sys.exit(1)

    house_df = spark.sql(f"SELECT * FROM {fullyQualifiedTableName}")

    # Get rid of rogue value from Tenure. It is either Leasehold or Freehold
    house_df = house_df.filter(house_df["Tenure"] != "93347")
    # Cast the Tenure column to StringType
    house_df = house_df.withColumn("Tenure", col("Tenure").cast("string"))
    # Drop the columns from the DataFrame
    house_df = house_df.drop("CompanyRegistrationNo3")
    house_df = house_df.drop("Proprietor4Address3")
    # Define label column
    label_column = "is_fraud"
    # Create DataFrameProcessor instance
    df_processor = DataFrameProcessor()

    # Call the method to find columns with null values
    columns_with_null, null_percentage_dict = df_processor.find_columns_with_null_values(house_df)

    if len(columns_with_null) > 0:
        print("Columns with null values:")
        sorted_columns = sorted(null_percentage_dict.items(), key=lambda x: x[1], reverse=True)
        for col_name, null_percentage in sorted_columns:
            print(f"{col_name}:\t{null_percentage}%")  # Add \t for tab
        # Convert the sorted_columns list to a DataFrame
        null_percentage_df = spark.createDataFrame(sorted_columns, ["Column", "Null_Percentage"])
        # Add tab between column name and percentage in the DataFrame
        null_percentage_df = null_percentage_df.withColumn("Column_Null_Percentage", expr("concat(Column, '\t', cast(Null_Percentage as string), '%')"))
        # Write the DataFrame to a CSV file in the specified directory with overwrite
        output_file_path = f"file:///{DIRECTORY}/null_percentage_list.csv"
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

    # Define the schema for the CSV file based on filtered columns
    schema = StructType([
        StructField(col_name, StringType() if percentage > 0 else IntegerType(), True)  # Assuming non-null columns are IntegerType
        for col_name, percentage in filtered_null_percentage_dict.items()
    ])

    # Print the schema
    print("\nSchema for the CSV file:")
    print(schema)
 
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
    print("\ndynamic schema:")
    print(dynamic_schema)

    # 1. Encode Categorical Variables
    categorical_cols = ["Tenure", "PropertyAddress", "District", "County", "Region", "Postcode", "MultipleAddressIndicator"]
    print(categorical_cols)
    indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="skip") for col in categorical_cols]
    indexer_model = [indexer.fit(house_df) for indexer in indexers]
    house_df_indexed = house_df
    for model in indexer_model:
        house_df_indexed = model.transform(house_df_indexed)

    # 2. Drop Irrelevant Columns
    columns_to_drop = categorical_cols
    house_df_indexed = house_df_indexed.drop(*categorical_cols)

    # 3. Apply VectorAssembler
    numerical_cols = ["PricePaid"]
    assembler_input = numerical_cols + [col+"_index" for col in categorical_cols]
    assembler = VectorAssembler(inputCols=assembler_input, outputCol="features", handleInvalid="skip")  # or "skip"

    house_df_final = assembler.transform(house_df_indexed)

    # Display the transformed DataFrame
    house_df_final.printSchema()
    house_df_final.show(3)
    # Define the index value for leasehold tenure
    leasehold_index = 1.0

    # Filter rows where Tenure_index is equal to the leasehold index
    leasehold_index = 1.0  # Update this with the actual index value for leasehold tenure
    leasehold_tenure = col("Tenure_index") == leasehold_index

    # Apply additional transformations and add the is_fraud column
    # Define conditions for potential fraud
    high_price_threshold = 1000000
    missing_proprietor_name = col("ProprietorName1").isNull()
    foreign_incorporation = col("CountryIncorporated1") != "UK"

    # Add the is_fraud column with appropriate conditions
    house_df_final = house_df_final.withColumn(
        "is_fraud",
        when(
            (col("PricePaid") > high_price_threshold) |
            missing_proprietor_name |
            (missing_proprietor_name & leasehold_tenure) |
            foreign_incorporation,
            1
        ).otherwise(0)
    )

   
    house_df.printSchema()
    house_df_final.printSchema()
    
    # You can now use house_df_final for further analysis of potential fraud (e.g., filtering based on 'is_fraud' column)

    # Output suspicious transactions for further investigation
    print("Looking for suspicious transactions:")
    # Joining the aliased DataFrames on the TitleNumber column
    suspicious_transactions = house_df.alias('a').join(house_df_final.alias('b'),col('a.TitleNumber') == col('b.TitleNumber')). \
        select( \
                  col("b.is_fraud")
                , col("a.CountryIncorporated1").alias("CountryIncorporated")
                , col("a.PricePaid").alias("PricePaid")
                , col("a.TitleNumber").alias("TitleNumber")
                , col("a.Postcode").alias("Postcode")
                , col("a.District")
                , col("b.MultipleAddressIndicator_index") 
                , col("a.ProprietorshipCategory1")
                , col("a.companyregistrationno1")
              ) 
              #.orderBy(col("a.PricePaid").desc())

    suspicious_transactions.show(20, False)
             

    print("\n\n Random forrest stuff based on this suspicious_transactions schema")
    # Prepare data for Random Forest classification

    suspicious_transactions.printSchema()
    # Define features and label columns
    feature_cols = ["PricePaid", "MultipleAddressIndicator_index"]
    label_col = "is_fraud"

    # Assemble features into a vector
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Define the Random Forest classifier
    rf = RandomForestClassifier(featuresCol="features", labelCol=label_col)

    # Create a pipeline to chain assembler and RandomForestClassifier
    pipeline = Pipeline(stages=[assembler, rf])

    # Fit the pipeline to the data
    model = pipeline.fit(suspicious_transactions)

    # Make predictions
    predictions = model.transform(suspicious_transactions)

    # Show some predictions
    predictions.select("TitleNumber", "Postcode", "District", "is_fraud", "prediction").show(20, False)

    # Prepare data for Random Forest classification
    dataset = predictions.select("features", label_col)
  
    train_data, test_data = dataset.randomSplit([0.8, 0.2], seed=42)
   
    # Convert Tenure column from string to integer
    train_data = train_data.withColumn(label_column, train_data[label_column].cast(IntegerType()))

    # Train a Random Forest classifier
    rf = RandomForestClassifier(labelCol=label_column, featuresCol="features", numTrees=10)
    model = rf.fit(train_data)
    # Evaluate model performance on the training data
    predictions = model.transform(train_data)
    # Make predictions on test data
    predictions_test = model.transform(test_data)
    # Print the test dataset
    print("Sample Test Dataset:")
    test_data.show(2, truncate=False)
    print("Sample Training Dataset:")
    train_data.show(2, truncate=False)
    # Evaluate model performance
    evaluator = BinaryClassificationEvaluator(labelCol='is_fraud', metricName='areaUnderROC')
    auc_train = evaluator.evaluate(predictions)
    print("Area Under ROC (Training):", auc_train)
    # Evaluate model performance on the test data
    auc_test = evaluator.evaluate(predictions_test)
    print("Area Under ROC (Test):", auc_test)
    auc = evaluator.evaluate(predictions)
    print("Area Under ROC:", auc)
    
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

