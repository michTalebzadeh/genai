# Initialise imports
import sys
import datetime
import sysconfig
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, datediff, expr, when, format_number, udf, rand, concat
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, FloatType, DoubleType, BooleanType
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.functions import struct
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from pyspark.sql.window import Window
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
from lmfit.models import LorentzianModel
from pyspark.sql import DataFrame



# Load data from Hive table
def load_hive_data(spark):
    DSDB = "DS"
    #tableName = "ocod_full_2020_12"
    tableName = "ocod_full_2024_03" # fraud table
    fullyQualifiedTableName = f"{DSDB}.{tableName}"
    if spark.sql(f"SHOW TABLES IN {DSDB} LIKE '{tableName}'").count() == 1:
       print(f"\n\n** Reading from {fullyQualifiedTableName} **\n\n")
       spark.sql(f"ANALYZE TABLE {fullyQualifiedTableName} COMPUTE STATISTICS")
       rows = spark.sql(f"SELECT COUNT(1) FROM {fullyQualifiedTableName}").collect()[0][0]
       print(f"\nTotal number of rows in table {fullyQualifiedTableName} is {rows}\n")
    else:
       print(f"No such table {fullyQualifiedTableName}")
       sys.exit(1)

    # create a dataframe from the loaded data
    house_df = spark.sql(f"SELECT * FROM {fullyQualifiedTableName}")
    print(f"\nData columns after reading table {fullyQualifiedTableName} into Spark DataFrame\n")
    house_df.printSchema()
    return house_df

def write_data_to_hive_table(df, tableName) -> None:
    tableSchema = StructType([
        StructField("TitleNumber", StringType(), nullable=True),
        StructField("Tenure", StringType(), nullable=True),
        StructField("PropertyAddress", StringType(), nullable=True),
        StructField("District", StringType(), nullable=True),
        StructField("County", StringType(), nullable=True),
        StructField("Region", StringType(), nullable=True),
        StructField("Postcode", StringType(), nullable=True),
        StructField("MultipleAddressIndicator", StringType(), nullable=True),
        StructField("PricePaid", IntegerType(), nullable=True),
        
        # Proprietor Information (Repeating for up to 4 proprietors)
        StructField("ProprietorName1", StringType(), nullable=True),
        StructField("CompanyRegistrationNo1", StringType(), nullable=True),
        StructField("ProprietorshipCategory1", StringType(), nullable=True),
        StructField("CountryIncorporated1", StringType(), nullable=True),
        StructField("Proprietor1Address1", StringType(), nullable=True),
        StructField("Proprietor1Address2", StringType(), nullable=True),
        StructField("Proprietor1Address3", StringType(), nullable=True),
        StructField("ProprietorName2", StringType(), nullable=True),
        StructField("CompanyRegistrationNo2", StringType(), nullable=True),
        StructField("ProprietorshipCategory2", StringType(), nullable=True),
        StructField("CountryIncorporated2", StringType(), nullable=True),
        StructField("Proprietor2Address1", StringType(), nullable=True),
        StructField("Proprietor2Address2", StringType(), nullable=True),
        StructField("Proprietor2Address3", StringType(), nullable=True),
        StructField("ProprietorName3", StringType(), nullable=True),
        StructField("ProprietorshipCategory3", StringType(), nullable=True),
        StructField("CountryIncorporated3", StringType(), nullable=True),
        StructField("Proprietor3Address1", StringType(), nullable=True),
        StructField("Proprietor3Address2", StringType(), nullable=True),
        StructField("Proprietor3Address3", StringType(), nullable=True),
        StructField("ProprietorName4", StringType(), nullable=True),
        StructField("CompanyRegistrationNo4", StringType(), nullable=True),
        StructField("ProprietorshipCategory4", StringType(), nullable=True),
        StructField("CountryIncorporated4", StringType(), nullable=True),
        StructField("Proprietor4Address1", StringType(), nullable=True),
        StructField("Proprietor4Address2", StringType(), nullable=True),
        StructField("Proprietor4Address3", StringType(), nullable=True),
        StructField("DateProprietorAdded", DateType(), nullable=True),
        StructField("AdditionalProprietorIndicator", StringType(), nullable=True),  
        StructField("features", VectorUDT(), nullable=True),
        StructField("missing_proprietor_name", IntegerType(), nullable=False),
        StructField("is_fraud", IntegerType(), nullable=False),
    ])

    DSDB = "DS"
    fullyQualifiedTableName = f"{DSDB}.{tableName}"
    try:
        df.write.mode("overwrite").option("schema", tableSchema).saveAsTable(fullyQualifiedTableName)
        print(f"Dataframe data written to table: {fullyQualifiedTableName}")
    except Exception as e:
        print(f"Error writing data: {e}")

class DataFrameProcessor:
    def __init__(self, spark):
        self.spark = spark


    # Cleans up loaded data and retun back dynamic_schema
    def cleanup_data(self, house_df):
        # Get rid of rogue value from Tenure. It is either Leasehold or Freehold
        house_df = house_df.filter(house_df["Tenure"] != "93347")
        # Cast the Tenure column to StringType
        house_df = house_df.withColumn("Tenure", col("Tenure").cast("string"))
        # Drop the columns from the DataFrame
        house_df = house_df.drop("CompanyRegistrationNo3")
        house_df = house_df.drop("Proprietor4Address3")
        # Define label column
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

    def analyze_fraudulent_transactions(self, house_df_final)  -> None:
        """
        This method analyzes the characteristics of fraudulent transactions in the DataFrame.

        Args:
            house_df_final: The Spark DataFrame containing the "is_fraud" flag.

        Prints:
            Insights about fraudulent transactions (e.g., count, groupings).
        """
 
        # Count total fraudulent transactions
        house_df_final.printSchema()
        total_fraudulent = house_df_final.where("is_fraud = 1").count()
        print(f"Total Fraudulent rows depicting fraudulent transactions where is_fraud = 1: {total_fraudulent}")

        # Group by "CountryIncorporated1" and count
        # Sort fraud_by_country by descending order of count
        fraud_by_country = (
            house_df_final.where("is_fraud = 1")
            .groupBy("CountryIncorporated1")
            .count()
            .orderBy(col("count").desc())  # Order by count in descending order
        )
        print("Overseas juristictions with the highest overseas property ownership in the UK")
        fraud_by_country.select(       
                 col("CountryIncorporated1").alias("Country Incorporated")
               , col("count").alias("No. of Registered Overseas Companies")
               ).show(20, False)  # Show top 20, truncate long column names
        
        # overseas ownership by district

        print("District and Regions grouped by the highest overseas property ownership")

        # Define the window specification
        wSpecD = Window().partitionBy('district', 'county')

        # Create the DataFrame with count column
        df9 = house_df_final.select(
            'district', 'county',
            F.count('district').over(wSpecD).alias("Offshore Owned")
        ).distinct()

        # Calculate total count of Offshore Owned properties overall
        total_count = F.sum("Offshore Owned").over(Window.orderBy(F.lit(1)))

        # Calculate percentage share for each line
        percentage_share = (F.col("Offshore Owned") / total_count) * 100

        # Add the percentage share column
        df9 = df9.withColumn("Percentage Share", F.round(percentage_share, 2))

        # Order by the "Offshore Owned" column in descending order
        df9 = df9.orderBy(F.desc("Offshore Owned"))

        # Show the DataFrame with the percentage share column
        df9.show(20, False)

        df2 = house_df_final. \
            select(
                'district', 'county',
                F.count('district').over(wSpecD).alias("Offshore Owned"),
                F.max(col("pricepaid")).over(wSpecD).alias("Most expensive property")
            ).distinct()
             
        #df2.show(20, False) 

        wSpecR = Window().orderBy(df2['Offshore Owned'].desc())

        df3 = df2.select(
            col("district").alias("District"),
            col("county").alias("County"),
            F.dense_rank().over(wSpecR).alias("rank"),
            col("Most expensive property"),
         ).filter(col("rank") <= 10)  # Filter top 10
  
        
        """
        p_df = df3.toPandas()
        print(p_df)
        p_df.plot(kind='bar', stacked=False, x='District', y=['Most expensive property'])
        plt.xticks(rotation=90)
        plt.xlabel("District", fontdict=font)
        plt.ylabel("# Of Offshore owned properties", fontdict=font)
        plt.title(f"UK Properties owned by offshore companies by districts", fontdict=font)
        plt.margins(0.15)
        plt.subplots_adjust(bottom=0.50)

        # Save the plot as an image file
        plt.savefig('plot.png')
        #plt.show()
        plt.close()
        """

        """
        By aliasing the DataFrames before joining them and prefixing the column names with the corresponding DataFrame
        alias, we have made it clear to Spark which DataFrame each column belongs to. 
        This prevents the ambiguity issue and allows Spark to correctly resolve the column references.
        """
        
        print("Most expensive offshore owned properties in the UK")
      
 
        # Perform the join operation with aliasing directly in the join conditions
        df4 = df3.join(
            house_df_final.alias("house_final"),
            on=[
                (df3["District"] == col("house_final.district")) &
                (df3["Most expensive property"] == col("house_final.pricepaid"))
            ],
            how="inner"
        ).select(
            df3["District"],
            df3["County"],
            df3["rank"],
            concat(lit("£"), format_number(df3["Most expensive property"], 2)).alias("Property value in GBP"),
            col("house_final.ProprietorName1").alias("Company name"),
            col("house_final.CountryIncorporated1").alias("Country Incorporated"),
            col("house_final.Tenure").alias("Tenure"),
            col("house_final.PropertyAddress").substr(1, 40).alias("Property address"),
            col("house_final.MultipleAddressIndicator")
        )

        # Show the result
        df4.show(truncate=False)

        df4.show(20, False)
  
       
        # Additional analysis based on your needs (e.g., group by price range)
        # ...
    def analyze_SouthKensingtonTransactions(self, house_df_final: DataFrame) -> None:
            # Filter for transactions in Kensington and Chelsea district
            results = house_df_final.filter(col('District').rlike("(?i)KENSINGTON AND CHELSEA"))
            results = house_df_final.filter(col('TitleNumber').contains('BGL35462'))

            # Select columns that are not null for the filtered records
            non_null_columns = [col_name for col_name in results.columns if results.select(col(col_name)).dropna().count() > 0]
            # Show the top 5 results with non-null columns ordered by PricePaid in descending order
   
        # Perform the select and formatting operations
            results.select(
                col('TitleNumber'),
                col('PropertyAddress').substr(1, 50).alias("Property address"),
                col('Postcode').alias("post code"),
                concat(lit("£"), format_number(col('PricePaid'), 2)).alias("purchase price"),
                col('CountryIncorporated1').alias("Incorporated"),
                col('ProprietorName1').alias("ProprietorName1")
            ).orderBy(col('PricePaid').asc()).show(1, truncate=False)

            results.select('*').show(1, truncate=False)
            """
            # Example DataFrame schema based on your description
            data = [
                (
                    "TN123456",                          # TitleNumber
                    "Freehold",                          # Tenure
                    "1 Some Street, Some Town, Some City, ABC123",  # PropertyAddress
                    "Some District",                     # District
                    "Some County",                       # County
                    "Some Region",                       # Region
                    "AB12 3CD",                          # Postcode
                    "No",                                # MultipleAddressIndicator
                    5000000,                             # PricePaid
                    "Company A Ltd.",                    # ProprietorName1
                    "12345678",                          # CompanyRegistrationNo1
                    "Company",                           # ProprietorshipCategory1
                    "UK",                                # CountryIncorporated1
                    "Address Line 1",                    # Proprietor1Address1
                    "Address Line 2",                    # Proprietor1Address2
                    "Address Line 3",                    # Proprietor1Address3
                    "Company B Ltd.",                    # ProprietorName2
                    "87654321",                          # CompanyRegistrationNo2
                    "Company",                           # ProprietorshipCategory2
                    "UK",                                # CountryIncorporated2
                    "Address Line 4",                    # Proprietor2Address1
                    "Address Line 5",                    # Proprietor2Address2
                    "Address Line 6",                    # Proprietor2Address3
                    "Company C Ltd.",                    # ProprietorName3
                    "11223344",                          # CompanyRegistrationNo3
                    "Company",                           # ProprietorshipCategory3
                    "UK",                                # CountryIncorporated3
                    "Address Line 7",                    # Proprietor3Address1
                    "Address Line 8",                    # Proprietor3Address2
                    "Address Line 9",                    # Proprietor3Address3
                    "Company D Ltd.",                    # ProprietorName4
                    "44332211",                          # CompanyRegistrationNo4
                    "Company",                           # ProprietorshipCategory4
                    "UK",                                # CountryIncorporated4
                    "Address Line 10",                   # Proprietor4Address1
                    "Address Line 11",                   # Proprietor4Address2
                    "2022-01-01",                        # DateProprietorAdded
                    "Yes",                               # AdditionalProprietorIndicator
                    None,                                # features (assuming vector type, can be null or some array of numbers)
                    0                                    # missing_proprietor_name (assuming an integer flag for missing names)
                )
            ]

            # Create DataFrame
            columns = [
                "TitleNumber", "Tenure", "PropertyAddress", "District", "County", "Region", "Postcode",
                "MultipleAddressIndicator", "PricePaid", "ProprietorName1", "CompanyRegistrationNo1",
                "ProprietorshipCategory1", "CountryIncorporated1", "Proprietor1Address1",
                "Proprietor1Address2", "Proprietor1Address3", "ProprietorName2", "CompanyRegistrationNo2",
                "ProprietorshipCategory2", "CountryIncorporated2", "Proprietor2Address1", "Proprietor2Address2",
                "Proprietor2Address3", "ProprietorName3", "CompanyRegistrationNo3", "ProprietorshipCategory3",
                "CountryIncorporated3", "Proprietor3Address1", "Proprietor3Address2", "Proprietor3Address3",
                "ProprietorName4", "CompanyRegistrationNo4", "ProprietorshipCategory4", "CountryIncorporated4",
                "Proprietor4Address1", "Proprietor4Address2", "DateProprietorAdded", "AdditionalProprietorIndicator",
                "features", "missing_proprietor_name"
            ]

            df = self.spark.createDataFrame(data, columns)
            df.show(truncate=False)
            
            # Filter for unusual patterns where the ratio is significantly high
            threshold = 2  # Example threshold for unusual patterns
            unusual_df = df.filter(col("MortgageToPriceRatio") > threshold)
            """


    def process_fraud_data(self, house_df):
        # 1. Encode Categorical Variables
        categorical_cols = ["Tenure", "PropertyAddress", "District", "County", "Region", "Postcode", "MultipleAddressIndicator"]
        print(f"\n\nCategorical columns are {categorical_cols}\n\n")
        indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="skip") for col in categorical_cols]
        indexer_model = [indexer.fit(house_df) for indexer in indexers]
        house_df_indexed = house_df
        #print("\nSchema before dropping categorical columns:")
        #house_df_indexed.printSchema()
     
        for model in indexer_model:
            house_df_indexed = model.transform(house_df_indexed)
        #print("\nSchema before dropping categorical columns:")
        #house_df_indexed.printSchema()
    
        # 2. Drop categorical columns
        #house_df_indexed = house_df_indexed.drop(*categorical_cols)
        
        house_df_indexed = house_df_indexed.drop( \
                      col("Tenure_index")
                    , col("PropertyAddress_index") \
                    , col("District_index") \
                    , col("County_index") \
                    , col("Region_index") \
                    , col("Postcode_index") \
                    , col("MultipleAddressIndicator_index") \
                    ) 
        
        #print("\nSchema after dropping categorical columns:")
        #house_df_indexed.printSchema()
    
        missing_value_analysis = self.analyze_missing_values(house_df_indexed)
        columns_with_null = missing_value_analysis['columns_with_null']
        null_percentage_dict = missing_value_analysis['null_percentage_dict']
        null_columns_percentage = missing_value_analysis['null_columns_percentage']

        print(f"\nColumns with missing values:")
        for col_name in columns_with_null:
            print(f"{col_name}:\t{null_percentage_dict[col_name]}%")

        print(f"\nSummary percentage of columns with no values (all null): {null_columns_percentage:.1f}%")

  # ... existing code for dropping categorical columns, vector assembling, etc. ...
        # 3. Apply VectorAssembler
        numerical_cols = ["PricePaid"]
        assembler_input = ["PricePaid"]
        assembler = VectorAssembler(inputCols=assembler_input, outputCol="features", handleInvalid="skip")  # or "skip"

        house_df_final = assembler.transform(house_df_indexed)
       
        # Define the index value for leasehold tenure
        leasehold_index = 1.0

        # Filter rows where Tenure_index is equal to the leasehold index
        leasehold_index = 1.0  # Update this with the actual index value for leasehold tenure
        leasehold_tenure = col("Tenure_index") == leasehold_index

        # Apply additional transformations and add the is_fraud column
        high_price_threshold = 1000000
        foreign_incorporation = col("CountryIncorporated1") != "UK"


        # Check for missing values in multiple proprietor names
        missing_proprietor_name = col("ProprietorName1").isNull() | col("ProprietorName2").isNull()

        # Additional condition based on missing data
        missing_proprietor_info = missing_proprietor_name | col("Proprietor1Address1").isNull()

        # Impute missing price (if applicable and data allows)
        # house_df_cleaned = house_df_cleaned.fillna(average_price, subset=["PricePaid"])  # Example imputation

        # Create a feature for missing proprietor name
        house_df_final = house_df_final.withColumn("missing_proprietor_name", missing_proprietor_name.cast("int"))

        # Chain when conditions (including missing value condition)
        house_df_final = house_df_final.withColumn(
            "is_fraud",
            when(col("PricePaid") > high_price_threshold, 1)
            .when(missing_proprietor_info, 1)
            .when(foreign_incorporation, 1)
            .when(col("missing_proprietor_name") == 1, 1)  # Flag for missing proprietor name
            .otherwise(0)
            # ... add other when conditions ...
        )

        self.analyze_SouthKensingtonTransactions(house_df_final)

        house_df_final.createOrReplaceTempView("house_df_final")
        house_df.createOrReplaceTempView("house_df")
        sqlText = f"""
        SELECT h.*
        FROM
            house_df h
        WHERE
            NOT EXISTS(SELECT 1 FROM house_df_final f WHERE h.TitleNumber = f.TitleNumber)       
        """
        good_offshore_data = self.spark.sql(sqlText)


        # Analyze fraudulent transactions after processing
        self.analyze_fraudulent_transactions(house_df_final) 
          
        return house_df_final, good_offshore_data

 
    def analyze_missing_values(self, house_df):
        """
        Analyzes missing values in the DataFrame and returns insights.

        Args:
            house_df (spark.sql.DataFrame): The DataFrame to analyze.

        Returns:
            dict: A dictionary containing information about missing values.
                - 'columns_with_null': List of column names with missing values.
                - 'null_percentage_dict': Dictionary mapping column names to percentages of missing values.
                - 'null_columns_percentage': Overall percentage of columns with 100% missing values.
        """
        null_value_percentages = {}

        total_rows = house_df.count()

        for col_name in house_df.columns:
            null_count = house_df.filter(col(col_name).isNull()).count()
            null_percentage = (null_count / total_rows) * 100
            null_value_percentages[col_name] = round(null_percentage, 1)

        columns_with_null = [col_name for col_name, percentage in null_value_percentages.items() if percentage > 0]

        null_columns_count = sum(1 for percentage in null_value_percentages.values() if percentage == 100.0)
        null_columns_percentage = (null_columns_count / len(null_value_percentages)) * 100

        return {
            'columns_with_null': columns_with_null,
            'null_percentage_dict': null_value_percentages,
            'null_columns_percentage': null_columns_percentage
        }

    def evaluate_synthetic_data_quality(self,bad_data, synthetic_bad_data):
        """
        Evaluate the quality of synthetic data compared to original bad data.

        Args:
            bad_data (DataFrame): DataFrame containing original bad data.
            synthetic_bad_data (DataFrame): DataFrame containing synthetic bad data.

        Returns:
            dict: Dictionary containing evaluation metrics and insights.
        """
        evaluation_results = {}

        # Compare the shape (number of rows and columns) of the two datasets
        original_shape = (bad_data.count(), len(bad_data.columns))
        synthetic_shape = (synthetic_bad_data.count(), len(synthetic_bad_data.columns))
        evaluation_results['original_shape'] = original_shape
        evaluation_results['synthetic_shape'] = synthetic_shape

        # Compare descriptive statistics for numerical features
        original_describe = bad_data.describe().toPandas().set_index('summary')
        synthetic_describe = synthetic_bad_data.describe().toPandas().set_index('summary')
        evaluation_results['original_describe'] = original_describe
        evaluation_results['synthetic_describe'] = synthetic_describe

        # Compare distributions of numerical features (e.g., histograms)
        # This can be visualized using plotting libraries like matplotlib or seaborn

        # Compare categorical feature distributions
        original_categorical_counts = {}
        synthetic_categorical_counts = {}
        categorical_cols = [col_name for col_name, data_type in bad_data.dtypes if data_type == 'string']
        for col_name in categorical_cols:
            original_categorical_counts[col_name] = bad_data.groupBy(col_name).count().collect()
            synthetic_categorical_counts[col_name] = synthetic_bad_data.groupBy(col_name).count().collect()
        evaluation_results['original_categorical_counts'] = original_categorical_counts
        evaluation_results['synthetic_categorical_counts'] = synthetic_categorical_counts

        return evaluation_results


def preprocess_data(features):
    # Your preprocessing logic here
    # For example, let's say you want to normalize the features
    
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
    normalized_features = [feature / 255.0 for feature in features]
    return normalized_features

# Register the preprocess_data function as a UDF
preprocess_udf = udf(preprocess_data, ArrayType(DoubleType()))


def main():
    appName = "fraud1.py"
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName(appName) \
        .enableHiveSupport() \
        .getOrCreate()
        # Set the configuration
    
     # Set the log level to ERROR to reduce verbosity
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    
    # Create DataFrameProcessor instance
    df_processor = DataFrameProcessor(spark)
    print(f"\n Running module {appName}\n")
    # Load hive data
    house_df = load_hive_data(spark)
    
    # Clean up data
    house_df_cleaned = df_processor.cleanup_data(house_df)

    # Process the fraud data
    house_df_processed, good_offshore_data  = df_processor.process_fraud_data(house_df_cleaned)
    print(f"printing house_df_processed")
    house_df_processed.printSchema()
    print(f"printing good_offshore_data")
    good_offshore_data.printSchema()

    total_data_count = house_df.count()
    print(f"Total data count: {total_data_count}")
    bad_data = house_df_processed.filter(col("is_fraud") == 1)
    bad_data_count = bad_data.count()
    print(f"bad data count: {bad_data_count}")
    # Calculate the percentage of bad rows
    bad_data_percentage = (bad_data_count / total_data_count) * 100
    print(f"Percentage of bad data: {bad_data_percentage:.2f}%")
    
    # Now use generative AI to create synthetic data based on bad rows
    # Select only the bad rows (fraudulent transactions)
    
    # Calculate the fraction of bad rows
    bad_data_fraction = bad_data_count / total_data_count

    # Sample from the bad rows to create bad synthetic data
    """
    In summary, the line of code is creating a synthetic dataset (synthetic_bad_data) by sampling 
    a fraction of instances from the original "bad" data (bad_data) with replacement, 
    meaning that instances can be sampled multiple times. 
    The random sampling process is controlled by setting a seed value (42) for reproducibility.
    This synthetic dataset can be used for various purposes, such as augmenting training data for fraud detection models or 
    conducting experiments without modifying the original dataset.
    """
    synthetic_bad_data = bad_data.sample(
        withReplacement=True, fraction=bad_data_fraction, seed=42)
    """
    By unioning the original "bad" data with the synthetic data, the new_bad_data DataFrame combines both
    real instances of "bad" data and synthetic instances generated to augment the dataset.
    This combined dataset can then be used for training machine learning models, such as fraud detection algorithms,
    to improve their performance and robustness by providing a more diverse and representative set of training examples.
    """
    # Union the synthetic data with the original bad rows DataFrame
    new_bad_data = bad_data.union(synthetic_bad_data)

    # Get the count of synthetic rows created
    synthetic_bad_rows_count = synthetic_bad_data.count()
    print(f"synthetic bad data row count: {synthetic_bad_rows_count}")
    # Optionally, shuffle the DataFrame to randomize the order of rows
    fraud_data = new_bad_data.orderBy(rand())
    total_bad_data_count = fraud_data.count()
    print(f"combined bad data row count: {total_bad_data_count}\n")
    
     # Calculate the fraction of good offshore data
    good_offshore_data_count = good_offshore_data.count()
    
    good_offshore_data_fraction = good_offshore_data_count / total_data_count
    good_data_percentage = (good_offshore_data_count / total_data_count) * 100
    print(f"good_offshore_data count: {good_offshore_data_count}")
    print(f"good_offshore_data percentage is {good_data_percentage:.2f}%")
 
    # Call the method to evaluate synthetic data quality
    evaluation_results = df_processor.evaluate_synthetic_data_quality(bad_data, synthetic_bad_data)

    # Extract and analyze each component of evaluation_results
    print(f"\n\nExtract and analyze each component of evaluation_results to evaluate synthetic data quality\n\n")
    # 1. Shape Comparison
    original_shape = evaluation_results['original_shape']
    synthetic_shape = evaluation_results['synthetic_shape']
    print("Original Data Shape:", original_shape)
    print("Synthetic Data Shape:", synthetic_shape)
    # Compare the shapes to see if they are similar

    # 2. Descriptive Statistics Comparison
    original_describe = evaluation_results['original_describe']
    synthetic_describe = evaluation_results['synthetic_describe']
    # You can print or analyze the descriptive statistics to compare the distributions of numerical features
    print("Original Data Descriptive Statistics:")
    print(original_describe)
    print("Synthetic Data Descriptive Statistics:")
    print(synthetic_describe)

    # 3. Distribution Comparison
    # You can visualize and compare the distributions of numerical features using plotting libraries like matplotlib or seaborn
    # Example:
    # import matplotlib.pyplot as plt
    # Plot histograms for numerical features
    # For example, if 'PricePaid' is a numerical feature:
    # plt.hist(bad_data.select('PricePaid').rdd.flatMap(lambda x: x).collect(), bins=20, alpha=0.5, label='Original')
    # plt.hist(synthetic_bad_data.select('PricePaid').rdd.flatMap(lambda x: x).collect(), bins=20, alpha=0.5, label='Synthetic')
    # plt.legend(loc='upper right')
    # plt.xlabel('PricePaid')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of PricePaid')
    # plt.show()

    # 4. Categorical Feature Comparison
    original_categorical_counts = evaluation_results['original_categorical_counts']
    synthetic_categorical_counts = evaluation_results['synthetic_categorical_counts']
    # Compare the distributions of categorical features
    # You can analyze the counts or visualize them using bar charts, for example
    # Example:
    # for col_name, counts in original_categorical_counts.items():
    #     print("Original Data -", col_name, "Counts:", counts)
    #     # Similarly, print or visualize the counts for synthetic data

    # write dataframes to Hive tables   
    print(f"\n\nWriting dataframes to Hive tables\n\n")
    write_data_to_hive_table(bad_data, 'bad_data') # bad data derived from csv
    write_data_to_hive_table(synthetic_bad_data, 'synthetic_bad_data')
    write_data_to_hive_table(fraud_data, 'bad_fraud_data') # total_bad_data_count = fraud_data.count()
    write_data_to_hive_table(good_offshore_data, 'good_offshore_data') 
  
    
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


