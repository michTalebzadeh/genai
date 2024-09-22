# Initialise imports
import sys
import datetime
import sysconfig
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, datediff, expr, when, format_number, udf, rand
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import ( 
    col, 
    count, 
    lit, 
    datediff, 
    when, 
    isnan, 
    expr, 
    min as spark_min,
    max as spark_max, 
    avg, 
    udf, 
    rand
)
import json
from pyspark.sql.types import (
    StringType,
    IntegerType,
    DoubleType,
    ArrayType,
    StructField,
    StructType,
    BooleanType,
)
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
import pandas as pd
from pyspark.ml.feature import Bucketizer
 


class DataFrameProcessor:
    def __init__(self, spark):
        self.spark = spark

    # Load data from Hive table
    def load_data(self):
        DSDB = "DS"
        #tableName = "ocod_full_2020_12"
        tableName = "ocod_full_2024_03" # fraud table
        fullyQualifiedTableName = f"{DSDB}.{tableName}"
        if self.spark.sql(f"SHOW TABLES IN {DSDB} LIKE '{tableName}'").count() == 1:
            self.spark.sql(f"ANALYZE TABLE {fullyQualifiedTableName} COMPUTE STATISTICS")
            rows = self.spark.sql(f"SELECT COUNT(1) FROM {fullyQualifiedTableName}").collect()[0][0]
            print(f"\nTotal number of rows in table {fullyQualifiedTableName} is {rows}\n")
        else:
            print(f"No such table {fullyQualifiedTableName}")
            sys.exit(1)
        
        # create a dataframe from the loaded data
        house_df = self.spark.sql(f"SELECT * FROM {fullyQualifiedTableName}")
        return house_df     

 
    def freedman_diaconis_bins(self, data):
        """
        Calculates the optimal number of bins using the Freedman-Diaconis rule.

        Args:
            data: A list or NumPy array containing the data points.

        Returns:
            The optimal number of bins as an integer.
        """
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr * len(data) ** (-1 / 3)
        num_bins = int(np.ceil((max(data) - min(data)) / bin_width))
        num_bins = 33
        
        return num_bins


    def analyze_column(self, dataframe, column_name=None, use_hll=False, num_buckets=None):
        """
        Analyzes columns in a PySpark DataFrame and returns a dictionary with information.

        Args:
            dataframe: The PySpark DataFrame containing the columns.
            column_name: The name of the column to analyze (optional).
            use_hll: Whether to use HyperLogLog for approximate distinct count.
            num_buckets: Number of buckets for histogram calculation (optional).

        Returns:
            A dictionary containing information for each column.
        """

        analysis_results = {}

        # Analyze all columns if column_name is not provided
        if column_name is None:
            for col_name in dataframe.columns:
                try:
                    analysis_results[col_name] = DataFrameProcessor.analyze_single_column(self, dataframe, col_name, use_hll, num_buckets)
                except (KeyError, AttributeError):
                    # Handle potential errors if column doesn't exist or has incompatible data types
                    analysis_results[col_name] = {"exists": False}
        else:
            analysis_results[column_name] = DataFrameProcessor.analyze_single_column(self, dataframe, column_name, use_hll, num_buckets)

        return analysis_results


    def analyze_single_column(self, dataframe, column_name, use_hll=False, num_buckets=None):
        """
        Analyzes a single column in a PySpark DataFrame and returns a dictionary with information.

        Args:
            dataframe: The PySpark DataFrame containing the column.
            column_name: The name of the column to analyze.
            use_hll: Whether to use HyperLogLog for approximate distinct count.
            num_buckets: Number of buckets for histogram calculation (optional).

        Returns:
            A dictionary containing information about the column.
        """

        # Check if column exists
        if column_name not in dataframe.columns:
            return {"exists": False}

        # Get basic information
        num_rows = dataframe.count()
        data_type = dataframe.schema[column_name].dataType.simpleString()

        # Handle null values
        null_count = dataframe.filter((col(column_name).isNull()) | (col(column_name) == "")).count()
        null_percentage = (null_count / num_rows * 100) if num_rows > 0 else 0.0
        
        # Distinct count (exact or approximate)
        if use_hll:
            distinct_count = dataframe.select(expr(f"approx_count_distinct({column_name})")).collect()[0][0]
        else:
            distinct_count = dataframe.select(col(column_name)).distinct().count()

        # Calculate distinct percentage excluding nulls
        distinct_percentage = ((distinct_count - null_count) / num_rows * 100) if num_rows > 0 else 0.0

        # Calculate min, max, avg for numeric columns
        min_value = max_value = avg_value = None

        if isinstance(dataframe.schema[column_name].dataType, (IntegerType, DoubleType)):
            min_value = dataframe.select(spark_min(when(col(column_name).isNotNull(), col(column_name)))).first()[0]
            max_value = dataframe.select(spark_max(when(col(column_name).isNotNull(), col(column_name)))).first()[0]

            avg_value = round(dataframe.select(avg(col(column_name))).collect()[0][0],2)

            # Histogram calculation (example with error handling)
            histogram = None
            if num_rows > 0:
                filtered_df = dataframe.filter(col(column_name).isNotNull())
                non_null_count = filtered_df.count()
                if non_null_count > 0:
                    try:
 
                        
                        # Assuming filtered_df might have missing values
                        data_to_plot = filtered_df.select(col(column_name)).dropna().toPandas().values

                        # Now data_to_plot is a NumPy array
                        column_name = "PricePaid"
                        num_bins = DataFrameProcessor.freedman_diaconis_bins(self, data_to_plot)
                        print(f"\n\nNumber of bins is {num_bins}")

                        # Generate histogram using Pandas and Matplotlib
                        plt.hist(data_to_plot, bins=num_bins, alpha=0.5, label='Original')
                        plt.legend(loc='upper right')
                        plt.xlabel(column_name)
                        plt.ylabel('Frequency')
                        plt.title(f'Distribution of {column_name}')
                        plt.savefig(f'./{column_name}.png')
                        # plt.show()  # Uncomment if you want to display the plot

                        # Prepare histogram data for JSON output
                        hist, bin_edges = np.histogram(data_to_plot, bins=num_bins)
                        histogram = {
                            "bins": bin_edges.tolist(),
                            "counts": hist.tolist()
                        }

                    except Exception as e:
                        print(f"Error during histogram generation for {column_name}: {e}")
                else:
                    print(f"No non-null values in column {column_name}. Skipping histogram.")
            else:
                print(f"DataFrame is empty for column {column_name}. Skipping histogram.")
    

        return {
            "exists": True,
            "num_rows": num_rows,
            "data_type": data_type,
            "null_count": null_count,
            "null_percentage": round(null_percentage, 2),
            "distinct_count": distinct_count,
            "distinct_percentage": round(distinct_percentage, 2),
            "min_value": min_value,
            "max_value": max_value,
            "avg_value": avg_value,
            "histogram": histogram
        }

    def analyze_overseas_ownership(self, house_df_final) -> None:

        # define font dictionary
        font = {'family': 'serif',
                'color':  'darkred',
                'weight': 'normal',
                'size': 10,
                }

        # define font dictionary
        font_small = {'family': 'serif',
                'color':  'darkred',
                'weight': 'normal',
                'size': 8,
                }

        ownership_by_country = (
            house_df_final.groupBy("CountryIncorporated1")
                .count()
                .orderBy(col("count").desc())  # Order by count in descending order
        )
        
        print("Number of Foreign-Owned Companies Registered by Jurisdiction")

        ownership_by_country.select(
            col("CountryIncorporated1").alias("Country Incorporated")
            , col("count").alias("No. of Registered Overseas Companies")
        ).show(20, False) 

        
        # Define the window specification
        wSpecD = Window().partitionBy('district')

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
        print("Overseas properties owned per district with percentage share of total")

        df9.show(100,False)


        county = "GREATER LONDON"
        print(f"{county} properties owned per district")

        # Filter, select, and order the DataFrame
        df2 = house_df_final.filter(F.col('county') == county) \
            .select(
                'district',
                F.count('district').over(wSpecD).alias("Offshore Owned")
            ) \
            .distinct() \
            .orderBy(F.col("Offshore Owned").desc())

        df2.show(33,False)

        wSpecR = Window().orderBy(df2['Offshore Owned'])

        p_df = df2.toPandas()
        print(p_df.info())
 
        #print(p_df)
        
        p_df.plot(kind='bar', stacked=False, x='district', y=['Offshore Owned'])
        plt.xticks(rotation=90)
        plt.xlabel("District", fontdict=font)
        plt.ylabel("No Of Offshore owned properties", fontdict=font)
        plt.title(f"{county} Properties owned by offshore companies", fontdict=font)
        plt.margins(0.15)
        plt.subplots_adjust(bottom=0.50)
        plt.savefig('offshoreOwned.png')
        #plt.show()
        plt.close()
        
        """
        # Plotting
        plt.figure(figsize=(14, 10))  # Adjust figure size if needed
        ax = p_df.plot(kind='bar', stacked=False, x='district						', y='Offshore Owned', legend=True)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=90)

        # Set axis labels and title with custom font settings
        font = {'size': 12}
        plt.xlabel("District", fontdict=font)
        plt.ylabel("No Of Offshore owned properties", fontdict=font)
        plt.title("{county} Properties owned by offshore companies", fontdict=font)

        # Annotate each bar with the "Offshore Owned" value
        for i in range(len(p_df)):
            ax.text(i, p_df['Offshore Owned'][i] + 50, str(p_df['Offshore Owned'][i]), ha='center', va='bottom')

        # Adjust margins and layout
        plt.margins(0.15)
        plt.subplots_adjust(bottom=0.35)

        # Save the plot as a PNG file
        plt.savefig('offshoreOwned.png')
        plt.close()  
        """
        county = 'GREATER LONDON'
        #district = 'CITY OF WESTMINSTER'
        district = 'KENSINGTON AND CHELSEA'
         # Filter, select, and order by PricePaid in descending order
        df3 = house_df_final.filter(
            (F.col('county') == county) & 
            (F.col('district') == district) &
            (F.col('PricePaid').isNotNull()) & 
            (F.col('PricePaid') >= 1000000) & 
            (F.col('PricePaid') <= 100000000)
        ).select(
            F.col('PricePaid')
        ).orderBy(F.col('PricePaid').desc())

        df3.show(100,False)
        p_df = df3.toPandas()
        # Plot the distribution of PricePaid
        plt.figure(figsize=(10, 6))
        plt.hist(p_df['PricePaid'], bins=30, edgecolor='black')
        plt.xlabel('Price Paid', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title(f'Distribution of Price Paid in {district	}, {county}', fontsize=16)
        plt.grid(True)
        plt.savefig('HousePrices.png')
        # Show the plot
        #plt.show()      
        plt.close()
      
    
def main():
    appName = "House prices distribution"
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

    # Load data
    house_df = df_processor.load_data()

    total_data_count = house_df.count()
    print(f"Total data count: {total_data_count}")
  
    df_processor.analyze_overseas_ownership(house_df)  

    #column_to_analyze = None
    column_to_analyze = 'PricePaid'
  
    if column_to_analyze is None:
        # No column is defined, doing the analysis for all columns in the dataframe
        print(f"\nNo column is defined so doing do the analysis for all columns in the dataframe")
        analysis_results = df_processor.analyze_column(house_df)
    else:
        # Do the analysis for the specific column
        print(f"\n Doing analysis for column {column_to_analyze}")
        analysis_results = df_processor.analyze_column(house_df, column_to_analyze)
    
    # Print the analysis results for the requested column(s) only
    # print(analysis_results)
    # Convert analysis_results to JSON format
    analysis_results_json = json.dumps(analysis_results, indent=4)

    # Print the JSON-formatted analysis results
    print(f"\nJson formatted output\n")
    print(analysis_results_json)


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