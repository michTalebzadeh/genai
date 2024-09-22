#  Initialise imports
import sys
import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import sum, lit, datediff, expr, when, format_number
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, FloatType, DoubleType, BooleanType
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.types import DoubleType
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.functions import struct
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType
from tensorflow.keras import layers, Model
import numpy as np
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
from pyspark.sql.functions import year, month, dayofmonth, col, expr, avg, round, udf, rand, struct, udf, lower


class DataFrameProcessor:
    def __init__(self, spark):
        self.spark = spark

    # Load data from Hive tables
    def load_data(self):
        DSDB = "DS"
        """
        Total number of rows in table DS.good_fraud_data is 272956
        Total number of rows in table  DS.bad_fraud_data is  41387
        """   
        # Load good data (includes synthetic data as well)
        good_table_name = "good_fraud_data"
        fully_qualified_good_table_name = f"{DSDB}.{good_table_name}"
        if self.spark.sql(f"SHOW TABLES IN {DSDB} LIKE '{good_table_name}'").count() == 1:
            self.spark.sql(f"ANALYZE TABLE {fully_qualified_good_table_name} COMPUTE STATISTICS")
            good_rows = self.spark.sql(f"SELECT COUNT(1) FROM {fully_qualified_good_table_name}").collect()[0][0]
            print(f"Total number of rows in table {fully_qualified_good_table_name} is {good_rows}")
        else:
            print(f"No such table {fully_qualified_good_table_name}")
            sys.exit(1)
        
        # Load bad_fraud_data as conditioned below
        """
            # Add the is_fraud column with appropriate conditions
            # Define conditions for potential fraud
            high_price_threshold = 1000000
            missing_proprietor_name = col("ProprietorName1").isNull()
            foreign_incorporation = col("CountryIncorporated1") != "UK"
            house_df_final = house_df_final.withColumn(
                "is_fraud",
                when(
                    (col("PricePaid") > high_price_threshold) |
                    missing_proprietor_name |
                    (missing_proprietor_name & leasehold_tenure) |
                    foreign_incorporation,
                    1
                ).otherwise(0)
        """
        bad_table_name = "bad_fraud_data"
        #bad_table_name = "ocod_full_2024_03"
        fully_qualified_bad_table_name = f"{DSDB}.{bad_table_name}"
        print(f"\n\nUsing table {fully_qualified_bad_table_name}\n\n")
        if self.spark.sql(f"SHOW TABLES IN {DSDB} LIKE '{bad_table_name}'").count() == 1:
            self.spark.sql(f"ANALYZE TABLE {fully_qualified_bad_table_name} COMPUTE STATISTICS")
            bad_rows = self.spark.sql(f"SELECT COUNT(1) FROM {fully_qualified_bad_table_name}").collect()[0][0]
            print(f"Total number of rows in table {fully_qualified_bad_table_name} is {bad_rows}")
        else:
            print(f"No such table {fully_qualified_bad_table_name}")
            sys.exit(1)
    
        # Create dataframes from the loaded data
        good_df = self.spark.sql(f"SELECT * FROM {fully_qualified_good_table_name}")
        bad_df = self.spark.sql(f"SELECT * FROM {fully_qualified_bad_table_name}")      
        
        return good_df, bad_df # as tuple

    def analyze_data(self, good_df, bad_df) -> None:
        #good_df.printSchema()
        print(f"Bad DF schema")
        bad_df.printSchema()
        # Compare average prices per region between good_df and bad_dbg
        
        #Work out a comparison of prices paid for offshore properties compared to notmal properties  
        print(f"Average price comparison between offshore and normal properties per district in Greater London\n\n")
        good_df.createOrReplaceTempView("good_data")
        bad_df.createOrReplaceTempView("bad_data")
        # Apply the function to the DataFrame
        
        sqltext = f"""select district, count(district) as NumberOfOffshoreOwned from bad_data group by district order by NumberOfOffshoreOwned"""
        ownership_df = self.spark.sql(sqltext)
        
        # UK wide analysis
        sqltext = f"""
              SELECT 
                    region,
                    COUNT(region) AS NumberOfOffshoreOwned,
                    ROUND(SUM(pricepaid)/1000000,1) AS  total_price_in_billions_gbp
                FROM
                    bad_data
              WHERE 
                    region IS NOT NULL
              GROUP BY
                    region
              ORDER BY
                    NumberOfOffshoreOwned
              DESC
        """
        nationwide_df = self.spark.sql(sqltext)
        nationwide_df.show(100,False)
      
        # Total investment by juristctions
        sqltext = f"""
              SELECT
                    t.countryOfIncorporation,
                    t.NumberIncorporated,
                    t.total_price_in_billions_gbp
                FROM (
                    SELECT
                        countryincorporated1 AS countryOfIncorporation,
                        COUNT(countryincorporated1) AS NumberIncorporated,
                        ROUND(SUM(pricePaid)/1000000/1000,1) AS total_price_in_billions_gbp
                    FROM
                        ds.ocod_full_2024_03
                    GROUP BY
                        countryincorporated1
                    ) t
                ORDER BY
                    t.total_price_in_billions_gbp
                DESC
                LIMIT 15
        """
        offshore_ownership_df = self.spark.sql(sqltext)
        offshore_ownership_df.show(100,False)
    

        sqltext = f"""
                SELECT
                    district,
                    COUNT(district) AS NumberOfOffshoreOwned,
                    ROUND(SUM(pricepaid)/1000000/1000,1) AS  total_price_in_billions_gbp,
                    ROUND(SUM(pricepaid) / COUNT(district), 2) AS average_price_per_property
                 FROM
                    bad_data 
                WHERE 
                    UPPER(county) = 'GREATER LONDON'
                AND
                    district <> 'CITY OF LONDON'
                GROUP BY
                    district
                ORDER BY
                    total_price_in_billions_gbp
                DESC
        """
        print(" CPC 11")
        totalprice_df = self.spark.sql(sqltext)
        totalprice_df = totalprice_df.limit(15)
        totalprice_df.show(33,False)

        sqltext = f"""
                SELECT
                    bd.county AS County,
                    --gd.regionname,  -- This column is commented out
                    bd.district AS District,
                    ROUND(AVG(bd.pricepaid)) AS AverageOffshoreOwnedInGBP,
                    ROUND(AVG(gd.averageprice)) AS AverageUKGoodDataInGBP,
                    -- ROUND(100.0 * (ROUND(AVG(bd.pricepaid)) - ROUND(AVG(gd.averageprice))) / ROUND(AVG(gd.averageprice))) AS PercentageDifference,
                    FLOOR(ROUND((ROUND(AVG(bd.pricepaid)) / ROUND(AVG(gd.averageprice))))) AS PriceRatio
                FROM
                    good_data gd
                LEFT JOIN bad_data bd ON lower(gd.regionname) = lower(bd.district)
                WHERE
                    COALESCE(bd.district, 'NULL') <> 'NULL'
                    AND
                    UPPER(bd.county) = 'GREATER LONDON'
                GROUP BY
                    bd.county,
                    gd.regionname,  -- This column is included in GROUP BY
                    bd.district
        """

        result_df = self.spark.sql(sqltext)

        # Join the two DataFrames
        joined_df = result_df.join(ownership_df, on=["district"])
        # Exclude CITY OF LONDON as there no real private properties! 
 
        # Filter and limit the DataFrame
        limit_15_df = joined_df.select(
            col("County"), col("District"), col("NumberOfOffshoreOwned"), col("AverageOffshoreOwnedInGBP"),
            col("AverageUKGoodDataInGBP"), col("PriceRatio")
        ).filter(
            col("District") != 'CITY OF LONDON'
        ).orderBy(
            col('NumberOfOffshoreOwned').desc()
        ).limit(15)

        
        offshore_owned_df = limit_15_df.select("District", "NumberOfOffshoreOwned")

        # Calculate the total offshore-owned properties
        total_offshore_owned = limit_15_df.select(F.sum(col("NumberOfOffshoreOwned"))).collect()[0][0]

        # Add a column for the percentage of offshore-owned properties
        offshore_owned_df = offshore_owned_df.withColumn("Percentage", F.round(col("NumberOfOffshoreOwned") / total_offshore_owned * 100, 1))

        # Show the DataFrame
        #offshore_owned_df.show()

       # Extract data for plotting
        districts = offshore_owned_df.select("District").rdd.flatMap(lambda x: x).collect()
        offshore_owned = offshore_owned_df.select("NumberOfOffshoreOwned").rdd.flatMap(lambda x: x).collect()
        percentages = offshore_owned_df.select("Percentage").rdd.flatMap(lambda x: x).collect()

        # Create the bar chart with numbers and percentages in green color
        plt.figure(figsize=(15, 6))
        bars = plt.bar(districts, offshore_owned, color='b')

        # Add labels for NumberOfOffshoreOwned and percentage above each bar
        for bar, num_offshore_owned, percent in zip(bars, offshore_owned, percentages):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f"{num_offshore_owned} ({percent}%)",
                    ha='center', va='bottom', color='green')

        # Add a caption in green color at the top center
        plt.text(0.5, 0.85, 'NumberOfOffshoreOwned (%)', color='green', ha='center', va='center',
                transform=plt.gca().transAxes)

        plt.xlabel('London Districts')
        plt.ylabel('Number of Offshore Owned')
        plt.title('Number of Offshore Owned Properties by London Districts')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig('plot1')  # Save the plot before showing it
        #plt.show()

        # Extract data for plotting
        districts = totalprice_df.select("District").rdd.flatMap(lambda x: x).collect()
        offshore_owned = totalprice_df.select("NumberOfOffshoreOwned").rdd.flatMap(lambda x: x).collect()
        totalprice = totalprice_df.select("total_price_in_billions_gbp").rdd.flatMap(lambda x: x).collect()
        

        # Create the bar chart
        plt.figure(figsize=(15, 6))
        bars = plt.bar(districts, totalprice, color='b')

        # Add labels for NumberOfOffshoreOwned above each bar
        for bar, num_offshore_owned in zip(bars, offshore_owned):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, str(num_offshore_owned),
                    ha='center', va='bottom', color='green')

        # Add label for "Number of Companies Incorporated"
        plt.text(0.5, 0.85, 'Number of offshore owned properties', color='green', ha='center', va='center', transform=plt.gca().transAxes)


        plt.xlabel('London Districts')
        plt.ylabel('Total Price per District (in billions GBP)')
        plt.title('Total Price paid and number of Offshore Owned Properties')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig('plot2')  # Save the plot before showing it
        #plt.show()

        # Extract data for plotting for juristiction
        countryOfIncorporation = offshore_ownership_df.select("countryOfIncorporation").rdd.flatMap(lambda x: x).collect()
        numberIncorporated = offshore_ownership_df.select("NumberIncorporated").rdd.flatMap(lambda x: x).collect()
        total_price_in_billions_gbp = offshore_ownership_df.select("total_price_in_billions_gbp").rdd.flatMap(lambda x: x).collect()

        # Create bar chart
        plt.figure(figsize=(15, 6))
        bars = plt.bar(countryOfIncorporation, total_price_in_billions_gbp, color='b')

        # Add labels for numberIncorporated above each bar
        for bar, num_incorporated in zip(bars, numberIncorporated):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f"{num_incorporated}",
                    ha='center', va='bottom', color='green')

        # Add label for "Number of Companies Incorporated"
        plt.text(0.5, 0.85, 'Number of Companies Incorporated', color='green', ha='center', va='center', transform=plt.gca().transAxes)

        plt.xlabel('Juristicion')
        plt.ylabel('Investment (in billions GBP)')
        plt.title('Total Investment by Jurisdictions')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('plot3')  # Save the plot before showing it
        #plt.show()

        
        districts = totalprice_df.select("District").rdd.flatMap(lambda x: x).collect()
        offshore_owned = totalprice_df.select("NumberOfOffshoreOwned").rdd.flatMap(lambda x: x).collect()
        totalprice = totalprice_df.select("total_price_in_billions_gbp").rdd.flatMap(lambda x: x).collect()


        # Calculate the total offshore-owned properties
        total_nw_offshore_owned = nationwide_df.select(F.sum(col("NumberOfOffshoreOwned"))).collect()[0][0]
        total_nw_offshore_paid = nationwide_df.select((round(F.sum(col("total_price_in_billions_gbp")) / 1000,3)).alias("TotalOffshorePaidInBillions")).collect()[0][0]
        average_price_per_property = (total_nw_offshore_paid * 1e9) / total_nw_offshore_owned
        print("CPC 10")
        print(f"\n\nTotal properties offshore owned = {total_nw_offshore_owned}, total paid for these properties in billion = {total_nw_offshore_paid}, average price per property = {average_price_per_property}\n\n")

        # For good_df
        good_df_sum = good_df.select(F.sum('AveragePrice')).collect()[0][0]

        # For bad_df
        bad_df_sum = bad_df.select(F.sum('PricePaid')).collect()[0][0]

        # Format the total sum of AveragePrice to two decimal points
        formatted_good_df_sum = "{:.2f}".format(good_df_sum)

        # Format the total sum of PricePaid to two decimal points
        formatted_bad_df_sum = "{:.2f}".format(bad_df_sum)

        print(f"Total sum of AveragePrice in good_df: {formatted_good_df_sum}")
        print(f"Total sum of PricePaid in bad_df: {formatted_bad_df_sum}")

        # Calculate the ratio of PricePaid to AveragePrice for bad_df
        ratio_price_paid_average_price = bad_df_sum / good_df_sum

        # Format the ratio to two decimal points
        formatted_ratio = "{:.2f}".format(ratio_price_paid_average_price)

        print(f"Ratio of PricePaid to AveragePrice: {formatted_ratio}")

       
        # Extract data from DataFrames
        good_data = good_df.select("AveragePrice").rdd.flatMap(lambda x: x).collect()
        bad_data = bad_df.select("PricePaid").rdd.flatMap(lambda x: x).collect()

        # Adjust the range of histogram bins based on the range of values in the AveragePrice column
        min_average_price = good_df.selectExpr("min(AveragePrice)").collect()[0][0]
        max_average_price = good_df.selectExpr("max(AveragePrice)").collect()[0][0]
        num_bins = 20  # Adjust the number of bins as needed
        bin_width = (max_average_price - min_average_price) / num_bins

        # Plot the histogram with adjusted bin range
        plt.hist(good_df.select("AveragePrice").rdd.flatMap(lambda x: x).collect(), bins=num_bins, range=(min_average_price, max_average_price), alpha=0.5, label='Good Data')
        #plt.hist(bad_df.select("PricePaid").rdd.flatMap(lambda x: x).collect(), bins=num_bins, alpha=0.5, label='Bad Data')
        plt.xlabel("Price")
        plt.ylabel("Frequency")
        plt.title("Histogram of Price Paid")
        plt.legend()
 
        # Save the plot to a PNG file
        plt.savefig('model_training')

        plt.close()

    def train_random_forest(self, bad_df, label_column='is_fraud', num_trees=10, split_ratio=[0.8, 0.2], seed=None):
        # Preprocess the data
        # Convert date columns to numerical features
        bad_df = bad_df.withColumn("year", year("Datetaken")) \
                       .withColumn("month", month("Datetaken")) \
                       .withColumn("day", dayofmonth("Datetaken"))
        
        # Encode categorical columns
        region_indexer = StringIndexer(inputCol="RegionName", outputCol="region_index")
        area_indexer = StringIndexer(inputCol="AreaCode", outputCol="area_index")
        encoder = OneHotEncoder(inputCols=["region_index", "area_index"], outputCols=["region_vec", "area_vec"])

        # Apply transformations
        region_indexer_model = region_indexer.fit(bad_df)
        bad_df_indexed = region_indexer_model.transform(bad_df)

        area_indexer_model = area_indexer.fit(bad_df_indexed)
        bad_df_indexed = area_indexer_model.transform(bad_df_indexed)

        encoder_model = encoder.fit(bad_df_indexed)
        bad_df_encoded = encoder_model.transform(bad_df_indexed)

        # Assemble features
        assembler = VectorAssembler(inputCols=["year", "month", "day", "region_vec", "area_vec"], outputCol="features")
        bad_df = assembler.transform(bad_df_encoded)

        # Split the data into training and testing sets
        train_data, test_data = bad_df.randomSplit(split_ratio, seed)

        # Initialize Random Forest classifier
        rf = RandomForestClassifier(labelCol=label_column, featuresCol="features", numTrees=num_trees)

        # Fit the model on the training data
        model = rf.fit(train_data)

        # Make predictions on training data
        predictions_train = model.transform(train_data)

        # Make predictions on test data
        predictions_test = model.transform(test_data)

        # Evaluate model performance on training data
        evaluator_train = BinaryClassificationEvaluator(labelCol=label_column, metricName='areaUnderROC')
        auc_train = evaluator_train.evaluate(predictions_train)
        print("Area Under ROC (Training):", auc_train)

        # Evaluate model performance on test data
        auc_test = evaluator_train.evaluate(predictions_test)
        print("Area Under ROC (Test):", auc_test)

        return model, auc_train, auc_test



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
            StructField("is_non_fraud", BooleanType(), nullable=False),
        ])
        DSDB = "DS"
        fullyQualifiedTableName = f"{DSDB}.{tableName}"
        try:
            df.write.mode("overwrite").option("schema", tableSchema).saveAsTable(fullyQualifiedTableName)
            print(f"Dataframe data written to table: {fullyQualifiedTableName}")
        except Exception as e:
            print(f"Error writing data: {e}")

    
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
        #house_df_indexed.printSchema()
        # Drop categorical columns
        # house_df_indexed = house_df_indexed.drop(*[col+"_index" for col in categorical_cols])
      
        house_df_indexed = house_df_indexed.drop( \
                      col("RegionName_index")
                    , col("AreaCode_index") \
                    ) 
        

        # Print the schema after dropping categorical columns
        print("\nSchema after dropping categorical columns:")
        #house_df_indexed.printSchema()
     
        # 3. Apply VectorAssembler
        numerical_cols = ["AveragePrice"]
        assembler_input = ["AveragePrice"]
        assembler = VectorAssembler(inputCols=assembler_input, outputCol="features", handleInvalid="skip")
        house_df_final = assembler.transform(house_df_indexed)

        # Print the schema of the final DataFrame
        #print("\nSchema of the final DataFrame:")
        #house_df_final.printSchema()
        #house_df_final.show(5,False)

        # Apply additional transformations and add the is_non_fraud column
        high_price_threshold = 1000000
     
        house_df_final = house_df_final.withColumn(
            "is_non_fraud",
            when(
                col("AveragePrice") < high_price_threshold,  # Condition
                1  # Value if condition is true
            ).otherwise(
                0  # Value if condition is false
            )
        )

        return house_df_final

    def combine_dataframes(self, good_df, bad_df):
        # Create temporary views for good_df and bad_df
        good_df.createOrReplaceTempView("good_data")
        bad_df.createOrReplaceTempView("bad_data")
        
        # Perform a left join on regionname and district columns
        sqltext = """
            SELECT DISTINCT
                COALESCE(LOWER(bd.district), LOWER(gd.regionname)) AS region_name,
                bd.*,
                gd.*
            FROM
                good_data gd
            LEFT JOIN bad_data bd ON LOWER(gd.regionname) = LOWER(bd.district)
            WHERE
                UPPER(bd.county) = 'GREATER LONDON'
             AND
                UPPER(gd.regionname) = 'GREATER LONDON'   
        """  
        # Execute the SQL query and retrieve the combined dataframe
        combined_df = self.spark.sql(sqltext)

        return combined_df

def main():
    appName = "model training"
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName(appName) \
        .enableHiveSupport() \
        .getOrCreate()
    # Set the log level to ERROR to reduce verbosity
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    print(f"\n\n Module name running is {appName}\n\n")
    
    # Create DataFrameProcessor instance
    df_processor = DataFrameProcessor(spark)

    # Load data
    good_df, bad_df = df_processor.load_data()

    # Combine dataframes
    combined_df = df_processor.combine_dataframes(good_df, bad_df)
    print(f"\n\n combined good & bad schemas\n")
    #combined_df.printSchema()
    combined_df_count = combined_df.count()
    print(f"combined_df count is: {combined_df_count}\n")
 
    print("Loaded good data:")
    #good_df.show()

    print("Loaded bad data:")
    #bad_df.show()

    df_processor.analyze_data(good_df, bad_df)

    df_processor.train_random_forest(good_df, bad_df)
    
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


