# Initialise imports
import os
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
    DateType,
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
from pyspark.sql.functions import col, avg, lit, round

from lmfit.models import LinearModel, LorentzianModel, VoigtModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.stat import ChiSquareTest
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pandas.plotting import scatter_matrix
from scipy.optimize import curve_fit
import matplotlib.dates as mdates
import matplotlib.ticker as tkr
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from pyspark.sql.functions import col, avg
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


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

    print(f"PNG directory created at: {PNG_DIR}")
def load_json_data():
    # Load data from JSON file
    with open('/home/hduser/dba/bin/python/genai/data/property_data.json', 'r') as f:
       data = json.load(f)
    return data
    
# Load data from Hive table
def load_hive_data(spark):
    DSDB = "DS"
    tableName = "ukhouseprices"
    fullyQualifiedTableName = f"{DSDB}.{tableName}"
    if spark.sql(f"SHOW TABLES IN {DSDB} LIKE '{tableName}'").count() == 1:
       spark.sql(f"ANALYZE TABLE {fullyQualifiedTableName} COMPUTE STATISTICS")
       rows = spark.sql(f"SELECT COUNT(1) FROM {fullyQualifiedTableName}").collect()[0][0]
       print(f"\nTotal number of rows in table {fullyQualifiedTableName} is {rows}\n")
    else:
       print(f"No such table {fullyQualifiedTableName}")
       sys.exit(1)
        
    # create a dataframe from the loaded data
    house_df = spark.sql(f"SELECT * FROM {fullyQualifiedTableName}")
    house_df.printSchema()
    return house_df     

class DataFrameProcessor:
    def __init__(self, spark, data):
        self.spark = spark
        self.sdata = data

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


    def functionLorentzian(self, x, amp1, cen1, wid1, amp2,cen2,wid2, amp3,cen3,wid3):
        return (amp1*wid1**2/((x-cen1)**2+wid1**2)) +\
            (amp2*wid2**2/((x-cen2)**2+wid2**2)) +\
                (amp3*wid3**2/((x-cen3)**2+wid3**2))
    
    def write_summary_data(self, df, tableName) -> None:
        tableSchema = StructType([
        StructField("DateTaken", StringType(), True),
        StructField("Average Price", DoubleType(), True),
        StructField("Detached Price", DoubleType(), True),
        StructField("Semi Detached Price", DoubleType(), True),
        StructField("Terraced Price", DoubleType(), True),
        StructField("Flat Price", DoubleType(), True),
        StructField("year_month", IntegerType(), True)
    ])
        DSDB = "DS"
        fullyQualifiedTableName = f"{DSDB}.{tableName}"
        try:
            df.write.mode("overwrite").format("hive").option("schema", tableSchema).saveAsTable(fullyQualifiedTableName)
            print(f"Dataframe data written to table: {fullyQualifiedTableName}")
        except Exception as e:
            print(f"Error writing data: {e}")

    def analyze_uk_ownership(self, house_df_final) -> None:

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

       
        #RegionName = 'GREATER LONDON'
        RegionName = 'Kensington and Chelsea'
        #RegionName = 'City of Westminster'
        new_string = RegionName.replace(" ", "")
        StartDate = '2009-01-01'
        EndDate   = '2018-12-01'
        #StartDate = '2020-01-01'
        #EndDate   = '2023-12-31'
        print(f"\nAnalysis for London Borough of {RegionName} for period {StartDate} till {EndDate}\n")
        # Filter, select, and order by AveragePrice in descending order
        df3 = house_df_final.filter(
            (F.col('RegionName') == RegionName) & 
            (F.col('AveragePrice').isNotNull()) & 
            (F.col('DateTaken') >= StartDate) &
            (F.col('DateTaken') <= EndDate)
        ).select(
            F.col('AveragePrice').alias("Average Price")
        ).orderBy(F.col('AveragePrice').desc())

        df3.show(1000,False)
        p_dfm = df3.toPandas()  # converting spark DF to Pandas DF
        # Non-Linear Least-Squares Minimization and Curve Fitting
        # Define model to be Lorentzian and deploy it
        model = LorentzianModel()
        #n = len(p_dfm.columns)
   
     
        # Plot the distribution of AveragePrice
        plt.figure(figsize=(10, 6))
     
        plt.hist(p_dfm['Average Price'], bins=30, edgecolor='black')
        plt.xlabel('Average Price', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title(f'Distribution of Average price in {RegionName} from {StartDate} to {EndDate}', fontsize=16)
        plt.grid(True)
        plt.savefig(os.path.join(PNG_DIR, f'{new_string}_AveragePricePaid.png'))
        # Show the plot
        #plt.show()      
        plt.close()
      
      # Filter and select data
        df4 = house_df_final.filter(
            (F.col('RegionName') == RegionName) & 
            (F.col('AveragePrice').isNotNull()) & 
            (F.col('DateTaken') >= StartDate) &
            (F.col('DateTaken') <= EndDate)
        ).select(
            F.col('DateTaken'),
            F.col('AveragePrice').alias("Average Price"),
            F.col('DetachedPrice').alias("Detached Price"),
            F.col('SemiDetachedPrice').alias("Semi Detached Price"),
            F.col('TerracedPrice').alias("Terraced Price"),
            F.col('FlatPrice').alias("Flat Price")
        ).orderBy(F.col('DateTaken'))
    
        print(f"\nAll Property prices for London Borough of {RegionName}\n")
        df4.show(1000,False)
    
        # Count the number of transactions for each house type
        df_count = house_df_final.filter(
            (F.col('RegionName') == RegionName) & 
            (F.col('AveragePrice').isNotNull()) & 
            (F.col('DateTaken') >= StartDate) &
            (F.col('DateTaken') <= EndDate)
        ).select(
            F.col('DateTaken'),
            F.col('DetachedPrice').alias("Detached Price"),
            F.col('SemiDetachedPrice').alias("Semi Detached Price"),
            F.col('TerracedPrice').alias("Terraced Price"),
            F.col('FlatPrice').alias("Flat Price")
        ).groupBy('DateTaken').agg(
            F.count('Detached Price').alias('Detached Count'),
            F.count('Semi Detached Price').alias('Semi Detached Count'),
            F.count('Terraced Price').alias('Terraced Count'),
            F.count('Flat Price').alias('Flat Count')
        ).orderBy(F.col('DateTaken'))

        df_count.show(1000, False)

        # Plotting
        p_dfm = df4.toPandas()
          # Display the entire DataFrame as a string
        print(p_dfm.to_string())

        plt.figure(figsize=(12, 6))

        # Plot average house price
        plt.plot(p_dfm['DateTaken'], p_dfm['Average Price'], label='Average Price', marker='o')
        # Adding titles and labels
        plt.xlabel('Date')
        plt.ylabel('Price (£)')
        plt.title(f"UK Registered Average House Prices in {RegionName} From {StartDate} until {EndDate}")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(PNG_DIR, f'{new_string}_AverageHousePrice_from_{StartDate}_until_{EndDate}.png'))


        plt.figure(figsize=(12, 6))

        # Plot each house type price
        plt.plot(p_dfm['DateTaken'], p_dfm['Average Price'], label='Average Price', marker='o')
        plt.plot(p_dfm['DateTaken'], p_dfm['Detached Price'], label='Detached Price', marker='o')
        plt.plot(p_dfm['DateTaken'], p_dfm['Semi Detached Price'], label='Semi Detached Price', marker='o')
        plt.plot(p_dfm['DateTaken'], p_dfm['Terraced Price'], label='Terraced Price', marker='o')
        plt.plot(p_dfm['DateTaken'], p_dfm['Flat Price'], label='Flat Price', marker='o')

        # Adding titles and labels
        plt.xlabel('Date')
        plt.ylabel('Price (£)')
        plt.title(f"UK Registered House Prices in {RegionName} From {StartDate} until {EndDate}")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(PNG_DIR, f'{new_string}_CombinedPricePaid_from_{StartDate}_until_{EndDate}.png'))

        # let us get the average of property values here
        df = pd.DataFrame(self.sdata)
        # Convert DateTaken column to datetime format
        df['DateTaken'] = pd.to_datetime(df['DateTaken'])
        # Set DateTaken as the index
        df.set_index('DateTaken', inplace=True)
        # Calculate the average for each property type (across entire index)
        property_averages = df.mean(axis=0)
        # Create a DataFrame from the averages Series
        property_averages_df = pd.DataFrame(property_averages, columns=['Price'])
        property_averages_df.reset_index(inplace=True)  # Set index as 'Property Type' column
        property_averages_df.rename(columns={'index': 'Property Type'}, inplace=True)
        # Formatting function for currency display
        def format_currency(price):
          return f"£{price:,.2f}"

        
class PropertyPricePredictor:
    def __init__(self, data):
        self.scaler = StandardScaler()
        self.linear_model = LinearRegression()
        self.gb_model = GradientBoostingRegressor()
        self.data = data

    def train_and_predict(self, RegionName, column_name):
        """
        Workflow for Linear Regression, Gradient Boosting Regressor, and ARIMA
        """
        new_string = RegionName.replace(" ", "")
        # Create DataFrame
        df = pd.DataFrame(self.data)

        # Check if 'DateTaken' is in the DataFrame
        if 'DateTaken' not in df.columns:
            raise KeyError("'DateTaken' column not found in the data")

        # Convert DateTaken to datetime
        df['DateTaken'] = pd.to_datetime(df['DateTaken'])

        # Convert DateTaken to a numerical format
        df['year_month'] = df['DateTaken'].dt.year * 100 + df['DateTaken'].dt.month

        # Extract feature and target columns
        X = df[['year_month']].values
        y = df[column_name]

        # Scale the features
        X_scaled = self.scaler.fit_transform(X)

        # Train the models
        self.linear_model.fit(X_scaled, y)
        self.gb_model.fit(X_scaled, y)

        # Prepare future dates for prediction
        future_dates = pd.date_range(start='2024-01-01', periods=12, freq='MS')
        future_year_month = future_dates.year * 100 + future_dates.month
        future_year_month_scaled = self.scaler.transform(future_year_month.values.reshape(-1, 1))

        # Predict future prices using Linear Regression and Gradient Boosting
        linear_predictions = self.linear_model.predict(future_year_month_scaled)
        gb_predictions = self.gb_model.predict(future_year_month_scaled)

        # Fit the SARIMAX model
        arima_model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        arima_results = arima_model.fit()

        # Forecast for the next 12 months using integer indexing
        forecast = arima_results.get_forecast(steps=12)
        arima_predictions = forecast.predicted_mean
        arima_conf_int = forecast.conf_int()

        # Combine predictions
        combined_predictions = (linear_predictions + gb_predictions + arima_predictions) / 3

        # Print predictions
        display_name = column_name.replace(" Price", "")
        print(f"Property type: {display_name}")
        for date, lin_pred, gb_pred, arima_pred, combined_pred in zip(future_dates, linear_predictions, gb_predictions, arima_predictions, combined_predictions):
            print(f"Predicted {column_name} for {date.strftime('%Y-%m')} - Linear: {lin_pred:.2f}, Gradient Boosting: {gb_pred:.2f}, ARIMA: {arima_pred:.2f}, Combined: {combined_pred:.2f}")

        # Plot historical data
        plt.figure(figsize=(14, 7))
        plt.plot(df['DateTaken'], df[column_name], label='Historical Data')

        # Plot predictions
        plt.plot(future_dates, linear_predictions, label='Linear Regression Predictions', linestyle='--', marker='o')
        plt.plot(future_dates, gb_predictions, label='Gradient Boosting Predictions', linestyle='--', marker='x')
        plt.plot(future_dates, arima_predictions, label='ARIMA Predictions', linestyle='--', marker='d')
        plt.plot(future_dates, combined_predictions, label='Combined Predictions', linestyle='--', marker='^')

        # Add confidence intervals for ARIMA predictions
        plt.fill_between(future_dates, arima_conf_int.iloc[:, 0], arima_conf_int.iloc[:, 1], color='k', alpha=0.1, label='ARIMA Confidence Interval')

        # Customize the plot
        plt.xlabel('Date')
        plt.ylabel(column_name)
        plt.title(f'{RegionName} Historical and Predicted {column_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(PNG_DIR, f'{new_string}_combined_predictions_with_confidence_intervals.png'))

        # Show the plot
        #plt.show()

def main():
    appName = "UK House prices distribution"
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName(appName) \
        .enableHiveSupport() \
        .getOrCreate()
        # Set the configuration
    
     # Set the log level to ERROR to reduce verbosity
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    
    # Load data from Hive tables
    house_df = load_hive_data(spark)
    # Load data from JSON file
    data = load_json_data()
    
    # Create DataFrameProcessor instance
    df_processor = DataFrameProcessor(spark, data)

    total_data_count = house_df.count()
    print(f"Total data count: {total_data_count}")
  
    df_processor.analyze_uk_ownership(house_df)  
    
    RegionName = 'Kensington and Chelsea'
    new_string = RegionName.replace(" ", "")
        
    # Create PropertyPricePredictordata instance
    predictor = PropertyPricePredictor(data)  
    predictor.train_and_predict(RegionName , 'Average Price')
    
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
