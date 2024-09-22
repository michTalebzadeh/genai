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


class DataFrameProcessor:
    def __init__(self, spark, data):
        self.spark = spark
        self.data = data

    # Load data from Hive table
    def load_data(self):
        DSDB = "DS"
        tableName = "ukhouseprices"
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
        house_df.printSchema()
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
            print(f"skipping analyze_single_column")
            #analysis_results[column_name] = DataFrameProcessor.analyze_single_column(self, dataframe, column_name, use_hll, num_buckets)

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
            # Calculate the average value of the specified column
            # avg_value_unrounded = dataframe.select(avg(col(column_name))).collect()[0][0]

            # Round the collected average value
            # avg_value = round(avg_value_unrounded, 2)

            #avg_value = round(dataframe.select(avg(col(column_name))).collect()[0][0],2)

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
                        column_name = "AveragePrice"
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

    # Function to train the model and predict future prices
    
    def train_and_predict(self, data, scaler, model, column_name):
        
        """
        Combined Workflow
        Here how these components fit into the workflow:

        Standardization:
        The scaler is fitted to the training data to compute the mean and standard deviation.
        The training data is transformed (standardized) using the fitted scaler.
        This ensures that the data used to train the model has a mean of 0 and a standard deviation of 1.

        Model Training:
        The model is fitted to the standardized training data.
        The linear regression algorithm learns the relationship between the input features and the target variable.

        Prediction:
        New data is also standardized using the same scaler.
        The standardized new data is passed to the trained model to make predictions.
        """
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Check if 'DateTaken' is in the DataFrame
        if 'DateTaken' not in df.columns:
            raise KeyError("'DateTaken' column not found in the data")
        
        # Convert DateTaken to datetime
        df['DateTaken'] = pd.to_datetime(df['DateTaken'])

        # Convert DateTaken to a numerical format
        df['year_month'] = df['DateTaken'].dt.year * 100 + df['DateTaken'].dt.month
        
        # Extract feature and target columns
        X = df[['year_month']].values
        y = df[column_name].values

        # Scale the features
        X_scaled = scaler.fit_transform(X)

        # Train the model
        model.fit(X_scaled, y)

        # Prepare future dates for prediction
        future_dates = pd.date_range(start='2024-01-01', periods=12, freq='MS')
        future_year_month = future_dates.year * 100 + future_dates.month
        future_year_month_scaled = scaler.transform(future_year_month.values.reshape(-1, 1))

        # Predict future prices
        predictions = model.predict(future_year_month_scaled)

        # Print predictions
        display_name = column_name.replace(" Price", "")
        print(f"Property type is {display_name}")
        for date, pred in zip(future_dates, predictions):
            print(f"Predicted linear regression {column_name} for {date.strftime('%Y-%m')}: {pred:.2f}")

        # Plot historical data
        plt.figure(figsize=(12, 6))
        plt.plot(df['DateTaken'], df[column_name], label='Historical Data')

        # Plot predictions
        plt.plot(future_dates, predictions, label='Predictions', linestyle='--', marker='o')

        # Customize the plot
        plt.xlabel('Date')
        plt.ylabel(column_name)
        plt.title(f'Historical and Predicted {column_name}')
        plt.legend()
        plt.grid(True)

        # Show the plot
        # plt.show()

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
        #StartDate = '2023-01-01'
        #EndDate   = '2023-12-01'
        StartDate = '2020-01-01'
        EndDate   = '2023-12-31'
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
        plt.savefig(f'{new_string}_AveragePricePaid.png')
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
        plt.savefig(f'{new_string}_AverageHousePrice_from_{StartDate}_until_{EndDate}.png')


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
        plt.savefig(f'{new_string}_CombinedPricePaid_from_{StartDate}_until_{EndDate}.png')

        """
        Lorentzian section
        """
        print(f"\nLorentzian fit for Property prices for London Borough of {RegionName} from {StartDate} till {EndDate}\n")

        # Sample Data
        # Load data from JSON file
        with open('/home/hduser/dba/bin/python/genai/data/property_data.json', 'r') as f:
           data = json.load(f)
        
        # Create DataFrame
        df = pd.DataFrame(data)

        # Convert DateTaken to datetime
        df['DateTaken'] = pd.to_datetime(df['DateTaken'])

        # Convert DateTaken to a numerical format
        df['year_month'] = df['DateTaken'].dt.year * 100 + df['DateTaken'].dt.month
        
        # Convert DateTaken to datetime in pandas DataFrame
        df['DateTaken'] = pd.to_datetime(df['DateTaken'])

        
        # Convert pandas DataFrame to Spark DataFrame
        sdf = self.spark.createDataFrame(df)
        # Explicitly cast DateTaken to IntegerType
        #sdf = sdf.withColumn("DateTaken", sdf["DateTaken"].cast(IntegerType()))

        # Write the DataFrame to Hive
        self.write_summary_data(sdf, "prices_from_2020_2023")

        # Plotting and fitting
        property_columns = ['Average Price', 'Detached Price', 'Semi Detached Price', 'Terraced Price', 'Flat Price']
        model = LorentzianModel()

        plt.figure(figsize=(14, 7))

        for col in property_columns:
            params = model.guess(df[col], x=df['year_month'])
            result = model.fit(df[col], params, x=df['year_month'])
            
            plt.plot(df['year_month'], df[col], 'o', label=col)
            plt.plot(df['year_month'], result.best_fit, '-', label=f'{col} fit')

        plt.legend(loc='upper left')
        plt.xlabel("Year/Month", fontsize=14)
        plt.ylabel("Price", fontsize=14)
        plt.title("Monthly Property Price Change in Kensington and Chelsea", fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True)

        # Save the plot
        plt.savefig(f'{new_string}_LorentzianFit_from_{StartDate}_until_{EndDate}.png')
        #plt.show()

        # Print fit report
        print(result.fit_report()) 
     
        # Extract numeric_dates and average_price for fitting
        numeric_dates = df['year_month'].values
        average_price = df['Average Price'].values

        # Fit a polynomial model (degree 3 for more flexibility)
        poly_coefficients = np.polyfit(numeric_dates, average_price, 3)
        poly_model = np.poly1d(poly_coefficients)

        # Predict values up to the end of 2024
        future_dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='MS')
        future_numeric_dates = future_dates.year * 100 + future_dates.month
        predicted_future_prices = poly_model(future_numeric_dates)
        """
        # Plot the data and the polynomial fit
        plt.figure(figsize=(10, 6))
        plt.scatter(df['DateTaken'], df['Average Price'], label='Average Price', color='blue')
        plt.plot(df['DateTaken'], poly_model(numeric_dates), label='Polynomial Fit (degree 3)', color='orange')

        # Add future predictions to the plot
        future_dates_full = pd.date_range(start=df['DateTaken'].min(), end='2024-12-01', freq='MS')
        future_numeric_dates_full = future_dates_full.year * 100 + future_dates_full.month
        predicted_full_prices = poly_model(future_numeric_dates_full)

        plt.plot(future_dates_full, predicted_full_prices, label='Predicted Prices', color='green', linestyle='--')

        plt.xlabel('DateTaken')
        plt.ylabel('Average Price')
        plt.title('Average Price with Polynomial Fit (degree 3) and Predictions to End of 2024')
        plt.legend()
        plt.grid(True)
        #plt.show()

        # Save the plot
        plt.savefig(f'{new_string}_PolynomialFitDegree3_Predictions_till_end_of_2024.png')
        """

        """
        combined polynomial
        """
        print(f"\nPolynomial degree 3 fit for Property prices for London Borough of {RegionName} from {StartDate} till {EndDate}\n")

        df = pd.DataFrame(data)
        df['DateTaken'] = pd.to_datetime(df['DateTaken'])
        df.set_index('DateTaken', inplace=True)

        # Convert dates to numerical values (e.g., months since start)
        df['Time'] = np.arange(len(df))

        # Function to fit a polynomial and make predictions
        def fit_and_predict(property_type):
            # Fit a cubic polynomial
            coefs = np.polyfit(df['Time'], df[property_type], 3)
            p = np.poly1d(coefs)

            # Extend the time values for prediction (until end of 2024)
            future_time = np.arange(len(df), len(df) + 12)
            future_dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
            future_predictions = p(future_time)

            # Combine historical and predicted data for plotting
            historical_dates = df.index
            historical_prices = df[property_type].values
            all_dates = historical_dates.append(future_dates)
            all_prices = np.concatenate([historical_prices, future_predictions])

            return historical_dates, historical_prices, future_dates, future_predictions, p

        # Plotting function
        def plot_predictions():
            property_types = ['Average Price', 'Detached Price', 'Semi Detached Price', 'Terraced Price', 'Flat Price']
            
            plt.figure(figsize=(14, 8))

            for prop in property_types:
                hist_dates, hist_prices, pred_dates, pred_prices, _ = fit_and_predict(prop)
                display_name = prop.replace(" Price", "")
                print(f"Property type is {display_name}")
                for date, pred in zip(pred_dates, pred_prices):
                    print(f"Predicted polynomial fit {prop} for {date.strftime('%Y-%m')}: {pred:.2f}")
     
                plt.plot(hist_dates, hist_prices, label=f'{prop} (Historical)', linestyle='-')
                plt.plot(pred_dates, pred_prices, label=f'{prop} (Predicted)', linestyle='--')

            plt.xlabel('Date')
            plt.ylabel('Price (£)')
            plt.title('Property Price Trends and Predictions with PolynomialFit Degree 3 in Kensington and Chelsea')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{new_string}_PolynomialFitDegree3_Predictions_forallProperties_till_end_of_2024.png')
            #plt.show()

        # Plot the predictions
        plot_predictions()

        # let us get the average of property values here
        df = pd.DataFrame(data)
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

        
        # Apply formatting function to the 'Price' column
        property_averages_df['Price'] = property_averages_df['Price'].apply(format_currency)
        # Print the DataFrame as a table
        print(f"\nAverage Prices for Each Property Type in {RegionName} from {StartDate} till {EndDate}\n")
        print(property_averages_df.to_string(index=False))
        

        """
        Linear regression fit here
        """
        print(f"\nLinear regression fit for Property prices for London Borough of {RegionName} from {StartDate} till {EndDate}\n")

        # Create DataFrame
        df = pd.DataFrame(data)

        # Convert DateTaken to datetime
        df['DateTaken'] = pd.to_datetime(df['DateTaken'])

        # Set DateTaken as index
        df.set_index('DateTaken', inplace=True)

        # Convert dates to numerical values (e.g., months since start)
        df['Time'] = np.arange(len(df))

        # Convert DateTaken to a numerical format for model training
        df['year_month'] = df.index.year * 100 + df.index.month
             
        # Initialize scaler and model
        scaler = StandardScaler()
        """
        Purpose:

        The StandardScaler class from sklearn.preprocessing is used to standardize features by removing the mean and scaling to unit variance. Standardizing data is a common preprocessing step before training a machine learning model, especially when features have different units or different scales.
        Why Standardize?

        Improved Performance: Many machine learning algorithms perform better when the features are standardized, as they tend to converge faster.
        Equal Weighting: Standardizing ensures that all features contribute equally to the model, preventing features with larger scales from dominating the learning process.
        How it works:

        StandardScaler computes the mean and standard deviation for each feature in the training data.
        It then uses these statistics to transform the data by subtracting the mean and dividing by the standard deviation for each feature.
        LinearRegression
        """
        model = LinearRegression()
        """
        Purpose:

        The LinearRegression class from sklearn.linear_model is used to create a linear regression model. Linear regression is a basic and commonly used type of predictive analysis.
        How it works:

        Fitting the Model: The linear regression model finds the line (in higher dimensions, a hyperplane) that best fits the data by minimizing the sum of the squares of the vertical distances of the points from the line (ordinary least squares).
        Prediction: Once the model is trained, it can be used to predict the output for new input data by applying the learned linear relationship.
        """
        # List of property types to predict
        property_types = ['Average Price', 'Detached Price', 'Semi Detached Price', 'Terraced Price', 'Flat Price']

        # Train and predict for each property type
        for property_type in property_types:
            self.train_and_predict(data, scaler, model, property_type)

        # Plotting function
        def plot_predictions_lr():
            property_types = ['Average Price', 'Detached Price', 'Semi Detached Price', 'Terraced Price', 'Flat Price']
            
            plt.figure(figsize=(14, 8))

            for prop in property_types:
                hist_dates, hist_prices, pred_dates, pred_prices, _ = fit_and_predict(prop)
                plt.plot(hist_dates, hist_prices, label=f'{prop} (Historical)', linestyle='-')
                plt.plot(pred_dates, pred_prices, label=f'{prop} (Predicted)', linestyle='--')

            plt.xlabel('Date')
            plt.ylabel('Price (£)')
            plt.title('Property Price Trends and Predictions with linear regression in Kensington and Chelsea')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{new_string}_LinearRegression_Predictions_forallProperties_till_end_of_2024.png')
            #plt.show()

        # Plot the predictions
        plot_predictions_lr()

        print(f"\nThe actual data and Polynomial plus Linear regression fit for Property prices for London Borough of {RegionName} from {StartDate} till {EndDate}\n")

        df = pd.DataFrame(self.data)

        # Convert DateTaken to datetime
        df['DateTaken'] = pd.to_datetime(df['DateTaken'])
        # Set DateTaken as index
        df.set_index('DateTaken', inplace=True)

        # Convert dates to numerical values (e.g., months since start)
        df['Time'] = np.arange(len(df))

        # Convert DateTaken to a numerical format for model training
        df['year_month'] = df.index.year * 100 + df.index.month

        # Predictions for polynomial regression
        poly_preds = {
            'Average Price': [1228172.64, 1215534.57, 1203800.15, 1193100.66, 1183567.42, 
                            1175331.72, 1168524.88, 1163278.17, 1159722.92, 1157990.41, 
                            1158211.95, 1160518.85],
            'Detached Price': [3371391.64, 3291572.30, 3210101.43, 3127223.23, 3043181.86, 
                            2958221.52, 2872586.36, 2786520.58, 2700268.35, 2614073.84, 
                            2528181.24, 2442834.72],
            'Semi Detached Price': [3516073.76, 3453419.90, 3390806.91, 3328545.36, 3266945.87, 
                                    3206319.01, 3146975.39, 3089225.59, 3033380.22, 2979749.86, 
                                    2928645.11, 2880376.56],
            'Terraced Price': [2316589.62, 2275867.72, 2237407.84, 2201404.22, 2168051.12, 
                            2137532.77, 2110043.44, 2085777.37, 2064928.81, 2047682.00, 
                            2034211.18, 2024700.61],
            'Flat Price': [1069051.52, 1053922.05, 1040202.91, 1027983.39, 1017352.80, 
                        1008400.42, 1001215.54, 996887.48, 993414.52, 991485.95, 
                        991091.09, 992219.22]
        }

        # Predictions for linear regression
        lin_preds = {
            'Average Price': [1171866.88, 1172637.91, 1173408.94, 1174179.97, 1174951.00, 
                            1175722.03, 1176493.06, 1177264.09, 1178035.12, 1178806.15, 
                            1179577.18, 1180348.21],
            'Detached Price': [3466704.52, 3466108.47, 3465512.42, 3464916.37, 3464320.32, 
                            3463724.27, 3463128.22, 3462532.17, 3461936.12, 3461340.07, 
                            3460744.02, 3460147.97],
            'Semi Detached Price': [3551445.98, 3549190.68, 3546935.38, 3544680.08, 3542424.78, 
                                    3540169.48, 3537914.18, 3535658.88, 3533403.58, 3531148.28, 
                                    3528892.98, 3526637.68],
            'Terraced Price': [2328687.17, 2326431.55, 2324175.93, 2321920.31, 2319664.69, 
                            2317409.07, 2315153.45, 2312897.83, 2310642.21, 2308386.59, 
                            2306130.97, 2303875.35],
            'Flat Price': [1107063.69, 1105750.52, 1104437.35, 1103124.18, 1101811.01, 
                        1100497.84, 1099184.67, 1097871.50, 1096558.33, 1095245.16, 
                        1093931.99, 1092618.82]
        }

        # Generating dates for predictions (2024)
        from datetime import datetime
        pred_dates = [datetime(2024, month, 1) for month in range(1, 13)]

        # Plotting function
        def plot_predictions(new_string, prop_type, actual, poly_pred, lin_pred, title):
            plt.figure(figsize=(14, 8))
            plt.plot(df.index, df[prop_type], label='Actual Prices', marker='o')
            plt.plot(pred_dates, poly_pred[prop_type], label='Polynomial Regression', linestyle='--', marker='x')
            plt.plot(pred_dates, lin_pred[prop_type], label='Linear Regression', linestyle='-.', marker='s')
            plt.title(title)
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{new_string}_Polynomial_and_LinearRegression_Predictions_for_{prop_type}_till_end_of_2024.png')
            plt.close()

        # Plot for each property type
        plot_predictions(new_string, 'Average Price', data['Average Price'], poly_preds, lin_preds, f'{RegionName} Average Price Predictions')
        plot_predictions(new_string, 'Detached Price', data['Detached Price'], poly_preds, lin_preds, f'{RegionName} Detached Price Predictions')
        plot_predictions(new_string, 'Semi Detached Price', data['Semi Detached Price'], poly_preds, lin_preds, f'{RegionName} Semi Detached Price Predictions')
        plot_predictions(new_string, 'Terraced Price', data['Terraced Price'], poly_preds, lin_preds, f'{RegionName} Terraced Price Predictions')
        plot_predictions(new_string, 'Flat Price', data['Flat Price'], poly_preds, lin_preds, f'{RegionName} Flat Price Predictions')


        """
        print(f"\nEnhanced prices for Property prices for {RegionName} for 2023\n")

        data = {}
        # Data for Kensington and Chelsea (2023)
        if (RegionName == "Kensington and Chelsea"):
            data = {
                'DateTaken': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', 
                            '2023-06-01', '2023-07-01', '2023-08-01', '2023-09-01', '2023-10-01', 
                            '2023-11-01', '2023-12-01'],
                'Average Price': [1364872.0, 1350849.0, 1384497.0, 1334814.0, 1363774.0, 
                                1400045.0, 1376043.0, 1366026.0, 1315895.0, 1375884.0, 
                                1268656.0, 1199227.0],
                'Detached Price': [4021058.0, 3956980.0, 4086739.0, 3917525.0, 3954089.0, 
                                4041221.0, 3980686.0, 4016724.0, 3854223.0, 3982877.0, 
                                3628941.0, 3301387.0],
                'Semi Detached Price': [4035780.0, 3984685.0, 4111878.0, 3953592.0, 4021585.0, 
                                        4125853.0, 4067773.0, 4082121.0, 3923799.0, 4069382.0, 
                                        3719860.0, 3420758.0],
                'Terraced Price': [2661903.0, 2620349.0, 2680795.0, 2572512.0, 2632418.0, 
                                2716269.0, 2671069.0, 2665975.0, 2561674.0, 2662450.0, 
                                2436004.0, 2259569.0],
                'Flat Price': [1163575.0, 1153696.0, 1182815.0, 1142170.0, 1166603.0, 
                            1195780.0, 1174990.0, 1164021.0, 1122277.0, 1176026.0, 
                            1087318.0, 1034924.0],
                'Detached Count': [1]*12,
                'Semi Detached Count': [1]*12,
                'Terraced Count': [1]*12,
                'Flat Count': [1]*12
            }
        elif (RegionName == 'City of Westminster'):
            # Data for City of Westminster (2023)
                data = {
                    'DateTaken': pd.date_range(start='2023-01-01', periods=12, freq='MS'),
                    'Average Price': [1184268.0, 1168177.0, 1149979.0, 1137278.0, 1152641.0, 1162850.0, 1108294.0, 1064033.0, 1020396.0, 1025787.0, 998272.0, 951032.0],
                    'Detached Price': [3707611.0, 3633928.0, 3599118.0, 3540352.0, 3551701.0, 3574563.0, 3410057.0, 3325852.0, 3174199.0, 3149070.0, 3021736.0, 2763080.0],
                    'Semi Detached Price': [3117125.0, 3056018.0, 3022174.0, 2974380.0, 3022563.0, 3055089.0, 2906638.0, 2805186.0, 2674650.0, 2664691.0, 2565803.0, 2376467.0],
                    'Terraced Price': [2086282.0, 2048652.0, 2010749.0, 1981121.0, 2003025.0, 2029797.0, 1941977.0, 1879779.0, 1802227.0, 1800474.0, 1737198.0, 1624515.0],
                    'Flat Price': [1102861.0, 1088771.0, 1072182.0, 1061082.0, 1075891.0, 1084655.0, 1033156.0, 990393.0, 949923.0, 956120.0, 931979.0, 891156.0],
                    'Detached Count': [1] * 12,
                    'Semi Detached Count': [1] * 12,
                    'Terraced Count': [1] * 12,
                    'Flat Count': [1] * 12
                }
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['DateTaken'] = pd.to_datetime(df['DateTaken'])

        # Plotting
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Price lines
        ax1.plot(df['DateTaken'], df['Average Price'], label='Average Price', color='blue')
        ax1.plot(df['DateTaken'], df['Detached Price'], label='Detached Price', color='orange')
        ax1.plot(df['DateTaken'], df['Semi Detached Price'], label='Semi Detached Price', color='green')
        ax1.plot(df['DateTaken'], df['Terraced Price'], label='Terraced Price', color='red')
        ax1.plot(df['DateTaken'], df['Flat Price'], label='Flat Price', color='purple')

        # Count lines
        ax2 = ax1.twinx()
        ax2.plot(df['DateTaken'], df['Detached Count'], label='Detached Count', color='orange', linestyle='dotted')
        ax2.plot(df['DateTaken'], df['Semi Detached Count'], label='Semi Detached Count', color='green', linestyle='dotted')
        ax2.plot(df['DateTaken'], df['Terraced Count'], label='Terraced Count', color='red', linestyle='dotted')
        ax2.plot(df['DateTaken'], df['Flat Count'], label='Flat Count', color='purple', linestyle='dotted')

        # Labels and title
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax2.set_ylabel('Transaction Count')
        plt.title(f'House Prices and Transaction Counts in {RegionName} for 2023')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{new_string}_EnhancedPricePaid_in_2023.png')
        #plt.show()  
        """    
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
        plt.savefig(f'{new_string}_combined_predictions_with_confidence_intervals.png')

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
    
    # Load data from JSON file
    with open('/home/hduser/dba/bin/python/genai/data/property_data.json', 'r') as f:
       data = json.load(f)
    
    # Create DataFrameProcessor instance
    df_processor = DataFrameProcessor(spark, data)

    # Load data from Hive tables
    house_df = df_processor.load_data()

    total_data_count = house_df.count()
    print(f"Total data count: {total_data_count}")
  
    df_processor.analyze_uk_ownership(house_df)  

    #column_to_analyze = None
    column_to_analyze = 'AveragePrice'
  
    if column_to_analyze is None:
        # No column is defined, doing the analysis for all columns in the dataframe
        print(f"\nNo column is defined so doing do the analysis for all columns in the dataframe")
        analysis_results = df_processor.analyze_column(house_df)
    else:
        # Do the analysis for the specific column
        #print(f"\n Doing analysis for column {column_to_analyze}")
        analysis_results = df_processor.analyze_column(house_df, column_to_analyze)
    
    # Print the analysis results for the requested column(s) only
    # print(analysis_results)
    # Convert analysis_results to JSON format
    analysis_results_json = json.dumps(analysis_results, indent=4)

    # Print the JSON-formatted analysis results
    ###print(f"\nJson formatted output\n")
    ###print(analysis_results_json)
    # Example usage

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
