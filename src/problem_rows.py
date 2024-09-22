# Initialise imports
import sys
import os
import datetime
import sysconfig
from pyspark.sql import SparkSession
from pyspark.sql.functions import ( 
    col, 
    count, 
    countDistinct,
    format_number,
    udf,
    lit, 
    datediff, 
    when, 
    isnan, 
    expr, 
    min as spark_min,
    max as spark_max, 
    avg, 
    udf, 
    rand,
    round as spark_round
)
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
import numpy as np
from pyspark.sql.window import Window
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import layers, models, losses
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError, binary_crossentropy
from tensorflow.keras.models import Model
import torch
import sklearn 
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import json
import seaborn as sns
from tensorflow.keras.layers import Input, Dense, Lambda
from sklearn.preprocessing import StandardScaler
import re
import locale
import warnings
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adagrad
from pyspark.sql.functions import regexp_replace


# Global variable for PNG directory
PNG_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated")

def create_png_dir():
    global PNG_DIR
    # Define the directory path relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    PNG_DIR = os.path.join(script_dir, "designs", "png_files")
    # Create the directory if it does not exist
    if not os.path.exists(PNG_DIR):
        os.makedirs(PNG_DIR)

def visualize_distribution(df, synthetic_df, feature):
    plt.figure(figsize=(12, 6))
    sns.histplot(df[feature], color='blue', label='Original', kde=True)
    sns.histplot(synthetic_df[feature], color='red', label='Synthetic', kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(PNG_DIR, f'Distribution_of_{feature}.png'))
    plt.close()

class DataFrameProcessor:
    def __init__(self, spark, latent_dim):
        self.spark = spark
        self.latent_dim = latent_dim
        self.vae = None  # VAE model will be built later once input_dim is known
    
    def define_tableschema(self):
        tableSchema = StructType([
            StructField("TitleNumber", StringType(), nullable=True),
            StructField("Tenure", StringType(), nullable=True),
            StructField("PricePaid", FloatType(), nullable=True),
            # Add other fields as necessary
            StructField("is_fraud", IntegerType(), nullable=False),
        ])
        return tableSchema

    def load_data(self):
        DSDB = "DS"
        tableName = "problem_rows"
        fullyQualifiedTableName = f"{DSDB}.{tableName}"

        # Check if the table exists
        if self.spark.sql(f"SHOW TABLES IN {DSDB} LIKE '{tableName}'").count() == 1:
            # Compute table statistics
            self.spark.sql(f"ANALYZE TABLE {fullyQualifiedTableName} COMPUTE STATISTICS")
            
            # Get total row count
            rows = self.spark.sql(f"SELECT COUNT(1) FROM {fullyQualifiedTableName}").collect()[0][0]
            print(f"\nTotal number of rows in source table {fullyQualifiedTableName} is {rows}\n")
            
            # Get row count for specific condition
            rows_kc = self.spark.sql(f"SELECT COUNT(1) FROM {fullyQualifiedTableName} WHERE DISTRICT = 'KENSINGTON AND CHELSEA' AND PRICEPAID > 0").collect()[0][0]
            print(f"\nTotal number of rows in Kensington and Chelsea with pricepaid > 0 is {rows_kc}\n")
        else:
            print(f"No such table {fullyQualifiedTableName}")
            sys.exit(1)

        # Read data from the Hive table
        house_df = self.spark.sql(f"SELECT * FROM {fullyQualifiedTableName} -- WHERE pricepaid is NOT NULL")
        
        # Sanitize data using sanitize_string_columns method
        house_df = self.sanitize_string_columns(house_df)
        
        return house_df

    def write_data(self, df, tableName) -> None:
        DSDB = "DS"
        fullyQualifiedTableName = f"{DSDB}.{tableName}"
        try:
            # Define a function to sanitize column names
            def sanitize_column_name(column_name, row_index):
                sanitized_name = (
                    column_name.replace(" ", "_")    # Replace spaces with underscores
                                .replace(",", "")     # Remove commas
                                .replace(".", "")     # Remove periods
                                .replace("(", "")     # Remove opening parentheses
                                .replace(")", "")     # Remove closing parentheses
                                .replace("/", "_")    # Replace slashes with underscores
                                .replace("-", "_")    # Replace hyphens with underscores
                                .replace("&", "and")  # Replace ampersands with 'and'
                                .replace("%", "percent")  # Replace percentage signs with 'percent'
                                .replace("#", "number")   # Replace hash signs with 'number'
                                .replace("@", "at")   # Replace at signs with 'at'
                                .replace("!", "")     # Remove exclamation marks
                                .replace("?", "")     # Remove question marks
                                .replace(":", "")     # Remove colons
                                .replace(";", "")     # Remove semicolons
                )
                return sanitized_name

            # Apply the sanitization function to all column names
            # Create a dictionary mapping original to sanitized column names
            sanitized_columns = {col: sanitize_column_name(col, row_index) for col, row_index in zip(df.columns, range(df.count()))}
            df_sanitized = df.select([F.col(col).alias(sanitized_columns[col]) for col in sanitized_columns])     
            df_sanitized.show(100, False)     
    
            # Write the sanitized dataframe to the Hive table
            df_sanitized.write.mode("overwrite").saveAsTable(fullyQualifiedTableName)
            print(f"Dataframe data written to table: {fullyQualifiedTableName}")
        except Exception as e:
            print(f"Error writing data: {e}")

    def cleanup_columns(self, df):
        print("\n CPC1\n")
        df.printSchema()
        print(f"\n df.count() is {df.count()}\n")

        def sanitize_column_name(column_name,row_index):
            print(f"\n Before: {column_name}")  # Print original column name
            sanitized_name = f"{column_name.replace(' ', '_')}" \
                            .replace(",", "") \
                            .replace(".", "") \
                            .replace("(", "") \
                            .replace(")", "") \
                            .replace("/", "_") \
                            .replace("-", "_") \
                            .replace("&", "and") \
                            .replace("%", "percent") \
                            .replace("#", "number") \
                            .replace("@", "at") \
                            .replace("!", "") \
                            .replace("?", "") \
                            .replace(":", "") \
                            .replace(";", "")
            #
            print(f" After : {sanitized_name}\n")  # Print sanitized column name
            return f"{sanitized_name}"
        # Create a dictionary mapping original to sanitized column names
        sanitized_columns = {col: sanitize_column_name(col, row_index) for col, row_index in zip(df.columns, range(df.count()))}
        # Create a new DataFrame with renamed columns
        #sanitized_columns = {col: sanitize_column_name(col) for col in df.columns}
        cleaned_df = df.select([F.col(col).alias(sanitized_columns[col]) for col in sanitized_columns])
        cleaned_df.printSchema()
        return cleaned_df

    def sanitize_string_columns(self, df):
        # Print the initial schema and row count for reference
        print("\nBefore Data Sanitization:")
        df.printSchema()
        print(f"\nRow Count: {df.count()}\n")

        # Function to sanitize individual string column values
        def sanitize_string_column(col):
            # Trim, lowercase, and remove unwanted characters
            return F.trim(F.lower(F.regexp_replace(col, "[\.,;:!?\(\)\[\]]", "")))

        # Dynamically get the list of all string columns
        columns_to_clean = [field.name for field in df.schema.fields if isinstance(field.dataType, StringType)]

        # Print the columns being sanitized
        print(f"Sanitizing the following string columns: {columns_to_clean}\n")

        # Apply sanitization transformations to each string column
        for col in columns_to_clean:
            df = df.withColumn(col, sanitize_string_column(F.col(col)))

        # Handle nulls or replace empty values with 'unknown' where appropriate
        df = df.fillna({col: 'unknown' for col in columns_to_clean})

        # Print the cleaned schema for verification
        print("\nAfter Data Sanitization:")
        df.printSchema()
        return df
  

    def cleanup_data(self, house_df, categorical_cols):
        # Filter out unwanted rows
        print(f"\n\n house_df size is {house_df.count()}\n")
        house_df = house_df.filter(house_df["Tenure"] != "93347")
        
        house_df = house_df.filter(house_df["TitleNumber"] != "276916") 
        house_df = house_df.filter(house_df["TitleNumber"] != "NGL305001")
        house_df = house_df.filter(house_df["TitleNumber"] != "350274")
        house_df = house_df.filter(house_df["TitleNumber"] != "EX671482")
        house_df = house_df.filter(house_df["TitleNumber"] != "NGL935756")
        house_df = house_df.filter(house_df["TitleNumber"] != "NGL616483")
        house_df = house_df.filter(house_df["TitleNumber"] != "WT103699")
        house_df = house_df.filter(house_df["TitleNumber"] != "SY775740")         
        house_df = house_df.filter(house_df["TitleNumber"] != "NGL942065")      
        house_df = house_df.filter(house_df["TitleNumber"] != "ESX314508")
        house_df = house_df.filter(house_df["TitleNumber"] != "NGL942065")
        house_df = house_df.filter(house_df["TitleNumber"] != "NGL942066")
        house_df = house_df.filter(house_df["TitleNumber"] != "GM330808")
        house_df = house_df.filter(house_df["TitleNumber"] != "NGL915622")
        house_df = house_df.filter(house_df["TitleNumber"] != "BGL55788")
        house_df = house_df.filter(house_df["TitleNumber"] != "BK416904")
        house_df = house_df.filter(house_df["TitleNumber"] != "NGL899979")
        house_df = house_df.filter(house_df["TitleNumber"] != "CL156115")
        #house_df = house_df.filter(~col("proprietor1address1").contains("JE4 8YJ"))
        #house_df = house_df.filter(~col("proprietor1address1").contains("JE1 0BD"))
        
               
        house_df = house_df.filter(~col("Proprietor2Address3").contains("Union Sreet"))
        house_df = house_df.filter(~col("ProprietorName1").contains("M. PANTHER LIMITED")) 
    
        
        # Cast Tenure to string
        house_df = house_df.withColumn("Tenure", col("Tenure").cast("string"))
        
        # Drop unnecessary columns
        house_df = house_df.drop("CompanyRegistrationNo3", "Proprietor4Address3")
        
        # Get columns with nulls and distinct counts
        columns_with_null, null_percentage_dict, columns_distinct_values = self.find_columns_with_null_and_distinct_counts(house_df)
        
        # Calculate percentage of columns that are all null
        total_columns = len(null_percentage_dict)
        null_columns_count = sum(1 for percentage in null_percentage_dict.values() if percentage == 100.0)
        null_columns_percentage = (null_columns_count / total_columns) * 100
        
        # Print columns with missing values
        print(f"\nColumns with missing values:\n")
        for col_name in columns_with_null:
            print(f"{col_name}:\t{null_percentage_dict[col_name]}%")
        
        # Print distinct value counts and mark categorical columns
        print(f"\nColumns and their distinct count:\n")
        print(f"{'Column':<25} {'Distinct Count':<15} {'Categorical Column'}")
        print(f"{'-'*25} {'-'*15} {'-'*18}")
        for col_name in columns_distinct_values:
            is_categorical = "Yes" if col_name in categorical_cols else "No"
            print(f"{col_name:<25} {str(columns_distinct_values[col_name]):<15} {is_categorical}")
        
        # Print summary percentage of columns with all null values
        print(f"\nSummary percentage of columns with no values all null: {null_columns_percentage:.1f}%")
        
        # Filter out columns with 100% null values
        filtered_null_percentage_dict = {col_name: percentage for col_name, percentage in null_percentage_dict.items() if percentage < 100}
        
        return house_df

        from pyspark.sql import functions as F

    def query_house_df(self, house_df):
        # Step 1: Filter data for 'JERSEY'
        filtered_df = house_df.filter(F.col("countryincorporated1") == 'JERSEY')
        
        # Step 2: Standardize addresses
        standardized_df = filtered_df.withColumn(
            "standardized_address",
            F.regexp_replace(
                F.regexp_replace(
                    F.regexp_replace(
                        F.lower(F.col("Proprietor1Address1")),
                        'st[\. ]?helier', 'St Helier'
                    ),
                    'esplandade|esplanande|esplande|esplanade', 'Esplanade'
                ),
                '[\.,]', ''
            )
        ).drop("Proprietor1Address1")

        # Step 3: Count the number of companies for each address
        count_df = standardized_df.groupBy("standardized_address").agg(
            F.count("*").alias("CompaniesIncorpotated")
        )

        # Step 4: Filter for addresses with more than 100 companies
        filtered_count_df = count_df.filter(F.col("CompaniesIncorpotated") > 100)

        # Step 5: Sort by number of companies in descending order
        sorted_df = filtered_count_df.orderBy(F.col("CompaniesIncorpotated").desc())

        # Step 6: Format columns for display
        formatted_df = sorted_df.withColumn(
            "standardized_address",
            F.rpad(F.col("standardized_address"), 53, ' ')
        ).withColumn(
            "CompaniesIncorpotated",
            F.lpad(F.col("CompaniesIncorpotated").cast("string"), 5, ' ')
        )
        
        return formatted_df


    def find_columns_with_null_and_distinct_counts(self, df):
        # Calculate total rows
        total_rows = df.count()
        
        # Create an empty dictionary to store results
        null_value_percentages = {}
        distinct_value_counts = {}
        
        # Iterate over each column
        for col_name in df.columns:
            # Calculate null count and null percentage
            null_count = df.filter(col(col_name).isNull()).count()
            null_percentage = (null_count / total_rows) * 100
            
            # Calculate distinct count
            distinct_count = df.select(col_name).distinct().count()
            
            # Store results
            null_value_percentages[col_name] = round(null_percentage, 1)
            distinct_value_counts[col_name] = distinct_count
        
        # Identify columns with null values
        columns_with_null = [col_name for col_name, percentage in null_value_percentages.items() if percentage > 0]
        
        
        return columns_with_null, null_value_percentages, distinct_value_counts


    def impute_data_vae(self, house_df):
        try:
            print(f"\n\n Started imputing data using VAE model\n")
            print(f"house_df size is {house_df.count()}\n")

           # Convert Spark DataFrame to Pandas DataFrame
            pandas_df = house_df.toPandas()
            print(f"Converted to Pandas DataFrame:\n{pandas_df.head()}\n")

            # Identify columns with non-numeric data
            non_numeric_columns = pandas_df.select_dtypes(include=['object']).columns
            print(f"Non-numeric columns: {non_numeric_columns}\n")

            # Apply label encoding to high-cardinality columns
            label_encoders = {}
            for col in non_numeric_columns:
                if pandas_df[col].nunique() > 100:  # Threshold for high cardinality
                    le = LabelEncoder()
                    pandas_df[col] = le.fit_transform(pandas_df[col].astype(str))
                    label_encoders[col] = le
                    print(f"Applied label encoding to column: {col}")

            print(f"\n Applying one-hot encoding to remaining categorical column\n")
            # Apply one-hot encoding to remaining categorical columns
            pandas_df_encoded = pd.get_dummies(pandas_df, columns=[col for col in non_numeric_columns if col not in label_encoders])
            print(f"After one-hot encoding:\n{pandas_df_encoded.head()}\n")          

            # Prepare data for imputation (replace NaNs with a placeholder value, e.g., 0)
            data_to_impute = pandas_df_encoded.fillna(0)
            print(f"Data prepared for imputation (NaNs replaced with 0):\n{data_to_impute.head()}\n")

            # Scaling data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data_to_impute)
            print(f"Scaled data:\n{scaled_data[:5]}\n")

            # Convert scaled data to numpy array and ensure dtype is float32
            data_array = np.array(scaled_data, dtype=np.float32)
            print(f"Data converted to numpy array. Shape: {data_array.shape}")

            # Check for any remaining NaNs
            if np.isnan(data_array).any():
                raise ValueError("Data contains NaN values after preprocessing.")
            
            print(f"\n Dynamically determine input_dim based on data")
            input_dim = data_array.shape[1]

            # Build VAE model with determined input_dim
            self.vae = self.build_vae_model(input_dim, self.latent_dim)
            print(f"VAE model built with input_dim: {input_dim}, latent_dim: {self.latent_dim}")

            # Impute missing values using VAE
            imputed_data = self.vae.predict(data_array)
            print(f"Imputed data shape: {imputed_data.shape}")

            # Inverse scaling of the data
            imputed_data_unscaled = scaler.inverse_transform(imputed_data)
            print(f"Unscaled imputed data:\n{imputed_data_unscaled[:5]}\n")

            # Convert imputed data back to DataFrame
            imputed_df = pd.DataFrame(imputed_data_unscaled, columns=pandas_df_encoded.columns)
            print(f"Imputed DataFrame:\n{imputed_df.head()}\n")

            # Post-process the DataFrame to convert encoded columns back to their original form
            for col, le in label_encoders.items():
                if col in imputed_df.columns:
                    # Clip the imputed values to the known label range
                    imputed_df[col] = np.clip(imputed_df[col].astype(int), 0, len(le.classes_) - 1)
                    imputed_df[col] = le.inverse_transform(imputed_df[col].astype(int))

            # Convert Pandas DataFrame back to Spark DataFrame
            imputed_spark_df = self.spark.createDataFrame(imputed_df)
            
            print(f"\n\n Data imputation completed successfully\n")
            print(f"imputed_spark_df is {imputed_spark_df.count()}")
            return imputed_spark_df
        
        except Exception as e:
            print(f"Error during data imputation: {e}")
            raise

    def build_vae_model(self, input_dim, latent_dim):
        """Builds a Variational Autoencoder (VAE) model.

        Args:
            input_dim (int): Dimensionality of the input data.
            latent_dim (int): Dimensionality of the latent space.

        Returns:
            keras.models.Model: The compiled VAE model.
        """

        # Define the encoder layers
        inputs = Input(shape=(input_dim,))
        h = Dense(128, activation='relu')(inputs)
        z_mean = Dense(latent_dim)(h)
        z_log_var = Dense(latent_dim)(h)

        # Sampling layer to generate latent codes
        def sampling(args):
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        # Latent code (z) after sampling
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        # Define the decoder layers
        decoder_h = Dense(128, activation='relu')
        decoder_mean = Dense(input_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)

        # Define the VAE model
        vae = Model(inputs, x_decoded_mean)

        # Define the VAE loss
        xent_loss = input_dim * K.binary_crossentropy(inputs, x_decoded_mean)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)
        vae.add_loss(vae_loss)

        # Compile the VAE model with an optimizer
        vae.compile(optimizer='adam')
        
        # Return the compiled VAE model
        return vae

    def perform_analysis(self, df):
        # Convert Spark DataFrame to Pandas DataFrame for analysis
        df_pd = df.toPandas()

        # 1. Descriptive Statistics
        print(f"\n\n Point 1. Descriptive Statistics:\n\n")
        print(df_pd.describe())
        print(df_pd.count())

        # 2. Data Distribution
        def plot_distributions(df):
            for column in df.columns:
                if pd.api.types.is_numeric_dtype(df[column]):
                    # Remove punctuation and convert to lowercase
                    new_string = re.sub(r"[ /]", "", column)
                    plt.figure(figsize=(10, 4))
                    plt.subplot(1, 2, 1)
                    sns.histplot(df[column], kde=True)
                    plt.title(f'Histogram of {column}')
                    plt.subplot(1, 2, 2)
                    sns.boxplot(x=df[column])
                    plt.title(f'Boxplot of {column}')
                    plt.savefig(os.path.join(PNG_DIR, f'{new_string}.png'))
                    plt.close()  # Close the figure to avoid display

        #plot_distributions(df_pd)

        # 3. Correlation Analysis

        print(f"\n\n Point 3. Correlation Analysis\n")
        print("Initial data types df_pd:\n", df_pd.dtypes)

        # Convert columns to numeric, handling non-numeric values
        df_pd = df_pd.apply(pd.to_numeric, errors='coerce')

        # Drop rows with any NaN values, if necessary
        df_pd = df_pd.dropna()

        # Compute the correlation matrix
        correlation_matrix = df_pd.corr()

        print("Correlation matrix:\n", correlation_matrix)
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.savefig(os.path.join(PNG_DIR, 'correlation_matrix.png'))
        plt.close()  # Close the figure to avoid display

        # 4. Missing Values Analysis
        missing_values = df_pd.isnull().sum()
        print("Missing Values:")
        print(missing_values[missing_values > 0])

        # 5. Unique Values Analysis
        print("Unique Values in Categorical Columns:")
        for column in df_pd.columns:
            if pd.api.types.is_object_dtype(df_pd[column]):
                print(f'{column}: {df_pd[column].nunique()} unique values')

        # 6. Visualization of Relationships
        def plot_relationships(df, target_column):
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            for column in numeric_columns:
                if column != target_column:
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(x=df[column], y=df[target_column])
                    plt.title(f'{target_column} vs {column}')
                    plt.savefig(os.path.join(PNG_DIR, f'{target_column}_vs_{column}.png'))
                    plt.close()  # Close the figure to avoid display

        # Assuming you have a target column to analyze relationships
        target_column = 'your_target_column'  # Update this with your actual target column name
        if target_column in df_pd.columns:
            plot_relationships(df_pd, target_column)
        else:
            print(f"Target column '{target_column}' not found in DataFrame.")

        print("Data analysis completed successfully.")

    
class DataFrameReconstructor:
    def __init__(self, spark_session):
        self.spark = spark_session
        self.mappings = {
            'Tenure': {1.0: 'Freehold', 2.0: 'Leasehold'},
            'District': {3.0: 'WALSALL'},
            'County': {4.0: 'WEST MIDLANDS'},
            'Region': {5.0: 'WEST MIDLANDS'},
            'MultipleAddressIndicator': {0.0: 'N', 1.0: 'Y'},
            'ProprietorName1': {6.0: 'LAKESIDE HOLDINGS LIMITED'},
            'ProprietorshipCategory1': {7.0: 'Limited Company or Public Limited Company'},
            'CountryIncorporated1': {8.0: 'GUERNSEY'},
            'Proprietor1Address1': {9.0: 'The Anchorage, La Moye Lane, St Martins, Guernsey, GY4 6BN'},
            'AdditionalProprietorIndicator': {0.0: 'N', 1.0: 'Y'}
            # Add more mappings as needed for other columns
        }
        
        self.schema = {
            'TitleNumber': 'string',
            'Tenure': 'string',
            'PropertyAddress': 'string',
            'District': 'string',
            'County': 'string',
            'Region': 'string',
            'Postcode': 'string',
            'MultipleAddressIndicator': 'string',
            'PricePaid': 'integer',
            'ProprietorName1': 'string',
            'CompanyRegistrationNo1': 'string',
            'ProprietorshipCategory1': 'string',
            'CountryIncorporated1': 'string',
            'Proprietor1Address1': 'string',
            'Proprietor1Address2': 'string',
            'Proprietor1Address3': 'string',
            'ProprietorName2': 'string',
            'CompanyRegistrationNo2': 'string',
            'ProprietorshipCategory2': 'string',
            'CountryIncorporated2': 'string',
            'Proprietor2Address1': 'string',
            'Proprietor2Address2': 'string',
            'Proprietor2Address3': 'string',
            'ProprietorName3': 'string',
            'CompanyRegistrationNo3': 'string',
            'ProprietorshipCategory3': 'string',
            'CountryIncorporated3': 'string',
            'Proprietor3Address1': 'string',
            'Proprietor3Address2': 'string',
            'Proprietor3Address3': 'string',
            'ProprietorName4': 'string',
            'CompanyRegistrationNo4': 'string',
            'ProprietorshipCategory4': 'string',
            'CountryIncorporated4': 'string',
            'Proprietor4Address1': 'string',
            'Proprietor4Address2': 'string',
            'DateProprietorAdded': 'date',
            'AdditionalProprietorIndicator': 'string'
        }

    def reconstruct_dataframe(self, imputed_df):
        reconstructed_df = imputed_df

        for column, col_type in self.schema.items():
            if column in reconstructed_df.columns:
                if col_type == "string" and column in self.mappings:
                    mapping_expr = reconstructed_df[column]
                    for key, value in self.mappings[column].items():
                        mapping_expr = when(col(column) == key, value).otherwise(mapping_expr)
                    reconstructed_df = reconstructed_df.withColumn(column, mapping_expr)
                elif col_type == "integer":
                    reconstructed_df = reconstructed_df.withColumn(column, col(column).cast("integer"))
                elif col_type == "date":
                    # Adjust this as necessary for your date format
                    reconstructed_df = reconstructed_df.withColumn(column, to_date(col(column), 'yyyy-MM-dd'))

        return reconstructed_df

class VAEImputer:
    def __init__(self, input_dim, intermediate_dim, latent_dim, epochs=50, batch_size=128):
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self._build_vae()

    def _build_vae(self):
        # Encoder
        inputs = layers.Input(shape=(self.input_dim,))
        h = layers.Dense(self.intermediate_dim, activation='relu')(inputs)
        z_mean = layers.Dense(self.latent_dim)(h)
        z_log_var = layers.Dense(self.latent_dim)(h)

        # Sampling layer using the reparameterization trick
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = layers.Lambda(sampling)([z_mean, z_log_var])

        # Decoder
        decoder_h = layers.Dense(self.intermediate_dim, activation='relu')
        decoder_mean = layers.Dense(self.input_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)

        # Define the VAE model
        self.vae = Model(inputs, x_decoded_mean)

        # Losses: reconstruction and KL divergence
        reconstruction_loss = losses.binary_crossentropy(inputs, x_decoded_mean)
        reconstruction_loss *= self.input_dim

        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(reconstruction_loss + kl_loss)

        # Add the loss to the model
        self.vae.add_loss(vae_loss)

        # Compile the VAE model
        self.vae.compile(optimizer='rmsprop')

    def fit(self, data):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.vae.fit(data, data, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    def transform(self, data):
        encoder = Model(self.vae.input, self.vae.get_layer(index=2).output)  # Get z_mean layer output
        return encoder.predict(data)

    def generate_samples(self, n_samples=100):
        z_sample = np.random.normal(size=(n_samples, self.latent_dim))
        decoder_input = layers.Input(shape=(self.latent_dim,))
        _h_decoded = self.vae.layers[-2](decoder_input)
        _x_decoded_mean = self.vae.layers[-1](_h_decoded)
        generator = Model(decoder_input, _x_decoded_mean)
        return generator.predict(z_sample)
  
class VAEDataAugmentor:
    def __init__(self, vae_imputer, house_df_imputed):
        self.vae_imputer = vae_imputer
        self.house_df_imputed = house_df_imputed

    def generate_samples(self, n_samples=100):
        return self.vae_imputer.generate_samples(n_samples)

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("House Data Fraud Detection") \
        .enableHiveSupport() \
        .getOrCreate()

    # Set the log level to ERROR to reduce verbosity
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
   
    
    categorical_cols = ["Tenure", "PropertyAddress", "District", "County", "Region", "Postcode", "MultipleAddressIndicator"]
    print(f"\n\n Categorical columns are {categorical_cols}\n\n")
    
    input_dim = 10  # Set appropriately based on your data
    latent_dim = 32  # Set appropriately based on your needs

    print(f"\n Starting processor class\n")
    processor = DataFrameProcessor(spark, latent_dim)
    house_df = processor.load_data()
    house_df.show(20, False)


