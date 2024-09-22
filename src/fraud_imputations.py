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
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import MeanSquaredError
import sklearn 
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import json

class DataFrameProcessor:

    def __init__(self, spark, input_dim, latent_dim):
        self.spark = spark
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.vae = self.build_vae_model()

    def define_tableschema(self):

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
        return tableSchema

    # Load data from Hive tables
    def load_data(self):
        DSDB = "DS"
        tableName = "ocod_full_2024_03" # downloaded fraud table
        fullyQualifiedTableName = f"{DSDB}.{tableName}"
        if self.spark.sql(f"SHOW TABLES IN {DSDB} LIKE '{tableName}'").count() == 1:
            self.spark.sql(f"ANALYZE TABLE {fullyQualifiedTableName} COMPUTE STATISTICS")
            rows = self.spark.sql(f"SELECT COUNT(1) FROM {fullyQualifiedTableName}").collect()[0][0]
            print(f"\nTotal number of rows in source table {fullyQualifiedTableName} is {rows}\n")
        else:
            print(f"No such table {fullyQualifiedTableName}")
            sys.exit(1)
        
        # create a dataframe from the loaded data
        house_df = self.spark.sql(f"SELECT * FROM {fullyQualifiedTableName}")
        return house_df

    # Write data to Hive table
    def write_data(self, df, tableName) -> None:  
        DSDB = "DS"
        fullyQualifiedTableName = f"{DSDB}.{tableName}"
        try:
            df.write.mode("overwrite").option("schema", self.tableSchema).saveAsTable(fullyQualifiedTableName)
            print(f"Dataframe data written to table: {fullyQualifiedTableName}")
        except Exception as e:
            print(f"Error writing data: {e}")

    # Cleans up loaded data 
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

        """
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
        """

        # Calculate the summary percentage of columns that have no values (all null)
        total_columns = len(null_percentage_dict)
        null_columns_count = sum(1 for percentage in null_percentage_dict.values() if percentage == 100.0)
        null_columns_percentage = (null_columns_count / total_columns) * 100

        # Print summary percentage
        print(f"\nSummary percentage of columns with no values all null: {null_columns_percentage:.1f}%")
        # Exclude columns with 100% null values
        filtered_null_percentage_dict = {col_name: percentage for col_name, percentage in null_percentage_dict.items() if percentage < 100}
        # Generate dynamic schema
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

        def process_data(self, house_df_indexed):
            """
            Process data, including analyzing missing values, applying transformations, and analyzing fraudulent transactions.

            Args:
                house_df_indexed (spark.sql.DataFrame): The indexed DataFrame to process.

            Returns:
                tuple: A tuple containing the processed DataFrame and additional data.
            """   
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
  
    def impute_data_vae(self, house_df):
        """
        Imputes missing values in the DataFrame using a Variational Autoencoder (VAE).

        Args:
            house_df (spark.sql.DataFrame): The DataFrame containing missing values.

        Returns:
            spark.sql.DataFrame: The DataFrame with imputed values for numerical columns.
        """
        try:
            house_df.printSchema()
            # Preprocess data for VAE
            numerical_cols = [col_name for col_name, dtype in house_df.dtypes if dtype in ['int', 'double']]
            scaler = StandardScaler(withMean=True, withStd=True)  # Use withMean and withStd
            print(scaler)
            model = scaler.fit(house_df)
            print(model)
            house_df_normalized = model.transform(house_df)
            print(f"\n\nhouse_df_normalized is {house_df_normalized}")
            print(house_df_normalized.isnull().sum())

            # Convert DataFrame to Pandas for VAE training (can be done in PySpark as well)
            house_df_normalized = house_df_normalized.toPandas()

            # Train VAE model using Pandas
            X_train_numeric = house_df_normalized.values.astype(float)
            input_dim = X_train_numeric.shape[1]
            latent_dim = 2  # Adjust this based on your needs

            vae = VAE(input_dim, latent_dim)
            vae.compile(optimizer='adam', loss=MeanSquaredError())
            vae.fit(X_train_numeric, X_train_numeric, epochs=10, batch_size=32)

            print(f"\nImpute missing values using VAE")
            print(f"\nThis assumes missing values are only in numerical columns")
            imputed_data = vae.predict(X_train_numeric)

            # Convert imputed data back to Spark DataFrame
            imputed_df = self.spark.createDataFrame(imputed_data, schema=house_df.schema)

            # Join imputed data with original DataFrame (excluding numerical columns)
            other_cols = [col for col in house_df.columns if col not in numerical_cols]
            house_df_imputed = imputed_df.join(house_df.select(other_cols), on=other_cols, how='inner')

            return house_df_imputed

        except Exception as e:
            print("Error:", e)
            # Return the original DataFrame in case of error
            return house_df

    def build_vae_model(self):
        # Encoder
        inputs = Input(shape=(self.input_dim,))
        h = Dense(256, activation='relu')(inputs)
        z_mean = Dense(self.latent_dim)(h)
        z_log_var = Dense(self.latent_dim)(h)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        z = Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])

        # Decoder
        decoder_h = Dense(256, activation='relu')
        decoder_out = Dense(self.input_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        outputs = decoder_out(h_decoded)

        # Define models
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        decoder_input = Input(shape=(self.latent_dim,))
        decoder_h_decoded = decoder_h(decoder_input)
        decoder_outputs = decoder_out(decoder_h_decoded)
        decoder = Model(decoder_input, decoder_outputs, name='decoder')
        vae = Model(inputs, decoder(encoder(inputs)[2]), name='vae')

        # VAE loss
        def vae_loss(x, x_decoded_mean, z_mean=z_mean, z_log_var=z_log_var):
            xent_loss = tf.keras.losses.binary_crossentropy(x, x_decoded_mean)
            kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)

        vae.compile(optimizer='adam', loss=vae_loss)
        return vae
     
class VAE(Model):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        inputs = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(256, activation='relu')(inputs)
        z_mean = layers.Dense(self.latent_dim)(x)
        z_log_var = layers.Dense(self.latent_dim)(x)
        return Model(inputs, [z_mean, z_log_var])

    def build_decoder(self):
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(256, activation='relu')(latent_inputs)
        outputs = layers.Dense(self.input_dim, activation='sigmoid')(x)
        return Model(latent_inputs, outputs)

    def sample(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], self.latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sample(z_mean, z_log_var)

        return self.decoder(z)
 
def main():
    appName = "Generative Fraud Detection with VAE"
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName(appName) \
        .enableHiveSupport() \
        .getOrCreate()
    # Set the log level to ERROR to reduce verbosity
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    print(f"\n\n Module name running is {appName}\n\n")

 
    input_dim = 10 # Example: Adjust this based on the number of features in your data
    latent_dim = 2 # Example: Adjust this based on the desired dimensionality of latent space
    df_processor = DataFrameProcessor(spark, input_dim, latent_dim)
    print(f"\n\nLoading data")
    house_df = df_processor.load_data()
    #house_df.printSchema()
    categorical_cols = ["Tenure", "PropertyAddress", "District", "County", "Region", "Postcode", "MultipleAddressIndicator"]  
     #encoder = OneHotEncoder(sparse=False)  # Include the 'sparse' argument
    encoder = OneHotEncoder()  # Remove the 'sparse' argument
    
    encoded_df = encoder.fit_transform(house_df[categorical_cols].toPandas())
    
    for col_name in categorical_cols:
        print(f"\n\nColumn {col_name}: {house_df.select(col(col_name)).distinct().count()}")
    house_df = house_df.select(*categorical_cols, "PricePaid")
    house_df_pandas = house_df.toPandas()

    expected_num_columns = len(categorical_cols) + 1  # Add "PricePaid" column
    print(f"\n\nexpected_num_columns is {expected_num_columns}")

    house_df.printSchema()
    print(house_df_pandas.columns)
    
    # Assert that the number of columns matches expectations
    assert house_df_pandas.shape[1] == expected_num_columns, (
    f"Unexpected number of columns in house_df_pandas ({house_df_pandas.shape[1]}). "
    f"Expected {expected_num_columns} columns (categorical + PricePaid)."
    )
    
    try:
      X_train_numeric = np.concatenate((encoded_df, house_df_pandas["PricePaid"].values.reshape(-1, 1)), axis=1)
    except ValueError as e:
      if "zero-dimensional arrays" in str(e):
        print("encoded_df is zero-dimensional. Consider alternative approaches for handling categorical features with few unique values.")
      else:
        # Handle other potential ValueErrors (optional)
        raise e

    print(house_df_pandas)
    print(f"Shape of house_df_pandas: {house_df_pandas.shape}")

    try:
        if len(encoded_df.shape) > 1:
            X_train_numeric = np.concatenate((encoded_df, house_df_pandas["PricePaid"].values.reshape(-1, 1)), axis=1)
        else:
            print("encoded_df is zero-dimensional. Consider alternative approaches for handling categorical features with few unique values.")
    except ValueError as e:
        if "zero-dimensional arrays" in str(e):
            pass  # We already printed the message in the 'else' block
        else:
            # Handle other potential ValueErrors (optional)
            raise e

    # Print the shape of X_train_numeric for verification
    if 'X_train_numeric' in locals():  # Check if X_train_numeric is defined
      print(f"Shape of X_train_numeric: {X_train_numeric.shape}")


    # Improved code for counting unique categories and calculating total
    unique_category_counts = {col_name: house_df.select(col(col_name)).distinct().count()
    for col_name in categorical_cols}
    total_categories = sum(unique_category_counts.values())
    print(f"\nTotal number of unique categories: {total_categories}\n")

    # Set input_dim to be equal to the total number of unique categories
    input_dim_workedout = total_categories
    input_dim = input_dim_workedout 
 
    print(f"\n\nClean up and impute missing values with vae")
    house_df_cleaned = df_processor.cleanup_data(house_df)
    house_df_imputed = df_processor.impute_data_vae(house_df_cleaned)

    # Convert the DataFrame to a Pandas DataFrame
    house_df_pandas = house_df_imputed.toPandas()

    # Convert numeric columns to a NumPy array of floats
    numeric_columns = house_df_pandas.select_dtypes(include=[np.number]).columns
    print(f"\nnumeric_columns is {numeric_columns}")
    X_train_numeric = house_df_pandas[numeric_columns].values.astype(float)

    # Conditional VAE model training with error handling
    if X_train_numeric is not None:
        try:
            # Define input dimension and latent dimension
            input_dim = total_categories
            latent_dim = 2  # Example: Adjust this based on your needs

            print(f"\nCreate the VAE model, assuming VAE class is defined elsewhere")
            vae = VAE(input_dim, latent_dim)
            print(f"\nCompile the model")
            vae.compile(optimizer='adam', loss=MeanSquaredError())
            print(f"\nTrain the model with error handling")
            vae.fit(X_train_numeric, X_train_numeric, epochs=10, batch_size=32)
            print("VAE model training completed successfully.")
        except Exception as e:
            print(f"An error occurred during VAE model training: {e}")

    else:
        print("X_train_numeric is None, cannot proceed with model training.")
 
 
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


