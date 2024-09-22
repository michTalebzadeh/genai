# Import necessary libraries
from pyspark.sql import SparkSession  # Import SparkSession to manage Spark application
from pyspark.sql.functions import col, lower  # Import functions for DataFrame operations

# Initialize Spark session
spark = SparkSession.builder.appName("NikeTrainersInfo").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
# Create a Spark session named "NikeTrainersInfo" or retrieve an existing one

# Load the dataset of Nike products from CSV file
nike_data = spark.read.csv("hdfs://rhes75:9000/ds/nike_trainers.csv", header=True, inferSchema=True)
# Read the CSV file into a Spark DataFrame named `nike_data`
# - `header=True` indicates the first row contains column headers
# - `inferSchema=True` infers the schema (data types) of each column automatically

# Define a function to search for specific trainers based on a query
def search_nike_trainers(query):
    # Convert the query to lowercase for case-insensitive search
    query_lower = query.lower()
    
    # Filter the Nike dataset for descriptions containing the query string
    results = nike_data.filter(lower(col("description")).contains(query_lower))
    
    return results

# Example search query
query = "Air Max"
# Set an example query to search for Nike trainers containing "Air Max" in their description

results = search_nike_trainers(query)
# Call the `search_nike_trainers` function with the query and store the results

# Show the results
results.show(truncate=False)
# Display the filtered results matching the query in the console or notebook output
