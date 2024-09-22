from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType
from textblob import TextBlob
from pyspark.sql import SparkSession

appName = "sentiment"
# Create a Spark session

# Set up Spark session and context
spark_session = SparkSession.builder.appName(appName).getOrCreate()
spark_context = spark_session.sparkContext
# Set the log level to ERROR to reduce verbosity
spark_context.setLogLevel("ERROR")

# Sample product data
product_data = [("1", "Great product, durable and reliable."),
                ("2", "Average performance, needs improvement."),
                ("3", "Amazing features and sleek design.")]

# Define a schema for the product data
schema = ["product_id", "description"]

# Create a DataFrame from the sample data
df = spark_session.createDataFrame(product_data, schema=schema)

# Assuming your DataFrame is named df
df = df.withColumn("description", col("description").cast("string"))

# Define the sentiment analysis function
def analyze_sentiment(text):
    if text is not None:
        analysis = TextBlob(text)
        sentiment = analysis.sentiment.polarity
        # Add your logic to determine sentiment category based on polarity
        return sentiment
    else:
        # Handle the case when text is None
        return None

# Register the UDF
analyze_sentiment_udf = udf(analyze_sentiment, FloatType())
df = df.withColumn("calculated_sentiment", analyze_sentiment_udf("description"))

# Show the resulting DataFrame
df.printSchema()
df.show(truncate=False)
