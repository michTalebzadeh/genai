# Ensure OpenAI library is installed
try:
    import openai
except ImportError:
    import os
    os.system('pip install openai')
    import openai

# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Initialize OpenAI API (make sure to set your API key)
openai.api_key = 'your-openai-api-key'

# Create a Spark session
spark = SparkSession.builder \
    .appName("Spark Structured Streaming with Rate Source and Generative AI") \
    .getOrCreate()

# Set the log level to reduce verbosity
spark.sparkContext.setLogLevel("WARN")

# Read data from the rate source
rate_df = spark.readStream \
    .format("rate") \
    .option("rowsPerSecond", 1) \
    .load()

# Process the data (e.g., add a new column with a transformed value)
processed_df = rate_df.withColumn("value_squared", col("value") * col("value"))

# Function to send data to OpenAI GPT-4 for analysis
def analyze_with_gpt4(batch_df, batch_id):
    # Collect the data from the batch DataFrame
    data = batch_df.collect()
    for row in data:
        value = row['value']
        value_squared = row['value_squared']
        
        # Create a prompt for GPT-4
        prompt = f"The value is {value} and its square is {value_squared}. Provide some interesting insights or context about these numbers."
        
        try:
            # Call the GPT-4 API
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract and print the response
            gpt4_response = response['choices'][0]['message']['content'].strip()
            print(f"GPT-4 Response for value {value}: {gpt4_response}")
        except Exception as e:
            print(f"Error calling GPT-4 API: {e}")

# Write the processed data to the console and analyze with GPT-4
query = processed_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .foreachBatch(analyze_with_gpt4) \
    .start()

# Await termination
query.awaitTermination()
