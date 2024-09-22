# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import when

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Fraud Detection") \
    .getOrCreate()

# Set the log level to ERROR to reduce verbosity
sc = spark.sparkContext
sc.setLogLevel("ERROR")

# Read text data into a Spark DataFrame
DIRECTORY="/d4T/hduser/genai"
transaction_file=f"file://{DIRECTORY}/OCOD_FULL_2024_03.csv"
# Define the schema of the DataFrame
schema = "transaction_id INT, amount DOUBLE, merchant STRING, category STRING, is_fraud BOOLEAN"

# Read the text file using the defined schema and tab as the delimiter
data = spark.read.csv(transaction_file, schema=schema, sep="\t", header=True)
data.printSchema()
print(f"{data.count()} rows read in from {transaction_file}")
# Show the DataFrame
#data.show(truncate=False)
# Preprocess data
# Convert is_fraud column from boolean to numeric
data = data.withColumn("is_fraud_numeric", when(data["is_fraud"] == True, 1).otherwise(0))

# String Indexing for categorical columns
merchant_indexer = StringIndexer(inputCol="merchant", outputCol="merchant_index")
category_indexer = StringIndexer(inputCol="category", outputCol="category_index")

# One-Hot Encoding for indexed categorical columns
merchant_encoder = OneHotEncoder(inputCol="merchant_index", outputCol="merchant_encoded")
category_encoder = OneHotEncoder(inputCol="category_index", outputCol="category_encoded")

# Assemble feature vector
assembler = VectorAssembler(inputCols=['transaction_id', 'amount', 'merchant_encoded', 'category_encoded'], outputCol='features')

# Pipeline for preprocessing steps
pipeline = Pipeline(stages=[merchant_indexer, category_indexer, merchant_encoder, category_encoder, assembler])

# Fit pipeline on data
pipeline_model = pipeline.fit(data)

# Transform data
data = pipeline_model.transform(data)

# Split data into training and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Train a Random Forest classifier
rf = RandomForestClassifier(labelCol='is_fraud_numeric', featuresCol='features', numTrees=100)
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
evaluator = BinaryClassificationEvaluator(labelCol='is_fraud_numeric', metricName='areaUnderROC')
auc_train = evaluator.evaluate(predictions)
print("Area Under ROC (Training):", auc_train)
# Evaluate model performance on the test data
auc_test = evaluator.evaluate(predictions_test)
print("Area Under ROC (Test):", auc_test)
auc = evaluator.evaluate(predictions)
print("Area Under ROC:", auc)

# Stop SparkSession
spark.stop()
