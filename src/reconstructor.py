from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Initialize Spark session
spark = SparkSession.builder.appName("ReconstructedDF").getOrCreate()

# Define the schema
schema = StructType([
    StructField("TitleNumber", StringType(), True),
    StructField("PropertyAddress", StringType(), True),
    StructField("District", StringType(), True),
    StructField("County", StringType(), True),
    StructField("Postcode", StringType(), True),
    StructField("PricePaid", IntegerType(), True),
    StructField("ProprietorName1", StringType(), True),
    StructField("CompanyRegistrationNo1", StringType(), True),
    StructField("CountryIncorporated1", StringType(), True),
    StructField("Proprietor1Address1", StringType(), True),
    StructField("Proprietor1Address2", StringType(), True),
    StructField("Proprietor1Address3", StringType(), True),
    StructField("ProprietorName2", StringType(), True),
    StructField("Proprietor2Address1", StringType(), True),
    StructField("Proprietor2Address2", StringType(), True),
    StructField("Tenure_Freehold", IntegerType(), True),
    StructField("Tenure_Leasehold", IntegerType(), True),
    StructField("Region_EAST_ANGLIA", IntegerType(), True),
    StructField("Region_EAST_MIDLANDS", IntegerType(), True),
    StructField("Region_GREATER_LONDON", IntegerType(), True),
    StructField("Region_NORTH", IntegerType(), True),
    StructField("Region_NORTH_WEST", IntegerType(), True),
    StructField("Region_SOUTH_EAST", IntegerType(), True),
    StructField("Region_SOUTH_WEST", IntegerType(), True),
    StructField("Region_WALES", IntegerType(), True),
    StructField("Region_WEST_MIDLANDS", IntegerType(), True),
    StructField("Region_YORKS_AND_HUMBER", IntegerType(), True),
    StructField("MultipleAddressIndicator_N", IntegerType(), True),
    StructField("MultipleAddressIndicator_Y", IntegerType(), True),
    StructField("ProprietorshipCategory1_Corporate_Body", IntegerType(), True),
    StructField("ProprietorshipCategory1_Limited_Company_or_Public_Limited_Company", IntegerType(), True),
    StructField("ProprietorshipCategory1_Limited_Liability_Partnership", IntegerType(), True),
    StructField("ProprietorshipCategory1_Unlimited_Company", IntegerType(), True),
    StructField("CompanyRegistrationNo2", StringType(), True),
    StructField("ProprietorshipCategory2_Corporate_Body", IntegerType(), True),
    StructField("ProprietorshipCategory2_Limited_Company_or_Public_Limited_Company", IntegerType(), True),
    StructField("ProprietorshipCategory2_Limited_Liability_Partnership", IntegerType(), True),
    StructField("ProprietorshipCategory2_Unlimited_Company", IntegerType(), True),
    StructField("CountryIncorporated2", StringType(), True),
    StructField("Proprietor2Address3", StringType(), True)
])

# Create an empty DataFrame with the schema
df = spark.createDataFrame([], schema)

# Show the DataFrame schema
df.printSchema()


