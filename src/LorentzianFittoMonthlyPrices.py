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
 

from matplotlib import pyplot as plt
from lmfit.models import LinearModel, LorentzianModel, VoigtModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.stat import ChiSquareTest
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pandas.plotting import scatter_matrix
from pyspark.sql.functions import array, col
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.dates as mdates
import matplotlib.ticker as tkr
from pyspark.sql import functions as F
from pyspark.sql.functions import col, round
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

def functionLorentzian(x, amp1, cen1, wid1, amp2,cen2,wid2, amp3,cen3,wid3):
    return (amp1*wid1**2/((x-cen1)**2+wid1**2)) +\
            (amp2*wid2**2/((x-cen2)**2+wid2**2)) +\
                (amp3*wid3**2/((x-cen3)**2+wid3**2))

def main():
    regionname = "KENSINGTON AND CHELSEA"
    short = regionname.replace(" ", "").lower()

    appName = "lorenzian fit"
    spark = SparkSession.builder \
        .appName(appName) \
        .enableHiveSupport() \
        .getOrCreate()
        # Set the configuration
    
     # Set the log level to ERROR to reduce verbosity
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    #
    # Get data from Hive table
    DSDB = "DS"
    tableName = f"""percentmonthlyhousepricechange_{short}"""
    fullyQualifiedTableName = f"{DSDB}.{tableName}"
    print(f"\ntablename is {fullyQualifiedTableName}")

    start_date = "201001"
    end_date = "202001"
    start_time = datetime.datetime.now()
    print("PySpark code started at:", start_time)
    if spark.sql(f"SHOW TABLES IN {DSDB} LIKE '{tableName}'").count() == 1:
        spark.sql(f"ANALYZE TABLE {fullyQualifiedTableName} COMPUTE STATISTICS")
        rows = spark.sql(f"SELECT COUNT(1) FROM {fullyQualifiedTableName}").collect()[0][0]
        print(f"\nTotal number of rows in table {fullyQualifiedTableName} is {rows}\n")
    else:
        print(f"No such table {fullyQualifiedTableName}")
        sys.exit(1)
    # Model predictions
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    summary_df = spark.sql(f"""SELECT cast(year_month as int) as year_month, percent_change FROM {fullyQualifiedTableName}""")
    df_10 = summary_df.filter(col("year_month").between(f'{start_date}', f'{end_date}'))
    print(df_10.toPandas().columns.tolist())
    p_dfm = df_10.toPandas()  # converting spark DF to Pandas DF
    # Non-Linear Least-Squares Minimization and Curve Fitting
    # Define model to be Lorentzian and deploy it
    model = LorentzianModel()
    n = len(p_dfm.columns)
    for i in range(n):
      if (p_dfm.columns[i] != 'year_month'):   # yyyyMM is x axis in integer
         # it goes through the loop and plots individual average curves one by one and then prints a report for each y value
         vcolumn = p_dfm.columns[i]
         print(vcolumn)
         params = model.guess(p_dfm[vcolumn], x = p_dfm['year_month'])
         result = model.fit(p_dfm[vcolumn], params, x = p_dfm['year_month'])
         # plot the data points, initial fit and the best fit
         plt.plot(p_dfm['year_month'], p_dfm[vcolumn], 'bo', label = 'data')
         #plt.plot(p_dfm['year_month'], result.init_fit, 'k--', label='initial fit')
         plt.plot(p_dfm['year_month'], result.best_fit, 'r-', label='best fit')
         plt.legend(loc='upper left')
         plt.xlabel("Year/Month", fontsize=14)
         plt.text(0.35,
                  0.55,
                  "Fit Based on Non-Linear Lorentzian Model",
                  transform=plt.gca().transAxes,
                  color="grey",
                  fontsize=9
                  )
         property = "Average price percent Change"
         if vcolumn == "flatprice": property = "Flat"
         if vcolumn == "terracedprice": property = "Terraced"
         if vcolumn == "semidetachedprice": property = "semi-detached"
         if vcolumn == "detachedprice": property = "detached"
         plt.ylabel(f"""{property} """, fontsize=14)
         plt.title(f"""Monthly {property} in {regionname}""", fontsize=14)
         plt.xlim(200901, 202101)
         print(result.fit_report())
         plt.savefig(f'{fullyQualifiedTableName}.png')
         # Show the plot
         #plt.show()      
         plt.close()
    
  
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    print("PySpark code finished at:", end_time)
    print("Execution time:", execution_time)

if __name__ == "__main__":
  print("\nworking on this code")
  main()