import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample Data below
"""
Dictionary Structure:

The dictionary data contains multiple key-value pairs.
Each key represents a type of data, such as DateTaken, Average Price, Detached Price, etc.
Each value is a list of values corresponding to the key.
Keys:

DateTaken: A list of dates when the data was recorded.
Average Price: A list of average property prices for each date in DateTaken.
Detached Price: A list of prices for detached properties for each date in DateTaken.
Semi Detached Price: A list of prices for semi-detached properties for each date in DateTaken.
Terraced Price: A list of prices for terraced properties for each date in DateTaken.
Flat Price: A list of prices for flats for each date in DateTaken.
Values:

Each key has an associated list of values.
The lists are aligned by index, meaning that the value at index i in each list corresponds to the same date in DateTaken.
DateTaken: List of dates from February 2021 to January 2024.
Average Price: List of average property prices for each date.
Detached Price: List of detached property prices for each date.
Semi Detached Price: List of semi-detached property prices for each date.
Terraced Price: List of terraced property prices for each date.
Flat Price: List of flat property prices for each date.
"""
data = {
    'DateTaken': [
        '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01',
        '2021-07-01', '2021-08-01', '2021-09-01', '2021-10-01', '2021-11-01',
        '2021-12-01', '2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01',
        '2022-05-01', '2022-06-01', '2022-07-01', '2022-08-01', '2022-09-01',
        '2022-10-01', '2022-11-01', '2022-12-01', '2023-01-01', '2023-02-01',
        '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01', '2023-07-01',
        '2023-08-01', '2023-09-01', '2023-10-01', '2023-11-01', '2023-12-01', '2024-01-01'
    ],
    'Average Price': [
        1265469.0, 1224768.0, 1198159.0, 1223380.0, 1196736.0, 
        1321844.0, 1445633.0, 1515740.0, 1527601.0, 1447296.0, 
        1410458.0, 1371047.0, 1374360.0, 1441513.0, 1513572.0, 
        1518352.0, 1520758.0, 1503688.0, 1517723.0, 1527039.0, 
        1410576.0, 1376685.0, 1304428.0, 1364872.0, 1350849.0, 
        1384497.0, 1334814.0, 1363774.0, 1400045.0, 1376043.0, 
        1366026.0, 1315895.0, 1375884.0, 1268656.0, 1199227.0, 1197249.0
    ],
    'Detached Price': [
        3339643.0, 3210883.0, 3180273.0, 3261376.0, 3209506.0,
        3609718.0, 4017462.0, 4257942.0, 4319779.0, 4105235.0,
        3991142.0, 3860414.0, 3850122.0, 4126667.0, 4324868.0,
        4397984.0, 4346722.0, 4320823.0, 4369358.0, 4494141.0,
        4125761.0, 4024638.0, 3769144.0, 4021058.0, 3956980.0,
        4086739.0, 3917525.0, 3954089.0, 4041221.0, 3980686.0,
        4016724.0, 3854223.0, 3982877.0, 3628941.0, 3301387.0, 3257710.0
    ],
    'Semi Detached Price': [
        3457594.0, 3330070.0, 3269217.0, 3341125.0, 3271765.0,
        3667087.0, 4100628.0, 4342912.0, 4403758.0, 4171473.0,
        4059804.0, 3944470.0, 3920283.0, 4178471.0, 4393418.0,
        4455597.0, 4451900.0, 4415196.0, 4458491.0, 4513078.0,
        4158258.0, 4058165.0, 3821416.0, 4035780.0, 3984685.0,
        4111878.0, 3953592.0, 4021585.0, 4125853.0, 4067773.0,
        4082121.0, 3923799.0, 4069382.0, 3719860.0, 3420758.0, 3402971.0
    ],
    'Terraced Price': [
        2347194.0, 2260217.0, 2218038.0, 2265759.0, 2227464.0,
        2475419.0, 2741704.0, 2887687.0, 2919685.0, 2765163.0,
        2687816.0, 2616501.0, 2605946.0, 2756785.0, 2909453.0,
        2940172.0, 2953079.0, 2912599.0, 2941838.0, 2960397.0,
        2741789.0, 2682415.0, 2540540.0, 2661903.0, 2620349.0,
        2680795.0, 2572512.0, 2632418.0, 2716269.0, 2671069.0,
        2665975.0, 2561674.0, 2662450.0, 2436004.0, 2259569.0, 2258623.0
    ],
    'Flat Price': [
        1099165.0, 1065405.0, 1041050.0, 1062748.0, 1037927.0,
        1143756.0, 1245289.0, 1303428.0, 1312071.0, 1243202.0,
        1212598.0, 1178268.0, 1183809.0, 1237634.0, 1297378.0,
        1297847.0, 1299009.0, 1285290.0, 1296957.0, 1304349.0,
        1204022.0, 1174193.0, 1113022.0, 1163575.0, 1153696.0,
        1182815.0, 1142170.0, 1166603.0, 1195780.0, 1174990.0,
        1164021.0, 1122277.0, 1176026.0, 1087318.0, 1034924.0, 1032989.0
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert DateTaken to datetime
df['DateTaken'] = pd.to_datetime(df['DateTaken'])

# Convert DateTaken to a numerical format for model training
df['year_month'] = df['DateTaken'].dt.year * 100 + df['DateTaken'].dt.month

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
# Function to train the model and predict future prices
def train_and_predict(column_name):
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
    future_year_month = scaler.transform(future_year_month.values.reshape(-1, 1))

    # Predict future prices
    predictions = model.predict(future_year_month)

    # Print predictions
    for date, pred in zip(future_dates, predictions):
        print(f"Predicted {column_name} for {date.strftime('%Y-%m')}: {pred:.2f}")

# List of property types to predict
property_types = ['Average Price', 'Detached Price', 'Semi Detached Price', 'Terraced Price', 'Flat Price']

# Train and predict for each property type
for property_type in property_types:
    train_and_predict(property_type)

