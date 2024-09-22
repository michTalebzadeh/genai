import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns

# Global variable for PNG directory
PNG_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')

def preprocess_data(df):
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with NaN values if necessary
    df.dropna(inplace=True)

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

np.random.seed(42)

data = {
    'age': np.random.randint(18, 65, size=1000),
    'price': np.random.randint(100000, 500000, size=1000),
    'gender': np.random.choice(['M', 'F'], size=1000),
    'district': np.random.choice(['A', 'B', 'C', 'D'], size=1000)
}

df = pd.DataFrame(data)

# Introduce missing values directly
df['age'][np.random.randint(0, 1000, 100)] = np.nan

# Fill missing values with mean for numerical data
df['age'] = df['age'].fillna(df['age'].mean())
df['price'] = df['price'].fillna(df['price'].mean())

# Encode categorical features
encoder_gender = OneHotEncoder(sparse_output=False)
encoded_gender = encoder_gender.fit_transform(df[['gender']])
encoder_district = OneHotEncoder(sparse_output=False)
encoded_district = encoder_district.fit_transform(df[['district']])

# Combine all features
numerical_features = df[['age', 'price']].values
all_features = np.hstack([numerical_features, encoded_gender, encoded_district])

# Standardize numerical features
scaler = StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)

# VAE model
original_dim = all_features_scaled.shape[1]
intermediate_dim = 64
latent_dim = 2

# Encoder
inputs = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Decoder
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim)
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# VAE model
vae = Model(inputs, x_decoded_mean)

# VAE loss
def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * tf.keras.losses.mean_squared_error(x, x_decoded_mean)
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss(inputs, x_decoded_mean))
vae.compile(optimizer='rmsprop')
vae.fit(all_features_scaled, all_features_scaled, epochs=50, batch_size=32, validation_split=0.2)

# Generate synthetic data
n_samples = 1000
z_sample = np.random.normal(size=(n_samples, latent_dim))
x_decoded = decoder_mean(decoder_h(z_sample))

# Combine back to DataFrame for inspection
encoded_gender_columns = encoder_gender.get_feature_names_out(['gender'])
encoded_district_columns = encoder_district.get_feature_names_out(['district'])
all_columns = ['age', 'price'] + encoded_gender_columns.tolist() + encoded_district_columns.tolist()

synthetic_features_original = scaler.inverse_transform(x_decoded.numpy())
synthetic_df = pd.DataFrame(synthetic_features_original, columns=all_columns)

# Process data
preprocess_data(df)
preprocess_data(synthetic_df)

print("Final DataFrame after Feature Engineering:")
print(df.head())
print("Synthetic DataFrame Head:")
print(synthetic_df.head())

# Visualize the distribution of each feature
for feature in df.columns:
    if feature not in ['gender', 'district']:
        visualize_distribution(df, synthetic_df, feature)


