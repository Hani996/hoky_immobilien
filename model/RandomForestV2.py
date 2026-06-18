import os
import time  # Library for tracking execution time
import numpy as np
import pandas as pd
import joblib  # Library for saving and loading models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("⏳ 1️⃣ Loading and preparing advanced real estate dataset...")

# Automatically detect the current directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go one step back (..) and enter the 'data' folder to read the CSV file safely
data_path = os.path.normpath(os.path.join(current_dir, "../data/price_clean.csv"))

print(f"📂 Reading data from: {data_path}")

# Read the dataset
df = pd.read_csv(data_path)

# Dynamic feature selection: dynamic check if the column exists in the CSV file
all_possible_features = [
    'obj_purchasePrice', 'obj_livingSpace', 'obj_yearConstructed',
    'geo_krs', 'obj_condition', 'obj_regio3', 'obj_noRooms',
    'obj_heatingType', 'obj_firingTypes', 'obj_hasKitchen', 'obj_cellar'
]

# Keep only features that actually exist in your price_clean.csv
features_to_keep = [col for col in all_possible_features if col in df.columns]
df = df[features_to_keep].copy()

print(f"📊 Initial data shape based on available features: {df.shape}")


def remove_outliers_by_quantile(dataframe, lower_q: float = 0.01, upper_q: float = 0.99):
    """
    Function to clean data from outliers based on quantile thresholds
    """
    df_filtered = dataframe.copy()
    
    # Calculate price threshold (obj_purchasePrice)
    price_low = df_filtered['obj_purchasePrice'].quantile(lower_q)
    price_high = df_filtered['obj_purchasePrice'].quantile(upper_q)
    
    # Calculate living space threshold (obj_livingSpace)
    space_low = df_filtered['obj_livingSpace'].quantile(lower_q)
    space_high = df_filtered['obj_livingSpace'].quantile(upper_q)
    
    print(f"📊 Calculated price threshold: from {price_low:,.0f} EUR to {price_high:,.0f} EUR")
    print(f"📐 Calculated space threshold: from {space_low:.1f}m² to {space_high:.1f}m²")
    
    # Apply filtering and exclude values outside boundaries
    df_filtered = df_filtered[
        (df_filtered['obj_purchasePrice'] >= price_low) & 
        (df_filtered['obj_purchasePrice'] <= price_high)
    ]
    
    df_filtered = df_filtered[
        (df_filtered['obj_livingSpace'] >= space_low) & 
        (df_filtered['obj_livingSpace'] <= space_high)
    ]
    
    return df_filtered


# Apply outlier removal function
df = remove_outliers_by_quantile(df, lower_q=0.01, upper_q=0.99)
print(f"🧹 Cleaned data shape after applying quantile thresholds: {df.shape}")

# Fill missing values for numerical features using median
if 'obj_yearConstructed' in df.columns:
    df['obj_yearConstructed'] = df['obj_yearConstructed'].fillna(df['obj_yearConstructed'].median())

if 'obj_noRooms' in df.columns:
    df['obj_noRooms'] = df['obj_noRooms'].fillna(df['obj_noRooms'].median())

# Fill missing values for categorical features using 'unknown' label safely
categorical_cols = ['geo_krs', 'obj_condition', 'obj_regio3', 'obj_heatingType', 'obj_firingTypes']
existing_categorical_cols = [col for col in categorical_cols if col in df.columns]

for col in existing_categorical_cols:
    df[col] = df[col].fillna('unknown')

# 🛠️ Safe Handling for boolean features (handling text like 'n', 'y', 'nein', 'ja')
boolean_cols = ['obj_hasKitchen', 'obj_cellar']
existing_boolean_cols = [col for col in boolean_cols if col in df.columns]

# Mapping dictionary to normalize text/boolean indicators to standard True/False
bool_mapping = {
    'y': True, 'yes': True, 'j': True, 'ja': True, True: True, 1: True, '1': True,
    'n': False, 'no': False, 'nein': False, False: False, 0: False, '0': False
}

for col in existing_boolean_cols:
    # Fill missing with False, map values using the dictionary, and convert safely to integer (0 or 1)
    df[col] = df[col].fillna(False).get(col, df[col]).mapping_values = df[col].map(bool_mapping).fillna(False).astype(int)

print(f"Missing values count in features: {df.isna().sum().sum()}")

# Apply mathematical log transformation to compress the price range
df['obj_purchasePrice_log'] = np.log1p(df['obj_purchasePrice'])

# Separate features (X) from the target logarithmic price variable (y)
X = df.drop(columns=['obj_purchasePrice', 'obj_purchasePrice_log'], errors='ignore')
y_log = df['obj_purchasePrice_log']

# Split data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

print("⏳ 2️⃣ Applying Target Encoding to high-cardinality categorical features...")
if existing_categorical_cols:
    encoder = TargetEncoder(smooth="auto", random_state=42)
    X_train[existing_categorical_cols] = encoder.fit_transform(X_train[existing_categorical_cols], y_train)
    X_test[existing_categorical_cols] = encoder.transform(X_test[existing_categorical_cols])

print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")

print("⏳ 3️⃣ Training the Advanced Random Forest model...")
# Start tracking training execution time
start_time = time.time()

# Instantiate the model with verbose=2 to actively display progress in the terminal
rf_model_v2 = RandomForestRegressor(
    n_estimators=600,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=2  # Displays tree construction logs in real-time
)

# Run model training
rf_model_v2.fit(X_train, y_train)

# End tracking training execution time
end_time = time.time()
duration = end_time - start_time
print(f"🌲 Training finished successfully in {duration:.2f} seconds!")
print("-" * 60)

# Calculate R² Score on log scale
train_r2 = rf_model_v2.score(X_train, y_train)
test_r2 = rf_model_v2.score(X_test, y_test)

print(f"🏋️ Random Forest V2 R² Score on Training data (Log Scale): {train_r2 * 100:.2f}%")
print(f"🧪 Random Forest V2 R² Score on Test data (Log Scale): {test_r2 * 100:.2f}%")
print("-" * 60)

print("⏳ 4️⃣ Calculating advanced mathematical error metrics...")
y_pred_log = rf_model_v2.predict(X_test)

mse_log = mean_squared_error(y_test, y_pred_log)
mae_log = mean_absolute_error(y_test, y_pred_log)

y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred_log)

mse_original = mean_squared_error(y_test_original, y_pred_original)
mae_original = mean_absolute_error(y_test_original, y_pred_original)
rmse_original = np.sqrt(mse_original)

print("🚀 ====== Error Metrics Valuation ======")
print(f"📉 Logarithmic MSE: {mse_log:.4f}")
print(f"📉 Logarithmic MAE: {mae_log:.4f}")
print("--------------------------------------------------")
print(f"Euro MSE: {mse_original:,.2f} EUR²")
print(f"🎯 Euro RMSE (Root Mean Squared Error): ±{rmse_original:,.2f} EUR")
print(f"📊 Euro MAE (Mean Absolute Error): ±{mae_original:,.2f} EUR")
print("-" * 60)

print("⏳ 5️⃣ Creating the .joblib file for Random Forest V2 model...")
model_filename = os.path.join(current_dir, 'rf_v2_estate_model.joblib')
joblib.dump(rf_model_v2, model_filename)

print(f"✅ Random Forest V2 model successfully exported to:\n   {model_filename}")
print("🚀 The advanced model is now frozen and ready for production deployment!")