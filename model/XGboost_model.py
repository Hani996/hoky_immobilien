import os
import time  # Library for tracking execution time
import numpy as np
import pandas as pd
import joblib  # Library for saving and loading models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

print("⏳ 1️⃣ Loading and preparing real estate dataset for Regression...")

# Automatically detect the current directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go one step back (..) and enter the 'data' folder to read the CSV file safely
data_path = os.path.normpath(os.path.join(current_dir, "../data/price_clean.csv"))

print(f"📂 Reading data from: {data_path}")

# Read the dataset
df_raw = pd.read_csv(data_path)

# Define extended features
features_to_keep = [
    'obj_purchasePrice', 'obj_livingSpace', 'obj_yearConstructed',
    'geo_krs', 'geo_plz', 'obj_condition', 'obj_regio3', 'obj_noRooms',
    'obj_cellar', 'obj_firingTypes'
]

df_raw = df_raw[df_raw.columns.intersection(features_to_keep)].copy()


def remove_outliers_by_quantile(dataframe, lower_q: float = 0.01, upper_q: float = 0.99):
    """
    Function to clean data from outliers based on quantile thresholds
    """
    df_filtered = dataframe.copy()
    
    # Calculate thresholds safely
    price_low = df_filtered['obj_purchasePrice'].quantile(lower_q)
    price_high = df_filtered['obj_purchasePrice'].quantile(upper_q)
    space_low = df_filtered['obj_livingSpace'].quantile(lower_q)
    space_high = df_filtered['obj_livingSpace'].quantile(upper_q)
    
    print(f"📊 Calculated price threshold: from {price_low:,.0f} EUR to {price_high:,.0f} EUR")
    print(f"📐 Calculated space threshold: from {space_low:.1f}m² to {space_high:.1f}m²")
    
    return df_filtered[
        (df_filtered['obj_purchasePrice'] >= price_low) & 
        (df_filtered['obj_purchasePrice'] <= price_high) & 
        (df_filtered['obj_livingSpace'] >= space_low) & 
        (df_filtered['obj_livingSpace'] <= space_high)
    ]


# Apply outlier removal
df_clean = remove_outliers_by_quantile(df_raw, lower_q=0.01, upper_q=0.99)
print(f"🧹 Cleaned data shape after applying quantile thresholds: {df_clean.shape}")

print("⏳ 2️⃣ Engineering advanced features for the regression model...")
# Feature Engineering Phase
df_clean['obj_age'] = 2026 - df_clean['obj_yearConstructed']
df_clean['space_per_room'] = df_clean['obj_livingSpace'] / (df_clean['obj_noRooms'] + 0.1)

# Geographical feature 1: Region popularity based on volume of occurrences
df_clean['location_popularity'] = df_clean['obj_regio3'].map(df_clean['obj_regio3'].value_counts())

# Geographical feature 2: Living space compared to its county average
county_avg = df_clean.groupby('geo_krs')['obj_livingSpace'].transform('mean')
df_clean['space_to_county_avg'] = df_clean['obj_livingSpace'] / (county_avg + 0.1)

# Apply mathematical log transformation to compress the price range and isolate target
y_log = np.log1p(df_clean['obj_purchasePrice'])
X_raw = df_clean.drop(columns=['obj_purchasePrice'], errors='ignore')

# Impute remaining missing values
X_raw = X_raw.fillna(X_raw.median(numeric_only=True))
X_raw = X_raw.fillna('missing')

# Split data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_log, test_size=0.2, random_state=42)

print("⏳ 3️⃣ Applying Target Encoding to high-cardinality categorical features...")
categorical_cols = ['geo_krs', 'geo_plz', 'obj_condition', 'obj_regio3', 'obj_cellar', 'obj_firingTypes']
existing_categorical_cols = [col for col in categorical_cols if col in X_train.columns]

# Initialize and train Target Encoder
encoder = TargetEncoder(smooth="auto", random_state=42)
X_train[existing_categorical_cols] = encoder.fit_transform(X_train[existing_categorical_cols], y_train)
X_test[existing_categorical_cols] = encoder.transform(X_test[existing_categorical_cols])

print("🔥 Advanced geographic and economic features prepared successfully!")
print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")

print("⏳ 4️⃣ Training the Advanced XGBoost Regressor model...")
# Start tracking training execution time
start_time = time.time()

# Build the constrained XGBoost Regressor structure focused on high generalization
xgb_model = xgb.XGBRegressor(
    n_estimators=600,
    max_depth=4,              # Lower depth to enforce generalization and lift test score
    learning_rate=0.02,       # Slower learning rate to capture complex patterns smoothly
    subsample=0.7,            # Introduce random rows selection to boost robustness
    colsample_bytree=0.7,     # Introduce random feature subsets selection
    reg_alpha=0.5,            # L1 regularization to damp weak features impact
    reg_lambda=1.5,           # L2 regularization to avoid weights inflation
    random_state=42,
    n_jobs=-1
)

# Run model training
xgb_model.fit(X_train, y_train)

# End tracking training execution time
end_time = time.time()
duration = end_time - start_time
print(f"⚡ XGBoost Regression Model Training finished successfully in {duration:.2f} seconds!")
print("-" * 60)

print("⏳ 5️⃣ Evaluating model performance metrics (Regression Focus)...")
# Predict log scale prices
y_pred_log = xgb_model.predict(X_test)

# Transform prices back to original natural Euro currency scale
y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred_log)

# 1. Calculate final R² Score on the actual real-world Euro scale
r2_original = r2_score(y_test_original, y_pred_original)

# 2. Calculate real-world error metrics on original natural scale (Euro)
mse_original = mean_squared_error(y_test_original, y_pred_original)
mae_original = mean_absolute_error(y_test_original, y_pred_original)

print("\n🚀 ====== FINAL REGRESSION MODEL PERFORMANCE METRICS ======")
print(f"📊 R² Score (Coefficient of Determination): {r2_original * 100:.2f}%")
print(f"📉 MSE (Mean Squared Error): {mse_original:,.2f} EUR²")
print(f"📊 MAE (Mean Absolute Error): ±{mae_original:,.2f} EUR")
print("===========================================================\n")

print("⏳ 6️⃣ Freezing and creating the .joblib file for deployment...")
# Export and serialize the trained model inside the model directory
model_filename = os.path.join(current_dir, 'xgb_estate_model.joblib')
joblib.dump(xgb_model, model_filename)

print(f"✅ Excellent! XGBoost regression model successfully exported to:\n   {model_filename}")
print("🚀 The final model is fully frozen and deployed for live property predictions!")