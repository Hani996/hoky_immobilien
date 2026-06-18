import os
import pandas as pd
import numpy as np
import joblib  # Library for saving and loading models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

print("⏳ 1️⃣ Loading and preparing real estate dataset...")

# Automatically detect the current directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go one step back (..) and enter the 'data' folder to read the CSV file safely
data_path = os.path.normpath(os.path.join(current_dir, "../data/price_clean.csv"))

print(f"📂 Reading data from: {data_path}")

features_to_keep = [
    'obj_purchasePrice',
    'obj_livingSpace',
    'obj_yearConstructed',
    'geo_krs',
    'geo_plz',
    'obj_condition',
    'obj_regio3',
    'obj_noRooms'
]

# Read file and filter columns
df = pd.read_csv(data_path)
df = df[df.columns.intersection(features_to_keep)].copy()

print(f"📊 Initial data shape: {df.shape}")


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

# Convert categorical features into dummy variables (0 and 1) and avoid multicollinearity
categorical_cols = ['geo_krs', 'obj_condition', 'obj_regio3']
dummies = pd.get_dummies(df[categorical_cols], drop_first=True, dtype=int)

# Concatenate dummy columns with the original dataframe
df = pd.concat([df, dummies], axis=1)

# Drop original categorical text columns and zip code column
df = df.drop(columns=categorical_cols + ['geo_plz'], errors='ignore')
print(f"📊 Data shape after merging dummies and dropping original text columns: {df.shape}")

# Fill missing values in construction year with median to stabilize calculation
median_year = df['obj_yearConstructed'].median()
df['obj_yearConstructed'] = df['obj_yearConstructed'].fillna(median_year)

# Verify there are no more missing (NaN) values in the features
print(f"Missing values count in X features: {df.isna().sum().sum()}")

# Separate features (X) from the target price variable (y)
X = df.drop(columns=['obj_purchasePrice'])
y = df['obj_purchasePrice']

# Split data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")

print("⏳ 2️⃣ Training the Random Forest model (This might take a few seconds)...")
# Instantiate and train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Calculate accuracy metrics
train_accuracy = rf_model.score(X_train, y_train)
test_accuracy = rf_model.score(X_test, y_test)

print(f"🏋️ Random Forest model accuracy on Training data: {train_accuracy * 100:.2f}%")
print(f"🧪 Random Forest model accuracy on Test data: {test_accuracy * 100:.2f}%")
print("-" * 60)

print("⏳ 3️⃣ Creating the .joblib file for Random Forest model...")
# Save and serialize the trained Random Forest model inside the current 'model' directory
model_filename = os.path.join(current_dir, 'rf_estate_model.joblib')
joblib.dump(rf_model, model_filename)

print(f"✅ Random Forest model successfully exported to:\n   {model_filename}")
print("🚀 The model is now frozen and ready for instant deployment!")