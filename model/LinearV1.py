import os
import time  # Library for tracking execution time
import numpy as np
import pandas as pd
import joblib  # Library for saving and loading models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sqlite3

# Define the database and table names
table_name = 'house_prices'

# 1. Automatically detect the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go one step back (..) to the root, then enter data/db/prices.db
db_filename = os.path.normpath(os.path.join(current_dir, "../data/db/prices.db"))

print(f"🗄️ Connecting to database at: {db_filename}")

# 3. Connect to the existing SQLite database using the absolute path
conn = sqlite3.connect(db_filename)

df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

# Ensure that the 'obj_telekomInternetProductAvailable' column is treated as an object (string) type for safe processing
df['obj_telekomInternetProductAvailable'] = df['obj_telekomInternetProductAvailable'].astype(object)

# 3. Always close the database connection when done
conn.close()

""" print("⏳ 1️⃣ Loading and preparing real estate dataset for Regression...")

# Automatically detect the current directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go one step back (..) and enter the 'data' folder to read the CSV file safely
data_path = os.path.normpath(os.path.join(current_dir, "../data/price_clean.csv"))

print(f"📂 Reading data from: {data_path}")

# Read the dataset from scratch
df = pd.read_csv(data_path)
 """
# Create dummies for the state column 'obj_regio1'
dummies = pd.get_dummies(df.obj_regio1, dtype=int)

# Concatenate dummies and drop 'Niedersachsen' to avoid multi-collinearity
if 'Niedersachsen' in dummies.columns:
    df_with_dummies = pd.concat([df, dummies.drop('Niedersachsen', axis=1)], axis=1)
else:
    df_with_dummies = pd.concat([df, dummies], axis=1)

# Drop explicit string and postal columns that disrupt the mathematical model
cols_to_drop = [
    'obj_houseNumber', 'obj_street', 'obj_regio1', 
    'obj_purchasePrice_per_qm', 'geo_krs', 'geo_plz'
]
df_filtered = df_with_dummies.drop(columns=[col for col in cols_to_drop if col in df_with_dummies.columns])

# Secure dataframe: keep only numeric or pure boolean types (int, float, bool)
X = df_filtered.select_dtypes(include=[np.number, bool]).copy()

# Fill remaining missing values in internet speeds with zeros for calculation stability
if 'obj_telekomUploadSpeed' in X.columns:
    X['obj_telekomUploadSpeed'] = X['obj_telekomUploadSpeed'].fillna(0)
if 'obj_telekomDownloadSpeed' in X.columns:
    X['obj_telekomDownloadSpeed'] = X['obj_telekomDownloadSpeed'].fillna(0)

# Separate features (X) from the target price variable (y)
y = X['obj_purchasePrice']
X = X.drop(columns=['obj_purchasePrice'])

# Split data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Cleaned feature matrix shape (X_train): {X_train.shape}")
print("✨ Split variables defined successfully in memory!")
print("-" * 60)

print("⏳ 2️⃣ Training the Linear Regression model...")
# Start tracking training execution time
start_time = time.time()

# Instantiate and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# End tracking training execution time
end_time = time.time()
duration = end_time - start_time
print(f"📈 Linear Regression Model Training finished successfully in {duration:.2f} seconds!")
print("-" * 60)

print("⏳ 3️⃣ Evaluating model performance metrics (Regression Focus)...")
# Predict actual prices on test data
y_pred = model.predict(X_test)

# 1. Calculate final R² Score on the actual Euro scale (Decimal scale between 0 and 1)
r2_result = r2_score(y_test, y_pred)

# 2. Calculate real-world error metrics (Euro Scale)
mse_result = mean_squared_error(y_test, y_pred)
mae_result = mean_absolute_error(y_test, y_pred)

print("\n🚀 ====== FINAL REGRESSION MODEL PERFORMANCE METRICS ======")
print(f"📊 R² Score (Coefficient of Determination): {r2_result:.4f}")  # Formatted as a decimal between 0 and 1
print(f"📉 MSE (Mean Squared Error): {mse_result:,.2f} EUR²")
print(f"📊 MAE (Mean Absolute Error): ±{mae_result:,.2f} EUR")
print("===========================================================\n")

print("⏳ 4️⃣ Creating the .joblib file for Linear Regression model...")
# Save and serialize the trained Linear Regression model with the requested name 'LinearV1.joblib'
model_filename = os.path.join(current_dir, 'LinearV1.joblib')
joblib.dump(model, model_filename)

print(f"✅ Linear Regression model successfully exported to:\n   {model_filename}")
print("🚀 The model is now frozen and ready for production predictions!")