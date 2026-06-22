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
""" 
print("⏳ 1️⃣ Loading and preparing real estate dataset for Linear Regression...")

# Automatically detect the current directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go one step back (..) and enter the 'data' folder to read the CSV file safely
data_path = os.path.normpath(os.path.join(current_dir, "../data/price_clean.csv"))

print(f"📂 Reading data from: {data_path}")

# Read the dataset from scratch
df = pd.read_csv(data_path)
 """
# Create dummy variables for the state column 'obj_regio1'
dummies = pd.get_dummies(df.obj_regio1, dtype=int)

# Concatenate dummies horizontally and drop 'Niedersachsen' to avoid the dummy variable trap
df = pd.concat([df, dummies.drop('Niedersachsen', axis=1, errors='ignore')], axis=1)

# Isolate the target variable (y)
y = df['obj_purchasePrice']

# Isolate only numeric and boolean columns (including the newly created state dummies)
X_numeric = df.select_dtypes(include=[np.number, bool]).copy()

# Drop large text columns, price per square meter, and the target variable itself
cols_to_drop = ['obj_purchasePrice', 'obj_purchasePrice_per_qm', 'obj_regio1']
X = X_numeric.drop(columns=[col for col in cols_to_drop if col in X_numeric.columns], errors='ignore')

# Fill all remaining missing values with 0 to ensure calculation stability
X = X.fillna(0)

# Split data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Cleaned feature matrix shape (X_train): {X_train.shape}")
print("✨ Train-test split variables defined successfully in memory!")
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
# Predict actual property prices on test data
y_pred = model.predict(X_test)

# 1. Calculate final R² Score on the actual Euro scale (Decimal scale between 0 and 1)
r2_result = r2_score(y_test, y_pred)

# 2. Calculate real-world error metrics (Euro Scale)
mse_result = mean_squared_error(y_test, y_pred)
mae_result = mean_absolute_error(y_test, y_pred)

print("\n🚀 ====== FINAL REGRESSION MODEL PERFORMANCE METRICS ======")
print(f"📊 R² Score (Coefficient of Determination): {r2_result:.4f}")  # Displayed as a clean decimal (e.g., 0.5234)
print(f"📉 MSE (Mean Squared Error): {mse_result:,.2f} EUR²")
print(f"📊 MAE (Mean Absolute Error): ±{mae_result:,.2f} EUR")
print("===========================================================\n")

print("⏳ 4️⃣ Creating the .joblib file for Linear Regression model...")
# Save and serialize the trained model with the designated name 'LinearV1.joblib'
model_filename = os.path.join(current_dir, 'LinearV2.joblib')
joblib.dump(model, model_filename)

print(f"✅ Linear Regression model successfully exported to:\n   {model_filename}")
print("🚀 The final model is fully frozen and ready for deployment!")