import os
import pandas as pd
import numpy as np
import joblib  # Library for saving and loading models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

""" print("⏳ 1️⃣ Loading and cleaning real estate dataset...")

# Automatically detect the current directory where this script (model) is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go one step back (..) and then enter the 'data' folder to read the CSV file safely
data_path = os.path.normpath(os.path.join(current_dir, "../data/price_clean.csv"))

print(f"📂 Reading data from the following path:\n   {data_path}")

# Read the dataset
df = pd.read_csv(data_path) """

# Select categorical features and convert them into dummy/indicator variables (0 and 1)
categorical_cols = ['obj_regio1', 'obj_regio3', 'obj_condition']
dummies = pd.get_dummies(df[categorical_cols], drop_first=True, dtype=int)

# Concatenate dummy columns horizontally with the original dataframe
df = pd.concat([df, dummies], axis=1)

# Isolate the target variable (Purchase Price) to predict
y = df['obj_purchasePrice']

# Keep only numeric and boolean columns for safe mathematical calculations
X_numeric = df.select_dtypes(include=[np.number, bool]).copy()

# Drop the target, price per qm, and high-level text columns to prevent model errors
cols_to_drop = ['obj_purchasePrice', 'obj_purchasePrice_per_qm', 'obj_regio1']
X = X_numeric.drop(columns=[col for col in cols_to_drop if col in X_numeric.columns], errors='ignore')

# Fill any remaining missing values with 0 to ensure stability
X = X.fillna(0)

print(f"📊 Cleaned feature matrix shape (X): {X.shape}")

print("⏳ 2️⃣ Splitting dataset and training the Linear Regression model...")
# Split data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Calculate model performance using R² Score
accuracy = model.score(X_test, y_test)
print(f"🎯 Success! Model trained. R² Score on test data: {accuracy * 100:.2f}%")
print("-" * 60)

print("⏳ 3️⃣ Creating the .joblib file for model deployment...")
# Save and serialize the trained model inside the current 'model' directory
model_filename = os.path.join(current_dir, 'linear_estate_model.joblib')
joblib.dump(model, model_filename)

print(f"✅ Model successfully exported to:\n   {model_filename}")
print("🚀 The model is now frozen and ready for instant deployment and predictions!")