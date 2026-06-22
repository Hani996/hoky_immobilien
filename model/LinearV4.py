import os
import time  # Library for tracking execution time
import numpy as np
import pandas as pd
import joblib  # Library for saving and loading models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("⏳ 1️⃣ Loading, cleaning, and preparing dataset for Linear Regression...")

# Automatically detect the current directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go one step back (..) and enter the 'data' folder to read the CSV file safely
data_path = os.path.normpath(os.path.join(current_dir, "../data/price_clean.csv"))

print(f"📂 Reading data from: {data_path}")

# Read the raw dataset
df = pd.read_csv(data_path)

# Define the lower and upper quantile thresholds (1% and 99%) for outlier removal
lower_q = 0.01
upper_q = 0.99

# Calculate dynamic thresholds for purchase price
price_low = df['obj_purchasePrice'].quantile(lower_q)
price_high = df['obj_purchasePrice'].quantile(upper_q)

# Calculate dynamic thresholds for living space
space_low = df['obj_livingSpace'].quantile(lower_q)
space_high = df['obj_livingSpace'].quantile(upper_q)

print(f"📊 Calculated Price Thresholds: from {price_low:,.0f} EUR to {price_high:,.0f} EUR")
print(f"📐 Calculated Space Thresholds: from {space_low:.1f}m² to {space_high:.1f}m²")

# Apply quantile filtering to safely eliminate statistical outliers
df = df[
    (df['obj_purchasePrice'] >= price_low) & 
    (df['obj_purchasePrice'] <= price_high)
]

df = df[
    (df['obj_livingSpace'] >= space_low) & 
    (df['obj_livingSpace'] <= space_high)
]

print(f"🧹 Dataset shape after removing statistical outliers: {df.shape}")

# Select categorical features that impact property prices
categorical_cols = ['obj_regio1', 'obj_regio3', 'obj_condition']

# Convert categorical text features into numeric 0 and 1 (Dummy variables) using drop_first=True
dummies = pd.get_dummies(df[categorical_cols], drop_first=True, dtype=int)

# Concatenate the dummy columns horizontally with the filtered dataframe
df = pd.concat([df, dummies], axis=1)

# Isolate the target price variable (y)
y = df['obj_purchasePrice']

# Keep only numeric and boolean columns (including the newly created dummies)
X_numeric = df.select_dtypes(include=[np.number, bool]).copy()

# Drop the target price, price per sqm, and original text columns to prevent data leakage
cols_to_drop = ['obj_purchasePrice', 'obj_purchasePrice_per_qm', 'obj_regio1']
X = X_numeric.drop(columns=[col for col in cols_to_drop if col in X_numeric.columns], errors='ignore')


# تعويض القيم المفقودة في الأعمدة الرقمية بالقيمة الوسطية (Median) لكل عمود
for col in X.columns:
    # التأكد من أن العمود رقمي وليس متغير فئوي (Dummy) مرمز بـ 0 و 1
    if X[col].dtype in [np.float64, np.int64]:
        mean_value = X[col].mean()
        X[col] = X[col].fillna(mean_value)


# Split data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Cleaned feature matrix shape (X_train): {X_train.shape}")
print("-" * 60)

print("⏳ 2️⃣ Training the Linear Regression model with filtered data...")
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
print(f"📊 R² Score (Coefficient of Determination): {r2_result:.4f}")  # Clean decimal output (e.g., 0.5234)
print(f"📉 MSE (Mean Squared Error): {mse_result:,.2f} EUR²")
print(f"📊 MAE (Mean Absolute Error): ±{mae_result:,.2f} EUR")
print("===========================================================\n")

print("⏳ 4️⃣ Creating the .joblib file for the optimized Linear Regression model...")
# 🌟 Save and serialize the trained Linear Regression model as LinearV4.joblib
model_filename = os.path.join(current_dir, 'LinearV4.joblib')
joblib.dump(model, model_filename)

print(f"✅ Linear Regression model successfully exported to:\n   {model_filename}")
print("🚀 The final model is fully frozen and ready for deployment!")