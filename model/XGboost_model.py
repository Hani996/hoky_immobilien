import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
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

df_raw = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

# 3. Always close the database connection when done
conn.close()
""" 
print("⏳ 1️⃣ Loading and preparing real estate dataset for Regression...")

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.normpath(os.path.join(current_dir, "../data/price_clean.csv"))

print(f"📂 Reading data from: {data_path}")
df_raw = pd.read_csv(data_path)
 """
features_to_keep = [
    'obj_purchasePrice', 'obj_livingSpace', 'obj_yearConstructed',
    'geo_krs', 'geo_plz', 'obj_condition', 'obj_regio3', 'obj_noRooms',
    'obj_cellar', 'obj_firingTypes'
]
df_raw = df_raw[df_raw.columns.intersection(features_to_keep)].copy()


def remove_outliers_by_quantile(dataframe, lower_q: float = 0.01, upper_q: float = 0.99):
    df_filtered = dataframe.copy()
    price_low  = df_filtered['obj_purchasePrice'].quantile(lower_q)
    price_high = df_filtered['obj_purchasePrice'].quantile(upper_q)
    space_low  = df_filtered['obj_livingSpace'].quantile(lower_q)
    space_high = df_filtered['obj_livingSpace'].quantile(upper_q)
    print(f"📊 Price threshold: {price_low:,.0f} EUR – {price_high:,.0f} EUR")
    print(f"📐 Space threshold: {space_low:.1f}m² – {space_high:.1f}m²")
    return df_filtered[
        (df_filtered['obj_purchasePrice'] >= price_low) &
        (df_filtered['obj_purchasePrice'] <= price_high) &
        (df_filtered['obj_livingSpace']   >= space_low)  &
        (df_filtered['obj_livingSpace']   <= space_high)
    ]


df_clean = remove_outliers_by_quantile(df_raw)
print(f"🧹 Cleaned shape: {df_clean.shape}")

print("⏳ 2️⃣ Engineering features and building lookup tables...")

df_clean['obj_age']        = 2026 - df_clean['obj_yearConstructed']
df_clean['space_per_room'] = df_clean['obj_livingSpace'] / (df_clean['obj_noRooms'] + 0.1)

# --- Lookup tables built from the FULL cleaned dataset (before split) ---
# Stored so app.py can compute location_popularity and space_to_county_avg
# for a single web-form row without needing the whole training set at runtime.
location_popularity_map     = df_clean['obj_regio3'].value_counts().to_dict()
location_popularity_default = float(np.median(list(location_popularity_map.values())))

space_to_county_avg_map     = df_clean.groupby('geo_krs')['obj_livingSpace'].mean().to_dict()
space_to_county_avg_default = float(np.median(list(space_to_county_avg_map.values())))

df_clean['location_popularity'] = df_clean['obj_regio3'].map(location_popularity_map)
df_clean['space_to_county_avg'] = df_clean['obj_livingSpace'] / (
    df_clean['geo_krs'].map(space_to_county_avg_map) + 0.1
)

y_log = np.log1p(df_clean['obj_purchasePrice'])
X_raw = df_clean.drop(columns=['obj_purchasePrice'], errors='ignore')

X_raw = X_raw.fillna(X_raw.median(numeric_only=True))
X_raw = X_raw.fillna('missing')

X_train, X_test, y_train, y_test = train_test_split(X_raw, y_log, test_size=0.2, random_state=42)

print("⏳ 3️⃣ Fitting TargetEncoder on training split...")
categorical_cols = ['geo_krs', 'geo_plz', 'obj_condition', 'obj_regio3', 'obj_cellar', 'obj_firingTypes']
existing_categorical_cols = [c for c in categorical_cols if c in X_train.columns]

encoder = TargetEncoder(smooth="auto", random_state=42)
X_train[existing_categorical_cols] = encoder.fit_transform(X_train[existing_categorical_cols], y_train)
X_test[existing_categorical_cols]  = encoder.transform(X_test[existing_categorical_cols])

print(f"✅ Encoding done | X_train: {X_train.shape} | X_test: {X_test.shape}")

print("⏳ 4️⃣ Training XGBoost Regressor...")
start_time = time.time()

xgb_model = xgb.XGBRegressor(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.02,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=1.5,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

duration = time.time() - start_time
print(f"⚡ Training finished in {duration:.2f}s")

print("⏳ 5️⃣ Evaluating on test set...")
y_pred_log      = xgb_model.predict(X_test)
y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred_log)

r2_original  = r2_score(y_test_original, y_pred_original)
mse_original = mean_squared_error(y_test_original, y_pred_original)
mae_original = mean_absolute_error(y_test_original, y_pred_original)

print("\n🚀 ====== FINAL REGRESSION MODEL PERFORMANCE METRICS ======")
print(f"📊 R²:  {r2_original * 100:.2f}%")
print(f"📉 MSE: {mse_original:,.2f} EUR²")
print(f"📊 MAE: ±{mae_original:,.2f} EUR")
print("===========================================================\n")

print("⏳ 6️⃣ Saving all components into a single .joblib bundle...")

# Everything app.py needs at inference time is packed into one dictionary
# and saved as a single file — no separate encoder or lookup files needed.
bundle = {
    "model":   xgb_model,   # Trained XGBRegressor
    "encoder": encoder,     # Fitted TargetEncoder (must match model's training split)
    "lookup": {             # Precomputed training-set aggregates for engineered features
        "location_popularity_map":     location_popularity_map,
        "location_popularity_default": location_popularity_default,
        "space_to_county_avg_map":     space_to_county_avg_map,
        "space_to_county_avg_default": space_to_county_avg_default,
        "categorical_cols":            existing_categorical_cols,
    }
}

model_filename = os.path.join(current_dir, 'xgb_estate_model.joblib')
joblib.dump(bundle, model_filename)

print(f"✅ Bundle saved to: {model_filename}")
print("   Contents: model + encoder + lookup tables (single file)")
print("🚀 Ready for deployment!")