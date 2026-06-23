import os
import time
import numpy as np
import pandas as pd
import joblib
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("⏳ 1️⃣ Loading real estate dataset from SQLite...")

current_dir = os.path.dirname(os.path.abspath(__file__))
db_filename = os.path.normpath(os.path.join(current_dir, "../data/db/prices.db"))

print(f"🗄️ Connecting to database at: {db_filename}")
conn = sqlite3.connect(db_filename)
df = pd.read_sql_query("SELECT * FROM house_prices", conn)
conn.close()

# SQLite stores booleans as 0/1 integers — cast back to object to match CSV dtype
# so select_dtypes doesn't accidentally include it as a numeric feature
if 'obj_telekomInternetProductAvailable' in df.columns:
    df['obj_telekomInternetProductAvailable'] = df['obj_telekomInternetProductAvailable'].astype(object)

# Keep only the features this model uses
all_possible_features = [
    'obj_purchasePrice', 'obj_livingSpace', 'obj_yearConstructed',
    'geo_krs', 'obj_condition', 'obj_regio3', 'obj_noRooms',
    'obj_heatingType', 'obj_firingTypes', 'obj_hasKitchen', 'obj_cellar'
]
features_to_keep = [col for col in all_possible_features if col in df.columns]
df = df[features_to_keep].copy()
print(f"📊 Initial shape based on available features: {df.shape}")


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


df = remove_outliers_by_quantile(df)
print(f"🧹 Cleaned shape: {df.shape}")

print("⏳ 2️⃣ Preprocessing features...")

# Fill missing numeric values with column medians
for col in ['obj_yearConstructed', 'obj_noRooms']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values with 'unknown'
categorical_cols = ['geo_krs', 'obj_condition', 'obj_regio3', 'obj_heatingType', 'obj_firingTypes']
existing_categorical_cols = [col for col in categorical_cols if col in df.columns]
for col in existing_categorical_cols:
    df[col] = df[col].fillna('unknown')

# Normalize boolean columns to 0/1 integers
bool_mapping = {
    'y': 1, 'yes': 1, 'j': 1, 'ja': 1, True: 1, 1: 1, '1': 1,
    'n': 0, 'no': 0, 'nein': 0, False: 0, 0: 0, '0': 0
}
for col in ['obj_hasKitchen', 'obj_cellar']:
    if col in df.columns:
        df[col] = df[col].map(bool_mapping).fillna(0).astype(int)

print(f"Remaining NaN count: {df.isna().sum().sum()}")

# Log-transform target
y_log = np.log1p(df['obj_purchasePrice'])
X = df.drop(columns=['obj_purchasePrice'], errors='ignore').fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

print("⏳ 3️⃣ Fitting TargetEncoder on training split...")
encoder = TargetEncoder(smooth="auto", random_state=42)
X_train[existing_categorical_cols] = encoder.fit_transform(X_train[existing_categorical_cols], y_train)
X_test[existing_categorical_cols]  = encoder.transform(X_test[existing_categorical_cols])
print(f"✅ Encoding done | X_train: {X_train.shape} | X_test: {X_test.shape}")

print("⏳ 4️⃣ Training Random Forest V2...")
start_time = time.time()

rf_model_v2 = RandomForestRegressor(
    n_estimators=600,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=1
)
rf_model_v2.fit(X_train, y_train)

duration = time.time() - start_time
print(f"🌲 Training finished in {duration:.2f}s")

print("⏳ 5️⃣ Evaluating on test set...")
y_pred_log      = rf_model_v2.predict(X_test)
y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred_log)

train_r2    = rf_model_v2.score(X_train, y_train)
test_r2     = r2_score(y_test_original, y_pred_original)
mae_original = mean_absolute_error(y_test_original, y_pred_original)
rmse_original = np.sqrt(mean_squared_error(y_test_original, y_pred_original))

print("\n🚀 ====== FINAL REGRESSION MODEL PERFORMANCE METRICS ======")
print(f"🏋️  R² on training data (log scale): {train_r2 * 100:.2f}%")
print(f"🧪  R² on test data   (EUR scale):   {test_r2 * 100:.2f}%")
print(f"🎯  RMSE: ±{rmse_original:,.2f} EUR")
print(f"📊  MAE:  ±{mae_original:,.2f} EUR")
print("===========================================================\n")

print("⏳ 6️⃣ Saving all components into a single .joblib bundle...")

# Everything email_processor_rfV2.py needs at inference time is packed into
# one dictionary — no separate encoder file needed.
bundle = {
    "model":             rf_model_v2,
    "encoder":           encoder,
    "categorical_cols":  existing_categorical_cols,
}

model_filename = os.path.join(current_dir, 'rf_v2_estate_model.joblib')
joblib.dump(bundle, model_filename)

print(f"✅ Bundle saved to: {model_filename}")
print("   Contents: model + encoder (single file)")
print("🚀 Ready for deployment!")