import os
import sys
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

# --- 1. PREVENT THE WINSORIZER TRAIN_MODEL SERIALIZATION TRAP ---
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.lower_q, axis=0)
        self.upper_ = np.nanquantile(X, self.upper_q, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)

import __main__
__main__.Winsorizer = Winsorizer

# --- 2. INITIALIZE FLASK & MODEL ---
app = Flask(__name__, template_folder="templates", static_folder="static")

MODEL_PATH = Path(r"C:\Users\omar_\Documents\hoky_immobilien\scripts\price_model.joblib")
print(f"Loading joblib model pipeline from {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)
print("✓ Model successfully initialized!")

def engineer_web_features(raw_data: dict) -> pd.DataFrame:
    """Takes web inputs safely and engineers them, forcing correct data types 

    to eliminate the string-to-float conversion crash.
    """
    
    def safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    # 1. Parse fields cleanly from user's web submission
    living_space = safe_float(raw_data.get('obj_livingSpace'), default=120.0)
    no_rooms = safe_float(raw_data.get('obj_noRooms'), default=4.0)
    year_constructed = safe_float(raw_data.get('obj_yearConstructed'), default=1995)
    
    try:
        geo_plz = int(safe_float(raw_data.get('geo_plz'), default=30159))
    except (TypeError, ValueError):
        geo_plz = 30159

    # 2. Build the exact dictionary feature configuration
    processed = {
        "obj_livingSpace": living_space,
        "obj_noRooms": no_rooms,
        "geo_krs": str(raw_data.get('geo_krs', 'Hannover')),
        "obj_regio3": str(raw_data.get('obj_regio3', 'other')),
        "obj_condition": str(raw_data.get('obj_condition', 'well_kept')),
        
        # Calculate interaction terms expected by your model pipeline
        "building_age": 2020.0 - year_constructed,
        "plz_prefix": str(geo_plz).zfill(5)[:2],
        "living_space_per_room": living_space / no_rooms if no_rooms > 0 else 30.0,
        
        # CRITICAL FIX: Convert strings ('y'/'n') to integers (1/0) directly here 
        # so scikit-learn numeric pipes don't encounter string flags down the line.
        "obj_newlyConst": 0,       # Map 'n' -> 0
        "obj_cellar": 1,           # Map 'y' -> 1
        "obj_barrierFree": 0,      # Map 'n' -> 0
        
        # Technical feature configurations for internet properties
        "obj_telekomInternetProductAvailable": 1,
        "obj_telekomUploadSpeed": 40.0,
        "obj_telekomDownloadSpeed": 100.0,
        "obj_firingTypes": "gas",
        
        # Missing data flag properties expected by training footprints
        "obj_firingTypes_missing": 0,
        "obj_condition_missing": 0,
        "obj_telekomInternetProductAvailable_missing": 0,
        "obj_telekomUploadSpeed_missing": 0,
        "obj_telekomDownloadSpeed_missing": 0
    }
    
    # Return as single row input array
    return pd.DataFrame([processed])# --- 4. ROUTES ---
@app.route('/')
def home():
    return render_template('vorhersage.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw_data = request.get_json()
        df_engineered = engineer_web_features(raw_data)
        
        # Compute estimation pipeline
        prediction = model.predict(df_engineered)[0]
        
        formatted_price = f"{prediction:,.2f} EUR"
        return jsonify({"predicted_price": formatted_price})
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)