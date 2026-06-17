"""
predict.py
===========
Loads the trained pipeline from train_model.py and predicts purchase
prices for new property listings.

Requires train_model.py to be in the same folder (or on PYTHONPATH) so
that the Winsorizer class and engineer_features() function it defines
are importable. The Winsorizer redefinition below isn't actually used
directly here -- it just needs to exist so joblib can deserialize the
pipeline, since the original training run pickled it under the
`__main__` module.
"""
from pathlib import Path
import sys
import pandas as pd
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# 1. Resolve paths dynamically so it doesn't matter where you run the script from
SCRIPT_DIR = Path(__file__).resolve().parent      # Points to .../hoky_immobilien/scripts
PROJECT_ROOT = SCRIPT_DIR.parent                  # Points to .../hoky_immobilien

# Add the project root to Python's search path so the 'model' import works flawlessly
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.train_model import engineer_features  # Now Python can find this!

# Define the precise path to your saved joblib file
MODEL_PATH = PROJECT_ROOT / "model" / "price_model.joblib"
# 2. Keep your Winsorizer class here
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


# 3. Load the trained pipeline safely
print(f"Loading model from {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)

# Note on category values: geo_krs and obj_regio3 need to roughly match
# the strings the model was trained on (e.g. "Hannover_Kreis" for the
# surrounding district vs. "Hannover" for the city itself; many cities
# only have "other" as their obj_regio3 value rather than a named
# neighborhood, since the source data doesn't break them down further).
# An unrecognized string isn't an error -- it just gets silently
# treated as "unknown", so the location signal for that field is lost.
new_properties = [
    {
        "obj_livingSpace": 120.0,
        "obj_noRooms": 4.0,
        "obj_yearConstructed": 1995,
        "geo_krs": "Hannover",
        "obj_regio3": "other",
        "geo_plz": 30159,
        "obj_condition": "well_kept",
        "obj_firingTypes": "gas",
        "obj_newlyConst": "n",
        "obj_cellar": "y",
        "obj_barrierFree": "n",
        "obj_telekomInternetProductAvailable": True,
        "obj_telekomUploadSpeed": 40.0,
        "obj_telekomDownloadSpeed": 100.0,
    },
    {
        "obj_livingSpace": 85.0,
        "obj_noRooms": 3.0,
        "obj_yearConstructed": 2018,
        "geo_krs": "Braunschweig",
        "obj_regio3": "other",
        "geo_plz": 38100,
        "obj_condition": "first_time_use",
        "obj_firingTypes": "district_heating",
        "obj_newlyConst": "y",
        "obj_cellar": "n",
        "obj_barrierFree": "y",
        "obj_telekomInternetProductAvailable": True,
        "obj_telekomUploadSpeed": 100.0,
        "obj_telekomDownloadSpeed": 250.0,
    },
]

df_new = pd.DataFrame(new_properties)

# Apply the exact same feature engineering used during training
# (building_age, plz_prefix, living_space_per_room, missing-value
# flags) -- the model can't predict on raw columns alone.
df_new_engineered = engineer_features(df_new)

print("Calculating price predictions...")
predictions = model.predict(df_new_engineered)

for i, price in enumerate(predictions):
    print(f"\nProperty {i + 1} Details:")
    print(f"  Location: {df_new.iloc[i]['geo_krs']} ({df_new.iloc[i]['geo_plz']})")
    print(f"  Size: {df_new.iloc[i]['obj_livingSpace']} sqm | Rooms: {df_new.iloc[i]['obj_noRooms']}")
    print(f"  --> Predicted Purchase Price: {price:,.2f} EUR")