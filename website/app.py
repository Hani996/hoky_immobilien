import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from pathlib import Path

app = Flask(__name__, template_folder="templates", static_folder="static")

CURRENT_DIR = Path(__file__).parent.resolve()
MODEL_PATH = CURRENT_DIR.parent / "model" / "xgb_estate_model.joblib"

print(f"Loading model bundle from {MODEL_PATH}...")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
encoder = bundle["encoder"]
lookup = bundle["lookup"]

location_popularity_map = lookup["location_popularity_map"]
location_popularity_default = lookup["location_popularity_default"]
space_to_county_avg_map = lookup["space_to_county_avg_map"]
space_to_county_avg_default = lookup["space_to_county_avg_default"]
categorical_cols = lookup["categorical_cols"]

MODEL_FEATURES = list(model.feature_names_in_)

print(f"✓ Ready. Features: {MODEL_FEATURES}")


def engineer_web_features(raw_data: dict) -> pd.DataFrame:
    def safe_float(val, default):
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    living_space = safe_float(raw_data.get("obj_livingSpace"), 120.0)
    year_constructed = safe_float(raw_data.get("obj_yearConstructed"), 1973.0)
    geo_plz = safe_float(raw_data.get("geo_plz"), 29468.0)
    no_rooms = safe_float(raw_data.get("obj_noRooms"), 4.0)

    geo_krs = str(raw_data.get("geo_krs", "Hannover_Kreis"))
    obj_regio3 = str(raw_data.get("obj_regio3", "other"))
    obj_condition = str(raw_data.get("obj_condition", "well_kept"))
    obj_firingTypes = str(raw_data.get("obj_firingTypes", "gas"))
    obj_cellar = str(raw_data.get("obj_cellar", "n"))

    obj_age = 2026 - year_constructed
    space_per_room = living_space / (no_rooms + 0.1)

    location_popularity = location_popularity_map.get(
        obj_regio3,
        location_popularity_default
    )

    county_avg = space_to_county_avg_map.get(
        geo_krs,
        space_to_county_avg_default
    )

    space_to_county_avg = living_space / (county_avg + 0.1)

    row = {
        "obj_yearConstructed": year_constructed,
        "obj_firingTypes": obj_firingTypes,
        "obj_cellar": obj_cellar,
        "obj_livingSpace": living_space,
        "geo_krs": geo_krs,
        "obj_condition": obj_condition,
        "geo_plz": geo_plz,
        "obj_noRooms": no_rooms,
        "obj_regio3": obj_regio3,
        "obj_age": obj_age,
        "space_per_room": space_per_room,
        "location_popularity": location_popularity,
        "space_to_county_avg": space_to_county_avg,
    }

    X = pd.DataFrame([row])[MODEL_FEATURES]

    existing_cats = [c for c in categorical_cols if c in X.columns]
    X[existing_cats] = encoder.transform(X[existing_cats])

    return X


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/vorhersage")
def vorhersage():
    return render_template("vorhersage.html")


@app.route("/datenschutz")
def datenschutz():
    return render_template("datenschutz.html")


@app.route("/kontakt")
def kontakt():
    return render_template("kontakt.html")


@app.route("/mehrÜberImmobilien")
def mehrÜberImmobilien():
    return render_template("mehrÜberImmobilien.html")

@app.route("/niedersachsen_price_heatmap")
def niedersachsen_price_heatmap():
    return render_template("niedersachsen_price_heatmap.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        raw_data = request.get_json()
        X = engineer_web_features(raw_data)

        prediction_log = model.predict(X)[0]
        prediction_eur = max(0.0, float(np.expm1(prediction_log)))

        return jsonify({
            "predicted_price": f"{prediction_eur:,.2f} EUR"
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)