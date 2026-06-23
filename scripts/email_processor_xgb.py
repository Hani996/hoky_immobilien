"""
Email Processor — XGBoost Model
Reads unread Gmail messages, uses Gemini to extract property features,
runs xgb_estate_model.joblib (bundle: model + TargetEncoder + lookup tables)
to predict price, and sends a reply. Prediction is on log scale — reversed with expm1.
"""
import os, json, sys, imaplib, email, smtplib
import numpy as np
import pandas as pd
import joblib
import google.generativeai as genai
from pathlib import Path
from email.mime.text import MIMEText
from dotenv import load_dotenv

# --- Paths ---
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_PATH   = PROJECT_ROOT / "model" / "xgb_estate_model.joblib"

# --- Load environment ---
load_dotenv()
GMAIL_ADDRESS      = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Load bundle ---
print(f"Loading XGB bundle from {MODEL_PATH}...")
bundle   = joblib.load(MODEL_PATH)
model    = bundle["model"]
encoder  = bundle["encoder"]
lookup   = bundle["lookup"]

location_popularity_map     = lookup["location_popularity_map"]
location_popularity_default = lookup["location_popularity_default"]
space_to_county_avg_map     = lookup["space_to_county_avg_map"]
space_to_county_avg_default = lookup["space_to_county_avg_default"]
categorical_cols            = lookup["categorical_cols"]

MODEL_FEATURES = list(model.feature_names_in_)
print(f"✓ Ready. Features: {MODEL_FEATURES}")


def engineer_features(features: dict) -> pd.DataFrame:
    def safe_float(v, default=0.0):
        try: return float(v)
        except: return default

    living_space     = safe_float(features.get("obj_livingSpace"),    120.0)
    year_constructed = safe_float(features.get("obj_yearConstructed"), 1973.0)
    geo_plz          = safe_float(features.get("geo_plz"),            29468.0)
    no_rooms         = safe_float(features.get("obj_noRooms"),         4.0)
    geo_krs          = str(features.get("geo_krs",         "Hannover"))
    obj_regio3       = str(features.get("obj_regio3",      "other"))
    obj_condition    = str(features.get("obj_condition",   "well_kept"))
    obj_firingTypes  = str(features.get("obj_firingTypes", "gas"))
    obj_cellar       = str(features.get("obj_cellar",      "n"))

    # Engineered features — must mirror XGboost_model.py exactly
    obj_age         = 2026 - year_constructed
    space_per_room  = living_space / (no_rooms + 0.1)

    location_popularity = location_popularity_map.get(obj_regio3, location_popularity_default)
    county_avg          = space_to_county_avg_map.get(geo_krs,    space_to_county_avg_default)
    space_to_county_avg = living_space / (county_avg + 0.1)

    row = {
        "obj_yearConstructed": year_constructed,
        "obj_firingTypes":     obj_firingTypes,
        "obj_cellar":          obj_cellar,
        "obj_livingSpace":     living_space,
        "geo_krs":             geo_krs,
        "obj_condition":       obj_condition,
        "geo_plz":             geo_plz,
        "obj_noRooms":         no_rooms,
        "obj_regio3":          obj_regio3,
        "obj_age":             obj_age,
        "space_per_room":      space_per_room,
        "location_popularity": location_popularity,
        "space_to_county_avg": space_to_county_avg,
    }
    X = pd.DataFrame([row])[MODEL_FEATURES]

    # Apply the TargetEncoder from the bundle
    existing_cats = [c for c in categorical_cols if c in X.columns]
    X[existing_cats] = encoder.transform(X[existing_cats])
    return X


def extract_features_via_gemini(email_body: str) -> dict:
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""Extract house features from this email. Output valid JSON only — no markdown.

Email:
{email_body}

Required JSON structure:
{{
    "obj_livingSpace": <float, sqm>,
    "obj_noRooms": <float>,
    "obj_yearConstructed": <float>,
    "obj_condition": <"well_kept" | "mint_condition" | "need_of_renovation" | "refurbished" | "first_time_use_after_refurbishment" | "fully_renovated" | "modernized" | "negotiable" | "ripe_for_demolition">,
    "obj_firingTypes": <"gas" | "oil" | "district_heating" | "electricity" | "geothermal" | "wood" | "unknown">,
    "geo_krs": <string, district/city in Niedersachsen e.g. "Hannover", "Braunschweig", "Göttingen">,
    "obj_regio3": <string, specific sub-district or "other">,
    "geo_plz": <int, German zip code>,
    "obj_cellar": <"y" or "n">
}}"""
    response = gemini_model.generate_content(
        prompt, generation_config={"response_mime_type": "application/json"}
    )
    return json.loads(response.text.strip())


def fetch_unread_emails() -> list:
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
    mail.select("inbox")
    _, message_ids = mail.search(None, "UNSEEN")
    if not message_ids[0]:
        print("No unread emails.")
        mail.logout()
        return []
    emails = []
    for msg_id in message_ids[0].split():
        _, msg_data = mail.fetch(msg_id, "(RFC822)")
        raw = email.message_from_bytes(msg_data[0][1])
        body = ""
        if raw.is_multipart():
            for part in raw.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode(errors="ignore")
                    break
        else:
            body = raw.get_payload(decode=True).decode(errors="ignore")
        emails.append({"from": raw["From"], "subject": raw["Subject"], "body": body})
    mail.logout()
    return emails


def send_email(to: str, subject: str, body: str):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = GMAIL_ADDRESS
    msg["To"]      = to
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, to, msg.as_string())
    print(f"  ✓ Reply sent to {to}")


if __name__ == "__main__":
    print("Fetching unread emails...")
    emails = fetch_unread_emails()
    print(f"Found {len(emails)} unread email(s)\n")

    for em in emails:
        print(f"Processing: {em['subject']} | from: {em['from']}")
        try:
            features = extract_features_via_gemini(em["body"])
            X = engineer_features(features)
            prediction = max(0.0, float(np.expm1(model.predict(X)[0])))
            print(f"  → Predicted price: {prediction:,.2f} EUR")

            summary = json.dumps(features, indent=2, ensure_ascii=False)
            reply = f"""Hello,

Thank you for your property inquiry: "{em['subject']}"

Our XGBoost model has evaluated the property:

  Predicted Valuation:    {prediction:,.2f} EUR
  Living Space:           {features.get('obj_livingSpace')} sqm
  Rooms:                  {features.get('obj_noRooms')}
  Year Constructed:       {features.get('obj_yearConstructed')}
  Condition:              {features.get('obj_condition')}
  Heating Type:           {features.get('obj_firingTypes')}
  District:               {features.get('geo_krs')} ({features.get('geo_plz')})
  Cellar:                 {features.get('obj_cellar')}

Extracted Features:
{summary}

Best regards,
Automated Property Evaluator — XGBoost
"""
            send_email(em["from"], f"Price Estimate (XGBoost): {em['subject']}", reply)

        except Exception as e:
            print(f"  ✗ Error: {e}")
        print("-" * 40)
