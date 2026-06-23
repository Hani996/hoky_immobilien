"""
Email Processor — LinearV4 Model
Reads unread Gmail messages, uses Gemini to extract property features,
runs LinearV4.joblib to predict price, and sends a reply.
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
MODEL_PATH   = PROJECT_ROOT / "model" / "LinearV4.joblib"

# --- Load environment ---
load_dotenv()
GMAIL_ADDRESS      = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Load model ---
print(f"Loading LinearV4 from {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)
MODEL_FEATURES = list(model.feature_names_in_)
print(f"✓ Ready. {len(MODEL_FEATURES)} features.")


def compute_living_space_range(living_space: float) -> int:
    if living_space <= 100:  return 1
    elif living_space <= 120: return 2
    elif living_space <= 140: return 3
    elif living_space <= 160: return 4
    elif living_space <= 180: return 5
    elif living_space <= 220: return 6
    elif living_space <= 260: return 7
    else:                     return 8


def engineer_features(features: dict) -> pd.DataFrame:
    def safe_float(v, default=0.0):
        try: return float(v)
        except: return default

    living_space     = safe_float(features.get("obj_livingSpace"),    120.0)
    year_constructed = safe_float(features.get("obj_yearConstructed"), 1973.0)
    geo_plz          = safe_float(features.get("geo_plz"),            29468.0)
    no_rooms         = safe_float(features.get("obj_noRooms"),         4.0)
    upload_speed     = safe_float(features.get("obj_telekomUploadSpeed"),   10.0)
    download_speed   = safe_float(features.get("obj_telekomDownloadSpeed"), 50.0)
    obj_regio3       = str(features.get("obj_regio3", "other"))
    obj_condition    = str(features.get("obj_condition", "well_kept"))

    # Start with all model features set to 0
    row = {col: 0 for col in MODEL_FEATURES}
    row["obj_livingSpace"]        = living_space
    row["obj_yearConstructed"]    = year_constructed
    row["geo_plz"]                = geo_plz
    row["obj_noRooms"]            = no_rooms
    row["obj_livingSpaceRange"]   = compute_living_space_range(living_space)
    row["obj_telekomUploadSpeed"] = upload_speed
    row["obj_telekomDownloadSpeed"] = download_speed

    # One-hot dummies — set the matching column to 1
    regio3_col    = f"obj_regio3_{obj_regio3}"
    condition_col = f"obj_condition_{obj_condition}"
    if regio3_col    in row: row[regio3_col]    = 1
    if condition_col in row: row[condition_col] = 1

    X = pd.DataFrame([row])[MODEL_FEATURES]
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
    "obj_regio3": <string, specific district in Niedersachsen or "other">,
    "geo_plz": <int, German zip code>,
    "obj_telekomUploadSpeed": <float or null>,
    "obj_telekomDownloadSpeed": <float or null>
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
            prediction = max(0.0, float(model.predict(X)[0]))
            print(f"  → Predicted price: {prediction:,.2f} EUR")

            summary = json.dumps(features, indent=2, ensure_ascii=False)
            reply = f"""Hello,

Thank you for your property inquiry: "{em['subject']}"

Our Linear Regression model (LinearV4) has evaluated the property:

  Predicted Valuation:    {prediction:,.2f} EUR
  Living Space:           {features.get('obj_livingSpace')} sqm
  Rooms:                  {features.get('obj_noRooms')}
  Year Constructed:       {features.get('obj_yearConstructed')}
  Condition:              {features.get('obj_condition')}
  Zip Code:               {features.get('geo_plz')}

Extracted Features:
{summary}

Best regards,
Automated Property Evaluator — LinearV4
"""
            send_email(em["from"], f"Price Estimate (LinearV4): {em['subject']}", reply)

        except Exception as e:
            print(f"  ✗ Error: {e}")
        print("-" * 40)
