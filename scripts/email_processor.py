import os
import imaplib
import email
import smtplib
import json
import sys
from pathlib import Path
from email.mime.text import MIMEText
from dotenv import load_dotenv
import pandas as pd
import joblib
import google.generativeai as genai

# --- Resolve Paths for the Model Import ---
SCRIPT_DIR = Path(__file__).resolve().parent
# If this file is sitting in the repo root or scripts folder, adjust PROJECT_ROOT
PROJECT_ROOT = SCRIPT_DIR 

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Adjust this path depending on exactly where 'price_model.joblib' is generated
MODEL_PATH = PROJECT_ROOT / "price_model.joblib"

# Import engineer_features safely from your training file structure
from train_model import engineer_features

# 1. Load environment variables
load_dotenv()

GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

# Initialize Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

import os
import sys
from pathlib import Path
import pandas as pd
import joblib
import google.generativeai as genai
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# --- 1. FORCE WINSORIZER INTO __main__ SCOPE SO JOBLIB CAN DESERIALIZE IT ---
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

# Bind it to the main module explicitly
import __main__
__main__.Winsorizer = Winsorizer


# --- 2. PATH RESOLUTION & ENGINE IMPORT ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Adjust this to point to your precise Windows file path from the log
MODEL_PATH = Path(r"C:\Users\omar_\Documents\hoky_immobilien\scripts\price_model.joblib")

# Import engineer_features safely from your training file structure
from train_model import engineer_features


# --- 3. LOAD THE MODEL PIPELINE ---
print(f"Loading predictive model from {MODEL_PATH}...")
try:
    # Now that __main__.Winsorizer exists, this will load flawlessly!
    model = joblib.load(MODEL_PATH)
    print("✓ Model pipeline successfully loaded!")
except Exception as e:
    print(f"CRITICAL: Failed to load model. Error: {e}")
    model = None

def extract_house_features(email_body):
    """
    Uses Gemini 2.5 Flash to parse unstructured email body text into 
    a structured JSON object matching the training model requirements perfectly.
    """
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    
    prompt = f"""Extract house features from this email. You must output a valid JSON object adhering exactly to the structure below.
    Use null for missing values. Do not wrap the response in markdown blocks like ```json.

    Email content:
    {email_body}

    Required Structure:
    {{
        "obj_livingSpace": <float, house size in sqm>,
        "obj_noRooms": <float, number of rooms>,
        "obj_yearConstructed": <float, year built>,
        "obj_condition": <string: first_time_use / refurbished / well_kept / need_of_renovation / no_information>,
        "obj_firingTypes": <string: central_heating / heat_pump / stove_heating / district_heating / gas / oil / unknown>,
        "geo_krs": <string, District or city in Niedersachsen e.g. Hannover, Braunschweig, Göttingen, Osnabrück>,
        "obj_regio3": <string, set to "other" if specific neighborhood is not known>,
        "geo_plz": <int, German zip code number>,
        "obj_newlyConst": <"y" or "n">,
        "obj_cellar": <"y" or "n">,
        "obj_barrierFree": <"y" or "n">
    }}"""

    response = gemini_model.generate_content(
        prompt, 
        generation_config={"response_mime_type": "application/json"}
    )

    raw_json = response.text.strip()
    extracted_data = json.loads(raw_json)
    
    # Fill in fallback defaults for model dependencies missing from standard emails
    if extracted_data.get("geo_krs") is None:
        extracted_data["geo_krs"] = "Hannover"
    if extracted_data.get("obj_regio3") is None:
        extracted_data["obj_regio3"] = "other"
    if extracted_data.get("geo_plz") is None:
        extracted_data["geo_plz"] = 30159 # Default city center fallback
        
    extracted_data["obj_telekomInternetProductAvailable"] = True
    extracted_data["obj_telekomUploadSpeed"] = 40.0
    extracted_data["obj_telekomDownloadSpeed"] = 100.0
    
    return extracted_data

def send_email(to, subject, body):
    """Sends a response email using Gmail's SMTP server."""
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = to

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, to, msg.as_string())
        print(f"  ✓ Email sent to {to}")

def fetch_unread_emails():
    """Connects via IMAP to look for unread messages and extracts their plaintext bodies."""
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
    mail.select("inbox")

    _, message_ids = mail.search(None, "UNSEEN")

    if not message_ids[0]:
        print("No unread emails found.")
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

        emails.append({
            "from": raw["From"],
            "subject": raw["Subject"],
            "body": body
        })

    mail.logout()
    return emails

if __name__ == "__main__":
    if not model:
        print("Please train your model pipeline via 'train_model.py' before launching.")
        sys.exit(1)

    print("Connecting to Gmail and searching for unread property inquiries...")
    emails = fetch_unread_emails()
    print(f"Found {len(emails)} unread email(s)\n")

    for em in emails:
        print(f"Processing email from: {em['from']}")
        print(f"Subject: {em['subject']}")

        try:
            # 1. Run feature extraction via Gemini API
            features = extract_house_features(em["body"])
            print("Extracted features from text successfully.")

            # 2. Convert features dictionary to Pandas DataFrame for Model
            df_new = pd.DataFrame([features])
            
            # 3. Apply feature engineering step from your pipeline
            df_new_engineered = engineer_features(df_new)
            
            # 4. Predict the property valuation price
            predicted_price = model.predict(df_new_engineered)[0]
            print(f"  --> Predicted Price: {predicted_price:,.2f} EUR")

            # 5. Format the diagnostic report to email back
            features_summary = json.dumps(features, indent=2, ensure_ascii=False)
            
            email_body = f"""Hello,

Thank you for your real-estate inquiry regarding: "{em['subject']}"

Based on the parameters parsed from your description by our AI engine, we have computed a predictive valuation for the property listing.

### Evaluation Summary:
* **Predicted Valuation:** {predicted_price:,.2f} EUR
* **Extracted Living Space:** {features.get('obj_livingSpace')} sqm
* **Total Rooms:** {features.get('obj_noRooms')}
* **Estimated Construction Year:** {features.get('obj_yearConstructed')}
* **Location:** {features.get('geo_krs')} ({int(features.get('geo_plz', 0))})

---
### Full Extracted Features Data Payload:
{features_summary}

Best regards,
Automated Property Evaluator Engine
"""
            # Send the email with predictions back to the sender
            send_email(
                to=em["from"],
                subject=f"Price Prediction: {em['subject']}",
                body=email_body
            )

        except Exception as e:
            print(f"  ✗ Failed to extract or calculate prediction data: {e}")

        print("-" * 40)