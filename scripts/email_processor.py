import os
import imaplib
import email
import smtplib
import json
from email.mime.text import MIMEText
from dotenv import load_dotenv
import google.generativeai as genai

# 1. Load environment variables
load_dotenv()

GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

# Initialize Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def extract_house_features(email_body):
    """
    Uses Gemini 2.5 Flash to parse unstructured email body text into 
    a structured JSON object matching the SQLite schema specifications.
    """
    # Using the correct, recommended free tier model
    model = genai.GenerativeModel("gemini-2.5-flash")
    
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
        "obj_heatingType": <string: central_heating / heat_pump / stove_heating / district_heating / gas / oil / no_information>,
        "obj_regio1": <string, German state e.g. Sachsen / Bayern / Niedersachsen>,
        "obj_zipCode": <float, zip code>,
        "obj_newlyConst": <"y" or "n">,
        "obj_cellar": <"y" or "n">,
        "obj_barrierFree": <"y" or "n">
    }}"""

    # Forces the API to natively respond with valid JSON text (no markdown ``` formatting)
    response = model.generate_content(
        prompt, 
        generation_config={"response_mime_type": "application/json"}
    )

    raw_json = response.text.strip()
    return json.loads(raw_json)

def send_email(to, subject, body):
    """
    Sends a confirmation email using Gmail's SMTP server.
    """
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = to

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, to, msg.as_string())
        print(f"  ✓ Confirmation email sent to {to}")

def fetch_unread_emails():
    """
    Connects via IMAP to look for unread messages and extracts their plaintext bodies.
    """
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
    mail.select("inbox")

    # Search for all unread/unseen messages
    _, message_ids = mail.search(None, "UNSEEN")

    if not message_ids[0]:
        print("No unread emails found.")
        mail.logout()
        return []

    emails = []
    # Loop through each message ID found
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
    print("Connecting to Gmail and searching for unread emails...")
    emails = fetch_unread_emails()
    print(f"Found {len(emails)} unread email(s)\n")

    for em in emails:
        print(f"Processing email from: {em['from']}")
        print(f"Subject: {em['subject']}")

        try:
            # 1. Run extraction via Gemini API
            features = extract_house_features(em["body"])
            print("Extracted features successfully:")
            for key, value in features.items():
                print(f"  {key}: {value}")

            # 2. Format a summary string and send back to yourself
            email_summary = json.dumps(features, indent=2, ensure_ascii=False)
            send_email(
                to=GMAIL_ADDRESS,
                subject=f"Extracted Features: {em['subject']}",
                body=f"Extracted the following features from the email:\n\n{email_summary}"
            )

        except Exception as e:
            print(f"  ✗ Failed to extract or send data: {e}")

        print("-" * 40)