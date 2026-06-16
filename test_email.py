import imaplib
import email
import os
import smtplib
import json
import google.generativeai as genai
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()

GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def extract_house_features(email_body):
    model = genai.GenerativeModel("gemini-2.5-computer-use-preview-10-2025")
    response = model.generate_content(f"""Extract house features from this email and return ONLY valid JSON, no extra text.

Email: {email_body}

Return this exact structure (use null for missing values):
{{
    "obj_livingSpace": <float, house size in sqm>,
    "obj_noRooms": <float, number of rooms>,
    "obj_yearConstructed": <float, year built>,
    "obj_condition": <string: first_time_use / refurbished / no_information>,
    "obj_heatingType": <string: central_heating / heat_pump / stove_heating / no_information>,
    "obj_regio1": <string, German state e.g. Sachsen / Bayern>,
    "obj_zipCode": <float, zip code>,
    "obj_newlyConst": <"y" or "n">,
    "obj_cellar": <"y" or "n">,
    "obj_barrierFree": <"y" or "n">
}}""")

    # Gemini syntax — just response.text, not response.content[0].text
    raw = response.text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

def send_email(to, subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = to

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, to, msg.as_string())
        print(f"✓ Email sent to {to}")

def fetch_unread_emails():
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
                    body = part.get_payload(decode=True).decode()
                    break
        else:
            body = raw.get_payload(decode=True).decode()

        emails.append({
            "from": raw["From"],
            "subject": raw["Subject"],
            "body": body
        })

    mail.logout()
    return emails

if __name__ == "__main__":
    print("Connecting to Gmail...")
    emails = fetch_unread_emails()
    print(f"Found {len(emails)} unread email(s)\n")

    for em in emails:
        print(f"Processing email from: {em['from']}")
        print(f"Subject: {em['subject']}")

        try:
            features = extract_house_features(em["body"])
            print("Extracted features:")
            for key, value in features.items():
                print(f"  {key}: {value}")

            send_email(
                to=GMAIL_ADDRESS,
                subject=f"Extracted Features: {em['subject']}",
                body=f"Extracted the following features:\n\n{json.dumps(features, indent=2)}"
            )

        except Exception as e:
            print(f"  ✗ Failed to extract features: {e}")

        print()