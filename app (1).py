import nltk
import spacy
import sqlite3
import requests
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from datetime import datetime
import smtplib
from email.mime.text import MIMEText

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Initialize Flask app
app = Flask(__name__)

# List of Ghanaian Cities for Better Location Extraction
GHANA_LOCATIONS = {"Accra", "Kumasi", "Tamale", "Takoradi", "Cape Coast", "Sunyani", "Techiman", "Bolgatanga", "Wa", "Tafo", "Bantama"}

# Predefined List of Common Ghanaian Names
GHANAIAN_NAMES = {"Kwame", "Kofi", "Yaw", "Sylvester", "Isaac", "Kojo", "Kwabena", "Kwaku", "Ama", "Akua", "Abena", "Adwoa", "Esi", "Afia", "Nana", "Mensah", "Boateng", "Asante", "Owusu", "Frimpong", "Darko", "Boadu", "Antwi", "Appiah", "Dapaah", "Osei", "Baah", "Agyemang", "Gyasi", "Twumasi", "Acheampong", "Sarpong", "Bonsu", "Ofori"}

# Training Data for Intent Classification
training_data = [
    ("How do I stay safe during a flood?", "safety_guideline"),
    ("What should I do during an earthquake?", "safety_guideline"),
    ("There’s a fire in my neighborhood!", "incident_report"),
    ("How do I report an emergency?", "faq"),
    ("Heavy rains are causing floods!", "incident_report"),
    ("Send me disaster alerts.", "disaster_alert"),
]

# Train a simple text classification model
texts, labels = zip(*training_data)
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(texts, labels)

# Function to Predict Intent
def classify_intent(user_message):
    prediction = model.predict([user_message])
    return prediction[0] if prediction[0] in [label for _, label in training_data] else None

# Function to Extract Location
def extract_location(text):
    """Extracts location from user message using NLP and keyword matching."""
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE" and ent.text in GHANA_LOCATIONS:
            return ent.text
    found_cities = [city for city in GHANA_LOCATIONS if city.lower() in text.lower()]
    return found_cities[0] if found_cities else "Unknown Location"

# Function to Extract Name
def extract_name(text):
    """Extracts name from user message using NLP and a predefined name list."""
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON" and ent.text in GHANAIAN_NAMES:
            return ent.text
    found_names = [name for name in GHANAIAN_NAMES if name.lower() in text.lower()]
    return found_names[0] if found_names else "Anonymous"

# Database Initialization
def init_db():
    conn = sqlite3.connect('nadmo_reports.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS reports 
                      (id INTEGER PRIMARY KEY, name TEXT, location TEXT, incident TEXT, date TEXT, time TEXT)''')
    conn.commit()
    cursor.close()
    conn.close()

# Function to Store Reports
def save_report(name, location, incident):
    conn = sqlite3.connect('nadmo_reports.db')
    cursor = conn.cursor()
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    cursor.execute("INSERT INTO reports (name, location, incident, date, time) VALUES (?, ?, ?, ?, ?)",(name, location, incident, date, time))
    conn.commit()
    cursor.close()
    conn.close()

# Email Integration for Emergency Services
def send_email(subject, body, to_email):
    """Send an email."""
    from_email = "sylvesterosei542@gmail.com"
    password = "ynoe rbfj uucv klrs"
    
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())

# Initialize Database
init_db()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"reply": "Invalid request. Please provide a message."})

    user_input = data.get("message", "").strip().lower()
    if not user_input:
        return jsonify({"reply": "Message cannot be empty."})

    # Greet new users
    if user_input in ["hello", "hi"]:
        return jsonify({"reply": "How may I help you?"})
    
    # Respond to chatbot name inquiry
    if "your name" in user_input or "who are you" in user_input:
        return jsonify({"reply": "I'm NADMO Chatbot for Ghana."})
    
    # NLP Processing
    intent = classify_intent(user_input)
    if intent not in ["safety_guideline", "faq", "incident_report", "disaster_alert"]:
        return jsonify({"reply": "Please I don't have any idea about that😊. I can only help you with things relating to Disaster😊."})
    
    location = extract_location(user_input)
    name = extract_name(user_input)
    response_msg = ""

    if intent == "safety_guideline":
        response_msg = "Stay indoors during floods. Avoid touching electrical appliances."
    elif intent == "faq":
        response_msg = "You can report an emergency by sending a message with details."
    elif intent == "incident_report":
        save_report(name, location, user_input)
        response_msg = f"Incident report received for {location} by {name}. Authorities have been notified."
        # Send email to emergency services
        send_email("New Incident Report", f"Incident: {user_input}\nLocation: {location}\nReported by: {name}", "emergency@example.com")
    elif intent == "disaster_alert":
        response_msg = "You will receive alerts when a disaster is detected."

    return jsonify({"reply": response_msg})

if __name__ == "__main__":
    app.run(debug=False)