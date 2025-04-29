import os
import sqlite3
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
import spacy
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from collections import defaultdict
from textblob import TextBlob

# Initialize Flask app with CORS
app = Flask(__name__)
CORS(app)

# Database setup with WAL mode for better concurrency
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nadmo_reports.db')

def get_db_connection():
    """Create a new database connection with WAL mode enabled"""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn

# Load SpaCy NLP model with error handling
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Expanded Ghanaian locations (over 150 entries)
GHANA_LOCATIONS = {
    # Regions and Major Cities
    "Accra", "Kumasi", "Tamale", "Takoradi", "Tema", "Cape Coast", "Sunyani",
    "Ho", "Wa", "Bolgatanga", "Koforidua", "Techiman", "Nkawkaw", "Dome",
    
    # Regional Capitals
    "Sekondi", "Obuasi", "Tarkwa", "Axim", "Elmina", "Saltpond", "Winneba",
    "Mampong", "Konongo", "Ejura", "Bekwai", "Agona Swedru", "Nsawam",
    
    # District Capitals
    "Aburi", "Ada", "Akatsi", "Akropong", "Akwatia", "Asamankese", "Asankragwa",
    "Asutuare", "Atebubu", "Bawku", "Bibiani", "Buipe", "Dambai", "Damongo",
    "Denu", "Drobo", "Duayaw Nkwanta", "Dunkwa-on-Offin", "Effiakuma", "Ejisu",
    "Foso", "Hohoe", "Kade", "Keta", "Kintampo", "Kpandu", "Kwesimintsim",
    
    # Towns and Villages
    "Aboso", "Aburi", "Adenta", "Agbogbloshie", "Agona", "Akim Oda", "Akwatia",
    "Anloga", "Apremdo", "Asafo", "Ashaiman", "Assin Foso", "Atebubu", "Atimpoku",
    "Awaso", "Berekum", "Bosome Freho", "Buduburam", "Busua", "Daboya", "Dodowa",
    "Dormaa Ahenkro", "Effiduase", "Fetteh", "Gbawe", "Kaneshie", "Kasoa", "Kibi",
    "Kpong", "Kumawu", "Lashibi", "Madina", "Mankessim", "Mpraeso", "Nalerigu",
    "Nkoranza", "Nungua", "Odumase", "Oduponkpehe", "Ofankor", "Oti", "Oyibi",
    "Pokuase", "Prestea", "Salaga", "Savelugu", "Sefwi Wiawso", "Shama", "Somanya",
    "Suhum", "Taifa", "Teshie", "Twifo Praso", "Wenchi", "Yendi", "Zuarungu",
    
    # Neighborhoods and Landmarks
    "Airport Hills", "Ashongman", "Atomic", "Awoshie", "Dansoman", "East Legon",
    "Haasto", "Labone", "Lapaz", "Legon", "North Ridge", "Osu", "Roman Ridge",
    "South Labone", "Spintex", "Tantra Hills", "Trassaco", "Tesano", "Achimota",
    "Adjiriganor", "Amasaman", "Ashale Botwe", "Bubuashie", "Cantonments",
    "Darkuman", "Dzorwulu", "Kokomlemle", "Lartebiokorshie", "Mallam", "Mataheko",
    "New Achimota", "Odorkor", "Oyarifa", "Pigfarm", "Sakaman", "Santa Maria",
    "Taifa", "Teshie-Nungua", "Tudu", "Weija",
    
    # Additional Locations
    "Aboadze", "Aboso", "Abutia", "Adidome", "Adjam", "Afienya", "Agbozume",
    "Akim Swedru", "Akwamufie", "Anomabu", "Asankragua", "Asawinso", "Asebu",
    "Asokore", "Assin Manso", "Aveyime", "Banda", "Bekwai", "Beposo", "Besease",
    "Bibiani", "Bimbilla", "Boso", "Breman", "Brewaniase", "Chinderi", "Daboase",
    "Dadieso", "Dagomba", "Dambai", "Domeabra", "Dzodze", "Ejisu", "Ekwamkrom",
    "Essam", "Feyiase", "Fumesua", "Gomoa", "Juaso", "Kadjebi", "Kete Krachi",
    "Kpandae", "Kpassa", "Kpeve", "Kpong", "Kumasi", "Kwahu", "Mampong", "Mankessim",
    "Mpraeso", "New Abirem", "Nkonya", "Nsuta", "Oda", "Old Tafo", "Prestea",
    "Sefwi Bekwai", "Suhum", "Tafo", "Takoradi", "Tarkwa", "Techiman", "Wassa"
}

# Expanded Ghanaian names (over 200 entries - mix of English and Ghanaian names)
GHANAIAN_NAMES = {
    # Traditional Male Names
    "Kwame", "Kofi", "Yaw", "Kwabena", "Kwaku", "Kojo", "Kwadwo", "Kwesi",
    "Kweku", "Kwamina", "Akwasi", "Ato", "Komla", "Koku", "Yao", "Ebo", "Efo",
    "Etse", "Kobina", "Kojovi", "Kpakpo", "Mawuli", "Nii", "Nkrumah", "Nortey",
    "Nyonkopa", "Ohene", "Oko", "Osei", "Otumfuo", "Owusu", "Paa", "Papa",
    "Prempeh", "Quarshie", "Sosu", "Tawia", "Tetteh", "Teye", "Torgbui",
    "Torgbuiga", "Torgbi", "Yaw", "Yoofi", "Adjetey", "Adotey", "Afari",
    "Agyeman", "Agyepong", "Agyei", "Akoto", "Amoako", "Ampofo", "Ankrah",
    "Annan", "Antwi", "Appiah", "Asamoah", "Asante", "Asare", "Atta", "Baah",
    "Baffoe", "Boateng", "Bonsu", "Danquah", "Darko", "Djan", "Donkor", "Fosu",
    "Frimpong", "Gyamfi", "Gyasi", "Kwarteng", "Mensah", "Nkansah", "Nkrumah",
    "Nsiah", "Nti", "Ntiamoah", "Ntim", "Ntow", "Nuertey", "Nyamekye", "Nyarko",
    "Obeng", "Ofori", "Opoku", "Oppong", "Osei", "Oti", "Owusu", "Poku", "Prempeh",
    "Quaye", "Quartey", "Safo", "Sarpong", "Sasu", "Sekyi", "Siaw", "Tagoe",
    "Tandoh", "Tawiah", "Tetteh", "Tuffour", "Tutu", "Twum", "Twumasi", "Wiafe",
    "Yamoah", "Yeboah", "Yirenkyi",
    
    # Traditional Female Names
    "Ama", "Akua", "Abena", "Adwoa", "Akosua", "Afua", "Aba", "Afi", "Araba",
    "Awo", "Dede", "Esi", "Efua", "Ekua", "Ekuwa", "Enyonam", "Esenam", "Fafa",
    "Fifi", "Kakra", "Kukua", "Maame", "Mansa", "Nana", "Nyaniba", "Nyankomago",
    "Panyin", "Serwaa", "Yaa", "Yawa", "Ye", "Yemisi", "Yoomley", "Dzidzor",
    "Efia", "Fosua", "Korkor", "Nana Ama", "Nhyira", "Adoma", "Adwoa", "Afia",
    "Agyapomaa", "Akos", "Akofa", "Akosua", "Ama", "Amma", "Araba", "Asantewaa",
    "Awura", "Dufie", "Efua", "Ekua", "Esi", "Fafa", "Kafui", "Maame", "Mame",
    "Mansa", "Naana", "Nana", "Nyamekye", "Obaa", "Ohenewaa", "Opokuwaa", "Pomaa",
    "Sika", "Tawia", "Yaa", "Yaa Asantewaa", "Yaa Nyarko", "Yawa", "Yeboaa",
    
    # Common English Names
    "John", "Michael", "David", "James", "Robert", "William", "Richard",
    "Joseph", "Thomas", "Daniel", "Matthew", "Andrew", "Edward", "Christopher",
    "George", "Paul", "Mark", "Steven", "Kenneth", "Anthony", "Charles", "Samuel",
    "Patrick", "Benjamin", "Peter", "Francis", "Stephen", "Raymond", "Nathaniel",
    "Timothy", "Elijah", "Gabriel", "Simon", "Philip", "Nicholas", "Frederick",
    "Mary", "Elizabeth", "Patricia", "Jennifer", "Linda", "Susan", "Margaret",
    "Dorothy", "Sarah", "Jessica", "Nancy", "Karen", "Lisa", "Betty", "Sandra",
    "Ashley", "Kimberly", "Emily", "Donna", "Michelle", "Carol", "Amanda",
    "Melissa", "Deborah", "Stephanie", "Rebecca", "Laura", "Sharon", "Cynthia",
    "Kathleen", "Helen", "Amy", "Angela", "Brenda", "Anna", "Pamela", "Nicole",
    "Ruth", "Katherine", "Christine", "Emma", "Catherine", "Debra", "Rachel",
    "Carolyn", "Janet", "Virginia", "Maria", "Heather", "Diane", "Julie",
    "Joyce", "Victoria", "Olivia", "Martha", "Lauren", "Judith", "Cheryl",
    "Megan", "Andrea", "Hannah", "Jacqueline", "Alice", "Teresa", "Sara",
    "Janice", "Doris", "Madeline", "Frances", "Gladys", "Evelyn", "Grace"
}

# Disaster definitions
DISASTER_DEFINITIONS = {
    "flood": "An overflow of water that submerges land that is usually dry.",
    "earthquake": "A sudden shaking of the ground caused by movements in Earth's crust.",
    "fire": "A rapid oxidation process that produces heat and light.",
    "tornado": "A violently rotating column of air extending from a thunderstorm to the ground.",
    "hurricane": "A powerful tropical storm with sustained winds exceeding 74 mph.",
    "landslide": "The movement of rock, earth, or debris down a slope.",
    "thunderstorm": "A storm with lightning and thunder, often with heavy rain.",
    "heatwave": "A prolonged period of excessively hot weather.",
    "tsunami": "A series of enormous ocean waves caused by underwater disturbances.",
    "volcanic eruption": "When lava and gas are discharged from a volcanic vent.",
    "disaster": "A sudden event that causes great damage or loss of life."
}

# Training data
training_data = [
    # Safety Guidelines 
    ("How do I stay safe during a flood?", "safety_guideline"),
    ("What should I do during an earthquake?", "safety_guideline"),
    ("How can I protect myself during a fire?", "safety_guideline"),
    ("What are the safety measures for a tornado?", "safety_guideline"),
    ("How do I prepare for a hurricane?", "safety_guideline"),
    ("What should I do if there's a landslide?", "safety_guideline"),
    ("How do I stay safe during a thunderstorm?", "safety_guideline"),
    ("What are the safety tips for a heatwave?", "safety_guideline"),
    ("How do I protect myself during a tsunami?", "safety_guideline"),
    ("What should I do if there's a volcanic eruption?", "safety_guideline"),

    # Incident Reports 
    ("There's a fire in my neighborhood!", "incident_report"),
    ("Heavy rains are causing floods!", "incident_report"),
    ("A building just collapsed in Accra!", "incident_report"),
    ("There's a gas leak in my area!", "incident_report"),
    ("I saw a landslide near Kumasi!", "incident_report"),
    ("There's a major accident on the highway!", "incident_report"),
    ("A tree fell on a house in Tamale!", "incident_report"),
    ("There's a flood in my community!", "incident_report"),
    ("I need to report a fire outbreak!", "incident_report"),
    ("There's a power outage in my area!", "incident_report"),

    # FAQs 
    ("How do I report an emergency?", "faq"),
    ("What is NADMO's emergency number?", "faq"),
    ("Where is the nearest NADMO office?", "faq"),
    ("How can I volunteer for NADMO?", "faq"),
    ("What services does NADMO provide?", "faq"),
    ("How do I get disaster preparedness training?", "faq"),
    ("What should I include in an emergency kit?", "faq"),
    ("How do I contact NADMO for assistance?", "faq"),
    ("What are the common disasters in Ghana?", "faq"),
    ("How do I apply for disaster relief?", "faq"),

    # Disaster Alerts 
    ("Send me disaster alerts.", "disaster_alert"),
    ("I want to receive emergency notifications.", "disaster_alert"),
    ("How do I sign up for disaster warnings?", "disaster_alert"),
    ("Can I get alerts for floods?", "disaster_alert"),
    ("Notify me about earthquakes.", "disaster_alert"),
    ("I need updates on fire outbreaks.", "disaster_alert"),
    ("How do I get weather warnings?", "disaster_alert"),
    ("Send me updates on landslides.", "disaster_alert"),
    ("I want to know about storms in my area.", "disaster_alert"),
    ("How do I get tsunami alerts?", "disaster_alert"),

    # General Queries
    ("What is NADMO?", "general_info"),
    ("Tell me about NADMO.", "general_info"),
    ("Who runs NADMO?", "general_info"),
    ("What does NADMO stand for?", "general_info"),
    ("How does NADMO help during disasters?", "general_info"),
    ("What is the role of NADMO?", "general_info"),
    ("Can NADMO help with medical emergencies?", "general_info"),
    ("Does NADMO provide financial aid?", "general_info"),
    ("How does NADMO coordinate with other agencies?", "general_info"),
    ("What are NADMO's core responsibilities?", "general_info"),

    # Out of Scope Examples
    ("what is banku", "out_of_scope"),
    ("how to cook jollof", "out_of_scope"),
    ("tell me about football", "out_of_scope"),
    ("who won the election", "out_of_scope"),
    ("what's the weather today", "out_of_scope"),
    ("do you know fufu", "out_of_scope"),
    ("tell me a joke", "out_of_scope"),
    ("what's your favorite food", "out_of_scope"),
    ("how old are you", "out_of_scope"),
    ("do you like music", "out_of_scope"),

    # Basic commands
    ("help", "help"),
    ("what can you do", "help"),
    ("hi", "greeting"),
    ("hello", "greeting"),
    
    # Additional incident report examples
    ("Emergency! Gas leak in Bantama!", "incident_report"),
    ("Reporting a building collapse in Tafo", "incident_report"),
    ("There's been an explosion at the market", "incident_report"),
    ("Chemical spill at industrial area", "incident_report"),
    ("Help! Flood waters rising fast!", "incident_report"),
    ("Urgent: Fire outbreak in my community", "incident_report"),
    ("I need to report a collapsed bridge", "incident_report"),
    ("Dangerous landslide on Aburi road", "incident_report"),
    ("Emergency situation at the school", "incident_report"),
    ("Report: Major accident on Accra-Kumasi highway", "incident_report")
]

# Enhanced model training with all data
texts, labels = zip(*training_data)
model = make_pipeline(
    CountVectorizer(ngram_range=(1, 2)), 
    MultinomialNB()
)
model.fit(texts, labels)

def init_db():
    """Initialize database with proper error handling"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
        conn = get_db_connection()
        try:
            # Check if table exists
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='reports'")
            table_exists = cursor.fetchone()
            
            if not table_exists:
                # Create new table with all columns
                conn.execute('''CREATE TABLE reports 
                              (id INTEGER PRIMARY KEY AUTOINCREMENT,
                               name TEXT NOT NULL, 
                               location TEXT NOT NULL, 
                               incident TEXT NOT NULL, 
                               date TEXT NOT NULL,
                               time TEXT NOT NULL,
                               sentiment TEXT)''')
                conn.commit()
                print("Created new database with all columns")
            else:
                # Check if sentiment column exists
                cursor.execute("PRAGMA table_info(reports)")
                columns = [column[1] for column in cursor.fetchall()]
                if 'sentiment' not in columns:
                    try:
                        conn.execute("ALTER TABLE reports ADD COLUMN sentiment TEXT")
                        conn.commit()
                        print("Added sentiment column to existing database")
                    except sqlite3.Error as e:
                        print(f"Error adding sentiment column: {e}")
            
            print("Database initialized successfully")
        finally:
            conn.close()
    except sqlite3.Error as e:
        print(f"Database initialization error: {e}")
        raise

def save_report(name, location, incident):
    """Save report with guaranteed timestamp values and sentiment analysis"""
    conn = None
    try:
        # Try to analyze sentiment, fallback to neutral if TextBlob fails
        sentiment = "neutral"
        try:
            analysis = TextBlob(incident)
            if analysis.sentiment.polarity > 0.1:
                sentiment = "positive"
            elif analysis.sentiment.polarity < -0.1:
                sentiment = "negative"
        except Exception as e:
            print(f"Sentiment analysis failed, using neutral: {e}")
        
        conn = get_db_connection()
        # Get current timestamp
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")
        
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO reports (name, location, incident, date, time, sentiment)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, location, incident, date, time, sentiment))
        
        conn.commit()
        print(f"Report saved successfully - ID: {cursor.lastrowid}")
        return True
    except sqlite3.Error as e:
        print(f"Failed to save report: {e}")
        return False
    finally:
        if conn:
            conn.close()

def fix_null_timestamps(conn=None):
    """Fix any existing records with null date/time values"""
    own_connection = False
    try:
        if conn is None:
            conn = get_db_connection()
            own_connection = True
            
        cursor = conn.cursor()
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")
        
        cursor.execute("SELECT id FROM reports WHERE date IS NULL OR time IS NULL")
        null_records = cursor.fetchall()
        
        if null_records:
            print(f"Found {len(null_records)} records with null timestamps, fixing...")
            for record in null_records:
                cursor.execute("UPDATE reports SET date = ?, time = ? WHERE id = ?",
                             (current_date, current_time, record[0]))
            conn.commit()
            print(f"Fixed {len(null_records)} records")
    except sqlite3.Error as e:
        print(f"Error fixing null timestamps: {e}")
    finally:
        if own_connection and conn:
            conn.close()

def save_report(name, location, incident):
    """Save report with guaranteed timestamp values and sentiment analysis"""
    conn = None
    try:
        conn = get_db_connection()
        # Get current timestamp
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")
        
        # Perform sentiment analysis
        analysis = TextBlob(incident)
        sentiment = "neutral"
        if analysis.sentiment.polarity > 0.1:
            sentiment = "positive"
        elif analysis.sentiment.polarity < -0.1:
            sentiment = "negative"
        
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO reports (name, location, incident, date, time, sentiment)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, location, incident, date, time, sentiment))
        
        conn.commit()
        print(f"Report saved successfully - ID: {cursor.lastrowid}")
        return True
    except sqlite3.Error as e:
        print(f"Failed to save report: {e}")
        return False
    finally:
        if conn:
            conn.close()

def send_email(subject, body, to_email="kwadwosylvester22@gmail.com"):
    try:
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
        return True
    except Exception as e:
        print(f"Email failed to send: {e}")
        return False

def extract_location(text):
    try:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "GPE" and ent.text in GHANA_LOCATIONS:
                return ent.text
        found = [loc for loc in GHANA_LOCATIONS if loc.lower() in text.lower()]
        return found[0] if found else "Unknown Location"
    except:
        return "Unknown Location"

def extract_name(text):
    try:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON" and ent.text in GHANAIAN_NAMES:
                return ent.text
        found = [name for name in GHANAIAN_NAMES if name.lower() in text.lower()]
        return found[0] if found else "Anonymous"
    except:
        return "Anonymous"

def classify_intent(text):
    try:
        # Lowercase the text for easier matching
        lower_text = text.lower()
        
        # Incident report triggers - expanded list
        incident_phrases = [
            "there's a", "there is a", "reporting a", "i need to report",
            "emergency in", "help there's", "urgent", "disaster in",
            "accident in", "leak in", "fire in", "flood in", "collapsed",
            "explosion", "gas leak", "building collapse", "landslide",
            "outbreak", "hazard", "danger", "evacuate", "a building just collapsed"
        ]
        
        # First check if this is clearly an incident report
        if any(phrase in lower_text for phrase in incident_phrases):
            return "incident_report"
            
        # Then check if it contains any disaster-related keywords
        disaster_keywords = [
            "flood", "fire", "earthquake", "emergency", "disaster", 
            "nadmo", "safety", "alert", "report", "accident",
            "tornado", "hurricane", "landslide", "thunderstorm",
            "heatwave", "tsunami", "volcanic", "explosion",
            "gas leak", "chemical spill", "building collapse"
        ]
        
        if not any(keyword in lower_text for keyword in disaster_keywords):
            return "out_of_scope"
            
        # If it might be relevant, use the model
        prediction = model.predict([text])
        return prediction[0] if prediction[0] in labels else "out_of_scope"
    except:
        return "out_of_scope"

# Initialize database
init_db()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/reports")
def view_reports():
    """Endpoint to view all reports for verification"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM reports ORDER BY date DESC, time DESC")
        reports = [dict(row) for row in cursor.fetchall()]
        return jsonify({"reports": reports})
    except sqlite3.Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            conn.close()

@app.route("/analytics")
def get_analytics():
    """Endpoint to get analytics data"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get total reports
        cursor.execute("SELECT COUNT(*) FROM reports")
        total_reports = cursor.fetchone()[0]
        
        # Get today's reports
        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute("SELECT COUNT(*) FROM reports WHERE date = ?", (today,))
        today_reports = cursor.fetchone()[0]
        
        # Get sentiment analysis
        cursor.execute("SELECT sentiment, COUNT(*) as count FROM reports GROUP BY sentiment")
        sentiment_counts = {row['sentiment']: row['count'] for row in cursor.fetchall()}
        total_with_sentiment = sum(sentiment_counts.values())
        positive_sentiment = round((sentiment_counts.get('positive', 0) / total_with_sentiment * 100)) if total_with_sentiment else 0
        
        # Get most common disaster type
        disaster_keywords = {
            'flood': ['flood', 'water', 'rain'],
            'fire': ['fire', 'burn', 'smoke'],
            'accident': ['accident', 'crash', 'collision'],
            'medical': ['medical', 'ill', 'sick', 'injury'],
            'earthquake': ['earthquake', 'shake', 'tremor'],
            'other': []
        }
        
        cursor.execute("SELECT incident FROM reports")
        incidents = [row[0] for row in cursor.fetchall()]
        
        disaster_counts = defaultdict(int)
        for incident in incidents:
            found = False
            lower_incident = incident.lower()
            for disaster, keywords in disaster_keywords.items():
                if any(keyword in lower_incident for keyword in keywords):
                    disaster_counts[disaster] += 1
                    found = True
                    break
            if not found:
                disaster_counts['other'] += 1
        
        most_common_disaster = max(disaster_counts.items(), key=lambda x: x[1])[0] if disaster_counts else None
        
        # Get incidents by type (simplified)
        incident_types = [{'type': k, 'count': v} for k, v in disaster_counts.items()]
        
        # Get top locations
        cursor.execute("SELECT location, COUNT(*) as count FROM reports GROUP BY location ORDER BY count DESC LIMIT 5")
        top_locations = [dict(row) for row in cursor.fetchall()]
        
        # Get incidents by date (last 30 days)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        cursor.execute("""
            SELECT date, COUNT(*) as count 
            FROM reports 
            WHERE date >= ?
            GROUP BY date 
            ORDER BY date
        """, (thirty_days_ago,))
        incidents_by_date = [dict(row) for row in cursor.fetchall()]
        
        return jsonify({
            'total_reports': total_reports,
            'today_reports': today_reports,
            'positive_sentiment': positive_sentiment,
            'most_common_disaster': most_common_disaster,
            'incident_types': incident_types,
            'top_locations': top_locations,
            'incidents_by_date': incidents_by_date
        })
        
    except Exception as e:
        print(f"Error generating analytics: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            conn.close()

@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"reply": "Please provide a valid message."})

        user_input = data.get("message", "").strip()
        if not user_input:
            return jsonify({"reply": "Your message cannot be empty."})

        normalized_input = user_input.lower()

        # Handle greetings
        if normalized_input in ["hi", "hello", "hey"]:
            return jsonify({"reply": "Hello! Welcome to NADMO Emergency Services. How can I assist you today?"})

        # Handle help requests
        if normalized_input in ["help", "what can you do"]:
            help_msg = """I can help with:
            - Providing definitions of disasters (e.g., "What is a flood?")
            - Giving safety guidelines for disasters
            - Processing emergency incident reports
            - Answering FAQs about NADMO services
            - Providing disaster alerts information
            What do you need help with specifically?"""
            return jsonify({"reply": help_msg})

        # Handle bot identity questions
        if any(q in normalized_input for q in ["who are you", "your name"]):
            return jsonify({"reply": "I'm the NADMO Emergency Response Chatbot, here to help with disaster management information and reporting."})

        # Handle disaster definition requests
        for disaster, definition in DISASTER_DEFINITIONS.items():
            if any(phrase in normalized_input for phrase in [f"what is {disaster}", f"define {disaster}", f"tell me about {disaster}"]):
                return jsonify({"reply": f"{disaster.capitalize()}: {definition}"})

        # Process the message
        intent = classify_intent(user_input)
        
        # Handle out-of-scope queries
        if intent == "out_of_scope":
            suggestions = """I'm sorry, I'm trained to help with NADMO and disaster-related queries only. 
            You can ask about:
            - Definitions of disasters (e.g., "What is a flood?")
            - Safety procedures for disasters
            - How to report emergencies
            - NADMO services and contacts"""
            return jsonify({"reply": suggestions})

        location = extract_location(user_input)
        name = extract_name(user_input)
        response_msg = ""
        
        if intent == "safety_guideline":
            if "flood" in normalized_input:
                response_msg = """Flood Safety Guidelines:
                1) Move to higher ground immediately
                2) Avoid walking or driving through floodwaters
                3) Stay informed through weather alerts
                4) Turn off utilities if instructed"""
                
            elif "earthquake" in normalized_input:
                response_msg = """Earthquake Safety Guidelines:
                1) Drop to the ground
                2) Take cover under sturdy furniture
                3) Hold on until shaking stops
                4) Stay indoors until shaking stops"""
                
            elif "fire" in normalized_input:
                response_msg = """Fire Safety Guidelines:
                1) Evacuate immediately if alarm sounds
                2) Use stairs, never elevators during a fire
                3) Stay low to avoid smoke inhalation
                4) Call emergency services once safe"""
                
            elif "tornado" in normalized_input:
                response_msg = """Tornado Safety Guidelines:
                1) Seek shelter in basement or storm cellar
                2) Stay away from windows and exterior walls
                3) Protect your head with sturdy objects
                4) Monitor weather alerts continuously"""
                
            elif "hurricane" in normalized_input:
                response_msg = """Hurricane Safety Guidelines:
                1) Secure your home by boarding windows
                2) Stock 3 days of supplies including water
                3) Know your community evacuation routes
                4) Follow all official instructions carefully"""
                
            elif "landslide" in normalized_input:
                response_msg = """Landslide Safety Guidelines:
                1) Move to higher ground immediately
                2) Avoid river valleys and drainage paths
                3) Stay alert after periods of heavy rain
                4) Listen for unusual sounds indicating moving debris"""
                
            elif "thunderstorm" in normalized_input:
                response_msg = """Thunderstorm Safety Guidelines:
                1) Stay indoors in a sturdy building
                2) Avoid using electrical equipment
                3) Stay away from windows and doors
                4) Don't use corded phones during the storm"""
                
            elif "heatwave" in normalized_input:
                response_msg = """Heatwave Safety Guidelines:
                1) Drink water regularly, don't wait until thirsty
                2) Limit outdoor activity during hottest hours
                3) Wear light, loose-fitting clothing
                4) Check on elderly neighbors and relatives"""
                
            elif "tsunami" in normalized_input:
                response_msg = """Tsunami Safety Guidelines:
                1) Move inland to higher ground immediately
                2) Go at least 2 miles inland or 100 feet above sea level
                3) Follow marked evacuation routes
                4) Don't return until authorities declare it safe"""
                
            elif "volcanic eruption" in normalized_input:
                response_msg = """Volcanic Eruption Safety Guidelines:
                1) Follow evacuation orders immediately
                2) Avoid river valleys and low-lying areas
                3) Protect yourself from falling ash
                4) Stay informed through official channels"""
                
            else:
                response_msg = "Please specify which disaster safety guidelines you need (e.g., 'flood safety guidelines')."

        elif intent == "faq":
            if "report an emergency" in normalized_input:
                response_msg = "Emergency reporting: Call NADMO at 0256 546 480 or provide details here."
            elif "emergency number" in normalized_input:
                response_msg = "NADMO emergency: 0256 546 480. National emergencies: 112 or 999."
            elif "nearest NADMO office" in normalized_input:
                response_msg = "NADMO offices: Visit nadmo.gov.gh or call 0256 546 480 for locations."
            elif "volunteer for NADMO" in normalized_input:
                response_msg = "Volunteering: Contact NADMO headquarters or visit their website."
            elif "services does NADMO provide" in normalized_input:
                response_msg = """NADMO provides these key services:
                
1. **Emergency Response**:
   - Disaster rescue operations
   - Emergency relief distribution
   - First responder coordination

2. **Disaster Prevention**:
   - Public education programs
   - Risk assessment surveys
   - Early warning systems

3. **Relief Services**:
   - Temporary shelter coordination
   - Emergency food/water distribution
   - Medical assistance coordination

4. **Recovery Programs**:
   - Reconstruction support
   - Livelihood restoration
   - Trauma counseling

5. **Capacity Building**:
   - Volunteer training programs
   - Community preparedness workshops
   - Disaster simulation exercises

For specific services, contact your regional NADMO office or call 0256 546 480"""
            elif "disaster preparedness training" in normalized_input:
                response_msg = """Disaster Preparedness Training Options:

1. **Basic Community Training**:
   - Duration: 2 days
   - Covers: First aid, evacuation procedures, emergency kits
   - Locations: All regional NADMO offices

2. **Advanced Volunteer Program**:
   - Duration: 2 weeks
   - Covers: Search/rescue, flood response, fire safety
   - Requirements: Minimum age 18, medical clearance

3. **School Safety Program**:
   - Earthquake/fire drills
   - Child-focused first aid
   - Available for all educational institutions

4. **Corporate Training**:
   - Workplace emergency plans
   - Business continuity strategies
   - Industrial accident prevention

To register: Visit nadmo.gov.gh/training or call 0256 546 480"""
            elif "emergency kit" in normalized_input:
                response_msg = """Essential Emergency Kit Contents:

✅ **Basic Supplies**:
   - 3-day water supply (4L per person per day)
   - Non-perishable food (canned goods, energy bars)
   - Manual can opener
   - First aid kit
   - Flashlight + extra batteries
   - Whistle

✅ **Documents**:
   - Copies of ID cards/passports
   - Insurance policies
   - Emergency contact list
   - Cash in small denominations

✅ **Additional Items**:
   - Medications (7-day supply)
   - Sanitation supplies
   - Multipurpose tool
   - Cell phone with charger
   - Emergency blanket

🔋 **For Families**:
   - Baby supplies if needed
   - Pet food if applicable
   - Entertainment for children

Store in waterproof container and check every 6 months."""
            elif "contact NADMO for assistance" in normalized_input:
                response_msg = """Multiple Ways to Contact NADMO:

📞 **Phone Contacts**:
- Emergency Hotline: 0302 937 992
- General Inquiries: 0256546480
- National Emergency: 112 or 999

📧 **Email**:
- info@nadmo.gov.gh (general inquiries)
- emergencies@nadmo.gov.gh (urgent matters)

🏢 **Office Locations**:
- Headquarters: NADMO HQ, Accra, Haatso
- Regional offices in all 16 regions
- District offices nationwide

🌐 **Online**:
- Website: nadmo.gov.gh
- Facebook: @NADMO.Ghana
- Twitter: @NADMO_Ghana

⏰ **Operating Hours**:
- Headquarters: Mon-Fri 8am-5pm
- Emergency lines: 24/7"""
            elif "common disasters in Ghana" in normalized_input:
                response_msg = """Here are the most common disasters in Ghana with comprehensive information:

🌊 **FLOODS** (Most Frequent):
- *Primary Causes*: Heavy rainfall, poor drainage systems, blocked waterways
- *High-Risk Areas*: Greater Accra (especially Accra Central), Kumasi, Northern regions
- *Seasonal Pattern*: Major peaks in May-June and September-October
- *Recent Major Events*: 2022 Accra floods, 2020 Kumasi floods
- *Prevention Measures*:
  • Regular drain clearance
  • Avoid building in flood plains
  • Proper waste disposal
  • Early warning system awareness

🔥 **FIRES** (Dry Season Hazard):
- *Common Types*:
  • Market fires (Kumasi Central Market, Kantamanto)
  • Domestic electrical fires
  • Bush fires (Northern regions)
- *Prevention Strategies*:
  • Proper electrical installations
  • Fire extinguishers in homes/businesses
  • No open flames in market areas
  • Regular safety inspections

⛰️ **LANDSLIDES** (Rainy Season Risk):
- *Vulnerable Areas*: Eastern Region hills (Aburi, Atwea), Volta Region
- *Main Causes*:
  • Deforestation
  • Unregulated construction on slopes
  • Heavy prolonged rainfall
- *Prevention*:
  • Proper land use planning
  • Terracing of slopes
  • Vegetation cover maintenance

⚡ **ELECTRICAL ACCIDENTS**:
- *Common Causes*:
  • Illegal connections
  • Overloaded circuits
  • Aging infrastructure
- *Prevention*:
  • Use only certified electricians
  • Avoid illegal connections
  • Regular maintenance checks

🏗️ **BUILDING COLLAPSES**:
- *Primary Causes*:
  • Substandard materials
  • Poor construction practices
  • Lack of proper permits
- *Prevention*:
  • Use certified professionals
  • Follow building codes
  • Proper inspections

Other Significant Risks:
- *Earthquakes*: Accra in seismic zone (last major quake 1939)
- *Droughts*: Northern regions vulnerable
- *Industrial Accidents*: Chemical spills in industrial areas

🚨 **Emergency Preparedness**:
• Maintain emergency kits
• Know evacuation routes
• Stay informed through official channels

📞 **Emergency Contacts**:
NADMO: 0302 937 992 / 0256546480
National Emergency: 112 or 999
🌐 Website: nadmo.gov.gh
📍 Regional offices nationwide for assistance"""
            elif "apply for disaster relief" in normalized_input:
                response_msg = """Disaster Relief Application Process:

1. **Initial Reporting**:
   - Report incident to local NADMO office immediately
   - Obtain incident report reference number

2. **Documentation Required**:
   - Proof of residency (national ID, utility bill)
   - Photos/videos of damage
   - Police report (if criminal activity involved)
   - Medical reports (for injury-related claims)

3. **Assessment Phase**:
   - NADMO officers will visit for damage assessment
   - Verification process takes 3-5 working days

4. **Relief Distribution**:
   - Based on assessment severity
   - May include: food supplies, building materials, financial aid

5. **Follow-up Support**:
   - Trauma counseling if needed
   - Reconstruction monitoring

⏳ Processing Time: 7-14 days after assessment
📞 Contact: 0302 937 992 for application status"""
            else:
                response_msg = "For more information, contact NADMO at 0256 546 480 or visit nadmo.gov.gh"

        elif intent == "incident_report":
            # Extract location and name more robustly
            location = extract_location(user_input)
            if location == "Unknown Location":
                location = "your area"  # More natural phrasing
            
            name = extract_name(user_input)
            if name == "Anonymous":
                name = "a concerned citizen"
            
            # Save the report with verification
            if save_report(name, location, user_input):
                response_msg = (
                    f"🚨 Emergency report received for {location} by {name}. "
                    "NADMO authorities have been notified and will respond. "
                    "For immediate assistance, call NADMO at 0256 546 480 or "
                    "the national emergency number 112/999."
                )
                
                # Send email with more details
                email_body = (
                    f"New Emergency Report:\n"
                    f"Incident Details: {user_input}\n"
                    f"Location: {location}\n"
                    f"Reported by: {name}\n"
                    f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                send_email(
                    "🚨 New Emergency Report - Immediate Attention Needed", 
                    email_body,
                    "kwadwosylvester22@gmail.com"
                )
            else:
                response_msg = ("⚠️ We encountered an issue saving your report. "
                              "Please call NADMO directly at 0256 546 480 for immediate assistance.")

        elif intent == "disaster_alert":
            response_msg = """Disaster Alert Subscription Options:

1. **SMS Alerts**:
   - Text 'ALERTS' to 1921
   - Receive instant notifications
   - Standard network rates apply

2. **Mobile App**:
   - Download 'NADMO Alerts' from app stores
   - Customize alert types
   - Location-based warnings

3. **Email Subscriptions**:
   - Register at nadmo.gov.gh/alerts
   - Get detailed bulletins
   - Choose frequency (instant/daily digest)

4. **Community Sirens**:
   - Available in high-risk areas
   - Tested first Wednesday of each month
   - Learn your local warning signals

5. **Radio/TV Broadcasts**:
   - Tune to official stations during emergencies
   - Regular updates on GBC, JoyFM, etc.

🔔 Alert Types Available:
- Flood warnings
- Fire outbreaks
- Earthquake alerts
- Severe weather
- Industrial hazards"""

        elif intent == "general_info":
            if "what is nadmo" in normalized_input:
                response_msg = """NADMO (National Disaster Management Organization):

📌 **Mandate**: Ghana's lead agency for comprehensive disaster management and emergency response

🔹 **Core Functions**:
   - Disaster prevention and mitigation
   - Emergency response coordination
   - Public education and awareness
   - Relief and recovery operations

📅 **Established**: 1996 under Act 517 of Parliament

🌍 **Operations**: Nationwide through regional and district offices

🧑‍🤝‍🧑 **Staff**: Combination of professionals and trained volunteers

💼 **Oversight**: Ministry of Interior

🌐 Learn more: nadmo.gov.gh"""
            elif "who runs nadmo" in normalized_input:
                response_msg = """NADMO Leadership Structure:

1. **Director General**:
   - Appointed by the President
   - Overall responsibility for operations
   - Current: Mr. Eric Nana Agyeman-Prempeh

2. **Deputy Directors**:
   - Technical, Operations, and Administration
   - Oversee specific divisions

3. **Regional Coordinators**:
   - One for each of the 16 regions
   - Implement national policies locally

4. **District Officers**:
   - Cover all 260+ districts
   - First responders at local level

5. **Governing Board**:
   - Multi-sectoral representation
   - Meets quarterly to set policies

Contact leadership through NADMO HQ: 0256 546 480"""
            elif "what does nadmo stand for" in normalized_input:
                response_msg = "NADMO: National Disaster Management Organization (Ghana's official disaster management agency)"
            elif "how does nadmo help" in normalized_input:
                response_msg = """NADMO Assistance Programs:

1. **Emergency Response**:
   - 24/7 emergency hotline (0302 937 992)
   - Search and rescue teams
   - First responder deployment

2. **Relief Services**:
   - Temporary shelter coordination
   - Emergency food/water distribution
   - Medical support coordination

3. **Prevention Programs**:
   - Community education workshops
   - Risk mapping and assessment
   - Early warning systems

4. **Recovery Support**:
   - Reconstruction assistance
   - Livelihood restoration
   - Trauma counseling

5. **Capacity Building**:
   - Volunteer training programs
   - Equipment provisioning
   - Disaster simulation exercises

Visit your nearest NADMO office for specific assistance."""
            elif "role of nadmo" in normalized_input:
                response_msg = """NADMO's Key Roles:

1. **Coordination**:
   - Lead agency for disaster management
   - Inter-agency collaboration
   - International disaster partnerships

2. **Preparedness**:
   - Develop national emergency plans
   - Conduct risk assessments
   - Maintain emergency stocks

3. **Response**:
   - Deploy emergency teams
   - Coordinate relief efforts
   - Manage evacuation centers

4. **Recovery**:
   - Damage assessment
   - Reconstruction oversight
   - Victim support services

5. **Education**:
   - Public awareness campaigns
   - School safety programs
   - Community training

6. **Policy**:
   - Develop disaster management frameworks
   - Advocate for risk reduction
   - Implement international protocols"""
            elif "medical emergencies" in normalized_input:
                response_msg = """Medical Emergency Support:

🚑 **NADMO Medical Services**:

1. **Emergency Response**:
   - First aid at disaster scenes
   - Triage coordination
   - Patient transport

2. **Partnerships**:
   - Works with Ghana Health Service
   - Coordinates with ambulance services
   - Links with hospital emergency units

3. **Specialized Teams**:
   - Trauma response units
   - Mass casualty incident teams
   - Disease outbreak responders

4. **Public Health**:
   - Epidemic prevention
   - Water purification
   - Sanitation monitoring

📞 For medical emergencies:
- NADMO: 0302 937 992
- National Ambulance: 112 or 193"""
            elif "financial aid" in normalized_input:
                response_msg = """Financial Assistance Options:

1. **Disaster Relief Grants**:
   - For verified disaster victims
   - Covers basic needs
   - Application through local NADMO office

2. **Reconstruction Loans**:
   - Low-interest options
   - For home/business repairs
   - Partner financial institutions

3. **Livelihood Support**:
   - Small business recovery grants
   - Agricultural rehabilitation
   - Skills training programs

4. **Special Funds**:
   - Presidential relief initiatives
   - International donor programs
   - Corporate sponsorship funds

📋 Requirements:
- Proof of loss/damage
- Valid ID documents
- Bank account details
- NADMO assessment report

ℹ️ Inquire at your regional NADMO office"""
            elif "coordinate with other agencies" in normalized_input:
                response_msg = """NADMO's Coordination Network:

1. **Government Partners**:
   - Ghana Police Service
   - Ghana Fire Service
   - Ghana Armed Forces
   - Ghana Health Service

2. **International Agencies**:
   - UN Office for Disaster Risk Reduction
   - World Food Programme
   - Red Cross/Red Crescent

3. **NGO Collaborations**:
   - Adventist Development and Relief Agency
   - Plan International Ghana
   - World Vision Ghana

4. **Technical Partners**:
   - Ghana Meteorological Agency
   - Environmental Protection Agency
   - National Development Planning Commission

5. **Community Groups**:
   - Traditional authorities
   - Religious organizations
   - Neighborhood associations"""
            elif "core responsibilities" in normalized_input:
                response_msg = """NADMO's Core Responsibilities:

1. **Risk Assessment**:
   - Identify disaster-prone areas
   - Maintain national risk register
   - Conduct vulnerability studies

2. **Prevention Planning**:
   - Develop mitigation strategies
   - Establish early warning systems
   - Enforce building codes in hazard zones

3. **Emergency Response**:
   - Coordinate rescue operations
   - Manage evacuation procedures
   - Deploy relief supplies

4. **Public Education**:
   - Conduct safety campaigns
   - Train community volunteers
   - School safety programs

5. **Recovery Management**:
   - Damage assessment
   - Reconstruction oversight
   - Victim support services

6. **Policy Development**:
   - Formulate disaster management policies
   - Implement international frameworks
   - Advocate for risk reduction"""
            elif "nadmo headquarters" in normalized_input:
                response_msg = """NADMO Headquarters Information:

📍 **Physical Address**:
NADMO Headquarters
Haatso, Accra
(Ghana Atomic Energy Road)

📌 **Mailing Address**:
P.O. Box 78
Accra, Ghana

📞 **Contact Numbers**:
- Main: 0302 937 992
- Alternate: 0256546480
- Fax: 0302 937 993

🕒 **Operating Hours**:
Monday-Friday: 8:00am - 5:00pm
Emergency lines: 24/7

🌐 **Online**:
Website: nadmo.gov.gh
Email: info@nadmo.gov.gh

🚗 **Access**:
- Accessible by public transport (trotros to Haatso)
- Parking available for visitors"""
            else:
                response_msg = "NADMO is Ghana's national agency for comprehensive disaster management and emergency response."

        else:
            response_msg = "I'm not sure how to help with that. Could you ask about disaster safety, reporting, or NADMO services?"

        return jsonify({"reply": response_msg})

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"reply": "Sorry, I'm having trouble processing your request. Please try again later."})

if __name__ == "__main__":
    # Initialize database before starting the app
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=False)