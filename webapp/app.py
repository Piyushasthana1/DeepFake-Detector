import os
import uuid
import sqlite3
import requests

from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from gradio_client import Client, handle_file

# ---------------- CONFIG ---------------- #

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

app.config['SECRET_KEY'] = 'secret123'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB Limit

# ---------------- LOGIN ---------------- #

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ---------------- DATABASE ---------------- #

DB_PATH = os.path.join(BASE_DIR, "users.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id=?", (user_id,))
    user = c.fetchone()
    conn.close()
    if user:
        return User(user[0], user[1], user[2])
    return None

# ---------------- FILE STORAGE ---------------- #

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------- HF GRADIO CLIENT ---------------- #

# Using the Space name is more stable than the direct URL
HF_SPACE_ID = "shivanshuasthana81/deepfake-detector"

try:
    hf_client = Client(HF_SPACE_ID)
    print(f"✅ Connected to Hugging Face Space: {HF_SPACE_ID}")
except Exception as e:
    print(f"⚠️ Warning: Could not connect to HF Space on startup: {e}")

def predict_video_api(filepath):
    try:
        print(f"🚀 Sending {filepath} to Hugging Face...")
        
        # Use handle_file to let Gradio manage the upload process
        # api_name="/predict" matches your gr.Interface fn name
        result = hf_client.predict(
            video=handle_file(filepath),
            api_name="/predict"
        )

        print("🔍 API RESULT:", result)

        # Result is the dictionary returned by your HF app.py
        label = result.get("label", "ERROR")
        confidence = result.get("confidence", 0)

        return label, round(float(confidence), 2)

    except Exception as e:
        print("❌ HF CLIENT ERROR:", e)
        return "ERROR", 0

# ---------------- ROUTES ---------------- #

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users VALUES (?, ?, ?)",
                      (str(uuid.uuid4()), username, password))
            conn.commit()
            flash("Registered Successfully")
            conn.close()
            return redirect(url_for('login'))
        except Exception as e:
            flash("Username already exists or database error")
            conn.close()

    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            login_user(User(user[0], user[1], user[2]))
            return redirect(url_for('dashboard'))

        flash("Invalid credentials")

    return render_template('login.html')

@app.route('/dashboard', methods=['GET','POST'])
@login_required
def dashboard():
    if request.method == 'POST':
        if 'video' not in request.files:
            flash("No file uploaded")
            return redirect(url_for('dashboard'))

        file = request.files['video']

        if file.filename == '':
            flash("No file selected")
            return redirect(url_for('dashboard'))

        if not allowed_file(file.filename):
            flash("Invalid format. Use mp4, avi, or mov.")
            return redirect(url_for('dashboard'))

        # Secure and save the file
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(filepath)

        # Call the prediction
        label, confidence = predict_video_api(filepath)

        # Clean up uploaded file to save space on Render
        if os.path.exists(filepath):
            os.remove(filepath)

        if label == "ERROR":
            flash("Model is currently sleeping or failed. Please try again in a moment.")
            return redirect(url_for('dashboard'))

        return render_template(
            'result.html',
            label=label,
            confidence=confidence
        )

    return render_template('dashboard.html', username=current_user.username)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/health')
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
