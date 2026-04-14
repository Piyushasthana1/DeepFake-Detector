import os
import uuid
import sqlite3
import time

from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from gradio_client import Client


# ---------------- CONFIG ---------------- #

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret123'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024


# ---------------- LOGIN ---------------- #

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


# ---------------- DATABASE ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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


# ---------------- HF CALL (FINAL FIXED) ---------------- #

def predict_video_api(filepath):

    for attempt in range(3):  # 🔁 retry for HF cold start
        try:
            print(f"🚀 Attempt {attempt+1}: Connecting to HF...")

            client = Client("shivanshuasthana81/deepfake-detector")

            result = client.predict(
                filepath,
                api_name="/predict"
            )

            print("🔍 RAW RESULT:", result)

            # ✅ SAFE PARSING
            if isinstance(result, (list, tuple)):

                # case: ['FAKE', 85.34]
                if len(result) == 2 and isinstance(result[0], str):
                    label = result[0]
                    confidence = float(result[1])

                # case: [['FAKE', 85.34]]
                elif len(result) == 1 and isinstance(result[0], (list, tuple)):
                    label = result[0][0]
                    confidence = float(result[0][1])

                else:
                    label = "ERROR"
                    confidence = 0

            else:
                label = "ERROR"
                confidence = 0

            return label, round(confidence, 2)

        except Exception as e:
            print(f"❌ Attempt {attempt+1} failed:", e)
            time.sleep(3)  # wait before retry

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
        except:
            flash("Username already exists")

        conn.close()
        return redirect(url_for('login'))

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
            flash("Invalid format")
            return redirect(url_for('dashboard'))

        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)

        file.save(filepath)

        print("📤 Uploaded:", filepath)

        label, confidence = predict_video_api(filepath)

        # cleanup
        if os.path.exists(filepath):
            os.remove(filepath)

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
