import sys
import os
import time
import uuid
import sqlite3
import gc

from werkzeug.utils import secure_filename

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

import torch
import cv2
import torchvision.transforms as T

from src.utils.face_detect import crop_faces_from_frame
from src.model import CNNFeatureExtractor, CNN_LSTM_Attention


# ---------------- FILE VALIDATION ---------------- #

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------- APP SETUP ---------------- #

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret123'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 🔥 reduce to 20MB

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


# ---------------- SQLITE DATABASE ---------------- #

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')

    conn.commit()
    conn.close()

init_db()


# ---------------- USER ---------------- #

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password


@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    c.execute("SELECT * FROM users WHERE id=?", (user_id,))
    user = c.fetchone()

    conn.close()

    if user:
        return User(user[0], user[1], user[2])
    return None


# ---------------- MODEL ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pth")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = 'cpu'

# 🔥 smaller transform
transform = T.Compose([
    T.Resize((112, 112)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

feat_extractor = CNNFeatureExtractor().to(device)
model = CNN_LSTM_Attention(feat_dim=feat_extractor.out_dim).to(device)

checkpoint = torch.load(MODEL_PATH, map_location='cpu')
feat_extractor.load_state_dict(checkpoint['feat_state'])
model.load_state_dict(checkpoint['model_state'])

feat_extractor.eval()
model.eval()

print("✅ Model loaded")


# ---------------- ULTRA-LIGHT PREDICTION ---------------- #
def predict_video(video_path):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < 10:
        cap.release()
        return "VIDEO TOO SHORT", 0

    # 🔥 sample 8 frames evenly
    num_frames_to_pick = 8
    frame_indices = [int(i * total_frames / num_frames_to_pick) for i in range(num_frames_to_pick)]

    frames = []
    current_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if current_idx in frame_indices:
            faces = crop_faces_from_frame(frame)

            if len(faces) > 0:
                face = faces[0]

                # 🔥 resize
                face = cv2.resize(face, (112, 112))

                # 🔥 VERY IMPORTANT FIX (BGR → RGB)
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                frames.append(face)

        current_idx += 1

        if len(frames) >= 8:
            break

    cap.release()

    if len(frames) < 4:
        return "NO FACE DETECTED", 0

    try:
        xs = torch.stack([transform(f) for f in frames]).unsqueeze(0)

        with torch.no_grad():
            B, S, C, H, W = xs.shape

            seqs = xs.view(B * S, C, H, W)

            feats = feat_extractor(seqs)
            feats = feats.view(B, S, -1)

            logits, _ = model(feats)

            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        print("DEBUG PROBS:", probs)  # 🔥 debug

        label = "FAKE" if probs[1] > probs[0] else "REAL"
        confidence = float(max(probs) * 100)

    except Exception as e:
        print("Prediction Error:", e)
        return "ERROR", 0

    finally:
        # 🔥 safe cleanup
        del xs, feats, logits
        gc.collect()

    return label, round(confidence, 2)

# ---------------- ROUTES ---------------- #

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        try:
            c.execute("INSERT INTO users VALUES (?, ?, ?)",
                      (username, username, hashed_password))
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

        conn = sqlite3.connect('users.db')
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
            flash("No file")
            return redirect(url_for('dashboard'))

        file = request.files['video']

        if file.filename == '':
            flash("No file selected")
            return redirect(url_for('dashboard'))

        if not allowed_file(file.filename):
            flash("Invalid file")
            return redirect(url_for('dashboard'))

        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(filepath)

        label, confidence = predict_video(filepath)

        # 🔥 delete immediately
        if os.path.exists(filepath):
            os.remove(filepath)

        return render_template('result.html',
                               label=label,
                               confidence=confidence)

    return render_template('dashboard.html', username=current_user.username)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/health')
def health():
    return "OK", 200


@app.errorhandler(413)
def too_large(e):
    return "File too large (Max 20MB)", 413


if __name__ == '__main__':
    app.run(debug=True)
