import os
import sys

# ✅ FORCE ADD PROJECT ROOT TO PATH (RENDER SAFE)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)


from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

import torch
import cv2
import torchvision.transforms as T

# ✅ NOW THIS WILL WORK
from src.utils.face_detect import crop_faces_from_frame
from src.model import CNNFeatureExtractor, CNN_LSTM_Attention


# ---------------- PATH SETUP ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pth")


# ---------------- APP SETUP ---------------- #

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret123'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


# ---------------- DATABASE ---------------- #

import sqlite3

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

device = 'cpu'

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

feat_extractor = CNNFeatureExtractor().to(device)
model = CNN_LSTM_Attention(feat_dim=feat_extractor.out_dim).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)

feat_extractor.load_state_dict(checkpoint['feat_state'])
model.load_state_dict(checkpoint['model_state'])

feat_extractor.eval()
model.eval()

print("✅ Model loaded correctly on Render")


# ---------------- PREDICTION ---------------- #

def predict_video(video_path):

    cap = cv2.VideoCapture(video_path)
    frames = []

    ok, frame = cap.read()
    idx = 0

    while ok and idx < 16:
        try:
            faces = crop_faces_from_frame(frame)
            if len(faces) > 0:
                frames.append(faces[0])
        except Exception as e:
            print("Face error:", e)

        ok, frame = cap.read()
        idx += 1

    cap.release()

    print("Frames:", len(frames))

    if len(frames) < 2:
        return "NO FACE DETECTED", 0

    xs = torch.stack([transform(f) for f in frames]).unsqueeze(0).to(device)

    with torch.no_grad():
        B,S,C,H,W = xs.shape
        seqs = xs.view(B*S, C, H, W)

        feats = feat_extractor(seqs)
        feats = feats.view(B, S, -1)

        logits, _ = model(feats)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    print("PROBS:", probs)

    label = "FAKE" if probs[1] > probs[0] else "REAL"
    confidence = max(probs) * 100

    return label, round(confidence, 2)


# ---------------- ROUTES ---------------- #

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/dashboard', methods=['GET','POST'])
@login_required
def dashboard():

    if request.method == 'POST':

        file = request.files['video']

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        label, confidence = predict_video(filepath)

        os.remove(filepath)

        return render_template('result.html', label=label, confidence=confidence)

    return render_template('dashboard.html', username=current_user.username)


@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        return redirect(url_for('dashboard'))
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# ---------------- RUN ---------------- #

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
