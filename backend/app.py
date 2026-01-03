from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import torch
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)

# ===============================
# Database Configuration
# ===============================
app.secret_key = "supersecretkey"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///first_aid_chats.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ===============================
# Database Models
# ===============================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(120))
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    messages = db.relationship('Message', backref='chat', lazy=True, cascade="all, delete")

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey('chat.id'), nullable=False)
    sender = db.Column(db.String(10))
    message = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ===============================
# Load Model and Data
# ===============================
MODEL_PATH = "./first_aid_model"
DATA_PATH = "./cleaned_first_aid.jsonl"

print("🔄 Loading model...")
model = SentenceTransformer(MODEL_PATH)
print("✅ Model loaded successfully!")

print("🔄 Loading first aid data...")
data = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)
            if "prompt" in obj and "completion" in obj:
                data.append({
                    "symptom": obj["prompt"].replace("Symptom:", "").strip(),
                    "first_aid": obj["completion"].replace("First Aid:", "").strip()
                })
        except json.JSONDecodeError:
            continue

print(f"✅ Loaded {len(data)} Q&A pairs")

print("🔄 Encoding symptom embeddings...")
symptom_texts = [item["symptom"] for item in data]
symptom_embeddings = model.encode(symptom_texts, convert_to_tensor=True, show_progress_bar=True)
print("✅ Embeddings ready!")

# ===============================
# Routes
# ===============================
@app.route("/")
def home():
    return render_template("index.html", body_class="")

@app.route("/login")
def login_page():
    return render_template("login.html", body_class="login-page")

@app.route("/signup")
def signup_page():
    return render_template("signup.html", body_class="login-page")

@app.route("/chat")
def chat_page():
    return render_template("index.html", body_class="")

@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    if User.query.filter_by(email=email).first():
        return jsonify({"success": False, "message": "User already exists"})
    hashed_pw = generate_password_hash(password)
    user = User(email=email, password=hashed_pw)
    db.session.add(user)
    db.session.commit()
    session["user_id"] = user.id
    session["email"] = user.email
    return jsonify({"success": True})

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password, password):
        return jsonify({"success": False, "message": "Invalid credentials"})
    session["user_id"] = user.id
    session["email"] = user.email
    return jsonify({"success": True})

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("home"))

@app.route("/me")
def me():
    if "user_id" in session:
        return jsonify({"logged_in": True, "email": session.get("email")})
    else:
        return jsonify({"logged_in": False})

@app.route("/chats")
def chats():
    user_id = session.get("user_id")
    if user_id:
        chats = Chat.query.filter_by(user_id=user_id).all()
    else:
        return jsonify({"chats": []})
    data = [{"id": c.id, "title": c.title or f"Chat {c.id}"} for c in chats]
    return jsonify({"chats": data})

@app.route("/new_chat", methods=["POST"])
def new_chat():
    user_id = session.get("user_id")
    chat = Chat(title="New Chat", user_id=user_id)
    db.session.add(chat)
    db.session.commit()
    return jsonify({"chat_id": chat.id})

@app.route("/messages/<int:chat_id>", methods=["GET"])
def get_messages(chat_id):
    messages = Message.query.filter_by(chat_id=chat_id).order_by(Message.timestamp).all()
    data = [{"sender": m.sender, "message": m.message} for m in messages]
    return jsonify({"messages": data})

@app.route("/chat/<int:chat_id>", methods=["POST"])
def chat(chat_id):
    user_input = request.json.get("message", "").strip()

    user_msg = Message(chat_id=chat_id, sender="user", message=user_input)
    db.session.add(user_msg)
    db.session.commit()

    # Small talk
    lower_input = user_input.lower()
    greetings = ["hi", "hello", "hey"]
    goodbyes = ["bye", "goodbye", "see you"]
    thanks = ["thank you", "thanks"]

    if any(word in lower_input for word in greetings):
        ai_reply = "Hello! 👋 How can I help you with first aid today?"
    elif any(word in lower_input for word in goodbyes):
        ai_reply = "Goodbye! 👋 Stay safe and take care."
    elif any(word in lower_input for word in thanks):
        ai_reply = "You're very welcome! 😊 Always here to help."
    else:
        query_embedding = model.encode(user_input, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, symptom_embeddings)[0]
        best_match_idx = torch.argmax(similarities).item()
        ai_reply = data[best_match_idx]["first_aid"]

    ai_msg = Message(chat_id=chat_id, sender="ai", message=ai_reply)
    db.session.add(ai_msg)
    db.session.commit()

    chat = Chat.query.get(chat_id)
    if chat.title == "New Chat":
        chat.title = user_input[:40] + "..."
        db.session.commit()

    return jsonify({"reply": ai_reply})

# ===============================
# Run Flask App
# ===============================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=True)