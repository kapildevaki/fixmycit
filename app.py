from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
import os, sqlite3, time
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ------------------ Flask Setup ------------------
app = Flask(__name__)
app.secret_key = "fixmycity_secret"

UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DB_PATH = os.path.join(app.root_path, "reports.db")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ------------------ Load ML Model ------------------
MODEL_PATH = os.path.join(app.root_path, "model.h5")
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    categories = ["Pothole", "Garbage", "Streetlight", "Waterlogging", "Other"]  # Replace with your labels
else:
    model = None
    categories = ["Unknown"]

# ------------------ Database Setup ------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            aadhaar TEXT,
            user_photo TEXT,
            status TEXT DEFAULT 'Pending',
            office_photo TEXT,
            category TEXT,
            latitude REAL,
            longitude REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ------------------ ML Prediction ------------------
def predict_category(image_path):
    if not model:
        return "Unknown"
    # Load as grayscale, resize to 240x240 (flattened input)
    img = image.load_img(image_path, target_size=(240, 240), color_mode="grayscale")
    x = image.img_to_array(img)        # Shape: (240, 240, 1)
    x = x / 255.0                      # Normalize
    x = x.reshape(1, -1)               # Flatten: (1, 57600)
    preds = model.predict(x)
    index = np.argmax(preds)
    return categories[index]

# ------------------ Routes ------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/user_login", methods=["GET","POST"])
def user_login():
    if request.method=="POST":
        aadhaar = request.form.get("aadhaar")
        if not aadhaar:
            return "❌ Aadhaar required"
        session["aadhaar"] = aadhaar
        return redirect("/user")
    return render_template("user_login.html")

@app.route("/user", methods=["GET","POST"])
def user():
    if "aadhaar" not in session:
        return redirect("/user_login")
    aadhaar = session["aadhaar"]

    if request.method=="POST":
        file = request.files.get("photo")
        lat = request.form.get("latitude") or 0
        lon = request.form.get("longitude") or 0

        if not file:
            return "❌ Please select a photo."

        # Save uploaded image
        filename = f"{int(time.time())}_{secure_filename(file.filename)}"
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)

        # ML prediction
        category = predict_category(path)

        # Insert into DB
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO reports (aadhaar, user_photo, latitude, longitude, category) VALUES (?,?,?,?,?)",
                (aadhaar, filename, lat, lon, category)
            )
            conn.commit()
        return redirect("/user")

    # Fetch user reports
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM reports WHERE aadhaar=? ORDER BY id DESC", (aadhaar,))
        reports = cur.fetchall()
    return render_template("user.html", reports=reports)

@app.route("/office", methods=["GET","POST"])
def office():
    if request.method=="POST":
        rid = request.form["report_id"]
        status = request.form["status"]
        file = request.files.get("office_photo")
        filename = None
        if file:
            filename = f"{int(time.time())}_{secure_filename(file.filename)}"
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

        with get_db_connection() as conn:
            cur = conn.cursor()
            if filename:
                cur.execute("UPDATE reports SET status=?, office_photo=? WHERE id=?", (status, filename, rid))
            else:
                cur.execute("UPDATE reports SET status=? WHERE id=?", (status, rid))
            conn.commit()
        return redirect("/office")

    # Fetch all reports
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM reports ORDER BY id DESC")
        reports = cur.fetchall()
    return render_template("office.html", reports=reports)

@app.route("/logout")
def logout():
    session.pop("aadhaar", None)
    return redirect("/")

@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ------------------ Run ------------------
if __name__=="__main__":
    app.run(debug=True, port=5000)
