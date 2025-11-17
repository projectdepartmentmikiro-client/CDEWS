#----------------- GCE ------------------

import os
import cv2
import requests
import sqlite3
import pandas as pd
import pathlib
import random
import string
from flask import Flask, request, jsonify, send_from_directory
from roboflow import Roboflow
from datetime import datetime
from werkzeug.utils import secure_filename
from google.cloud import storage

DEVICE_CODE = "IOLT-00001"
API_TOKEN = "0c098ff7-2c19-4c36-ac8f-edc2aa3fd203"
API_URL = "https://c-dews.synqbox.com/api/v1/device-data"

BASE_DIR = "/home/server"
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DB_DIR = os.path.join(BASE_DIR, "database")

for path in [UPLOAD_DIR, RESULTS_DIR, DB_DIR]:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

DB_PATH = os.path.join(DB_DIR, "egg_results.db")
CSV_PATH = os.path.join(DB_DIR, "egg_results.csv")

SERVICE_KEY_JSON = "service_key.json"
BUCKET_NAME = "recieved_image-bucket"

storage_client = storage.Client.from_service_account_json(SERVICE_KEY_JSON)
bucket = storage_client.bucket(BUCKET_NAME)

def upload_to_bucket(local_path, folder="uploads"):
    blob = bucket.blob(f"{folder}/{os.path.basename(local_path)}")
    blob.upload_from_filename(local_path)
    return blob.name

def get_signed_url(blob_name, expiration=3600):
    blob = bucket.blob(blob_name)
    return blob.generate_signed_url(version="v4", expiration=expiration, method="GET")

def generate_key(length=16):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

API_KEY = generate_key()
API_SECRET = generate_key()

def check_api(req):
    key = req.headers.get("x-api-key")
    secret = req.headers.get("x-api-secret")
    return key == API_KEY and secret == API_SECRET

app = Flask(__name__)

rf = Roboflow(api_key="wKgfqYQvBCHPuxftcOxP")
project = rf.workspace("cheah-ui-zhe").project("pdi-imbge")
model = project.version(3).model

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    device_code TEXT,
    image_path TEXT,
    binary_image_path TEXT,
    annotated_image_path TEXT,
    egg_count INTEGER
)
""")
conn.commit()

def to_binary(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    binary_path = os.path.join(RESULTS_DIR, os.path.basename(image_path).replace(".jpg", "_binary.jpg"))
    cv2.imwrite(binary_path, binary)
    return binary_path

def detect_eggs(image_path):
    result = model.predict(image_path).json()
    img = cv2.imread(image_path)
    if img is None:
        return 0, None
    egg_count = 0
    for pred in result.get("predictions", []):
        try:
            x, y = int(pred['x']), int(pred['y'])
            w, h = int(pred['width']), int(pred['height'])
            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x + w // 2, y + h // 2
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{pred['class']} ({pred['confidence']:.2f})",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            egg_count += 1
        except:
            pass
    annotated_path = os.path.join(RESULTS_DIR, os.path.basename(image_path).replace(".jpg", "_annotated.jpg"))
    cv2.imwrite(annotated_path, img)
    return egg_count, annotated_path

@app.route("/predict", methods=["POST"])
def predict():
    if not check_api(request):
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    upload_path = os.path.join(UPLOAD_DIR, filename)
    file.save(upload_path)

    binary_path = to_binary(upload_path)
    egg_count, annotated_path = detect_eggs(upload_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO results (timestamp, device_code, image_path, binary_image_path, annotated_image_path, egg_count)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (timestamp, DEVICE_CODE, upload_path, binary_path, annotated_path, egg_count))
    conn.commit()

    df = pd.read_sql_query("SELECT * FROM results", conn)
    df.to_csv(CSV_PATH, index=False)

    upload_blob = upload_to_bucket(upload_path)
    binary_blob = upload_to_bucket(binary_path) if binary_path else None
    annotated_blob = upload_to_bucket(annotated_path)

    image_url = get_signed_url(upload_blob)
    binary_url = get_signed_url(binary_blob) if binary_blob else None
    annotated_url = get_signed_url(annotated_blob)

    payload = {
        "device_code": DEVICE_CODE,
        "image_original": image_url,
        "image_processed": annotated_url,
        "egg_counts": egg_count,
        "datetime_captured": timestamp,
        "api_token": API_TOKEN
    }

    try:
        r = requests.post(API_URL, data=payload, timeout=10)
    except:
        r = None

    return jsonify({
        "timestamp": timestamp,
        "device_code": DEVICE_CODE,
        "egg_count": egg_count,
        "image_url": image_url,
        "binary_image_url": binary_url,
        "annotated_image_url": annotated_url,
        "api_response": r.text if r else "Failed to send"
    })

@app.route("/files/<path:filename>")
def serve_file(filename):
    for folder in [UPLOAD_DIR, RESULTS_DIR]:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return send_from_directory(folder, filename)
    return jsonify({"error": "File not found"}), 404

@app.route("/")
def home():
    return jsonify({
        "message": "CDEWS Egg Detection API Running",
        "predict_endpoint": "/predict",
        "device_code": DEVICE_CODE
    })

app.run(host="0.0.0.0", port=5000)
