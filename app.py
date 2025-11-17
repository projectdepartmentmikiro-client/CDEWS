import os
import time
import threading
import requests
import cv2
from flask import (
    Flask,
    Response,
    render_template,
    jsonify,
    send_from_directory,
    url_for,
)

app = Flask(__name__)

# Camera
camera = cv2.VideoCapture(0)

# Directories
DATA_DIR = "data"
SAMPLE_DIR = os.path.join(DATA_DIR, "samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Shared state
sample_image_lock = threading.Lock()
sample_image = None  # last captured image

freeze_lock = threading.Lock()
freeze_image = None
freeze_mode = False

last_sample_data_lock = threading.Lock()
last_sample_data = {"timestamp": "", "sample_image_filename": "", "egg_count": ""}


def safe_filename(prefQx: str) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{prefix}_{timestamp}.jpg"


def send_sample_to_colab(sample_path: str, retries: int = 3):
    """
    Send captured image to Colab for processing with retry logic.
    """
    colab_url = (
        "https://646827ed19bd.ngrok-free.app/predict"  # Replace with your endpoint
    )
    for attempt in range(retries):
        try:
            with open(sample_path, "rb") as f:
                files = {"file": f}
                response = requests.post(colab_url, files=files, timeout=10)
            if response.status_code == 200:
                data = response.json()
                egg_count = data.get("egg_count", 0)
                print(f"[INFO] Colab counted {egg_count} eggs for {sample_path}")
                with last_sample_data_lock:
                    last_sample_data["egg_count"] = egg_count
                return
            else:
                print(f"[WARN] Colab returned {response.status_code}, retrying...")
        except Exception as e:
            print(f"[ERROR] Failed to send sample (attempt {attempt+1}): {e}")
        time.sleep(2)
    print(f"[ERROR] Could not send {sample_path} to Colab after {retries} attempts.")


def capture_sample():
    """
    Capture sample, freeze livestream, send to Colab, then resume.
    """
    global sample_image, freeze_image, freeze_mode

    ret, frame = camera.read()
    if ret and frame is not None:
        sample_filename = safe_filename("sample")
        sample_path = os.path.join(SAMPLE_DIR, sample_filename)
        cv2.imwrite(sample_path, frame)

        with sample_image_lock:
            sample_image = frame.copy()
        with freeze_lock:
            freeze_image = frame.copy()
            freeze_mode = True

        # Save last sample info
        with last_sample_data_lock:
            last_sample_data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            last_sample_data["sample_image_filename"] = sample_filename
            last_sample_data["egg_count"] = ""

        # Send to Colab asynchronously
        threading.Thread(
            target=send_sample_to_colab, args=(sample_path,), daemon=True
        ).start()

        # Freeze for 3 seconds then resume live stream
        time.sleep(3)
        with freeze_lock:
            freeze_mode = False


def automated_capture_loop(interval_minutes=2):
    """
    Continuously capture samples every interval_minutes.
    """
    while True:
        time.sleep(interval_minutes * 60)
        capture_sample()


def generate_frames():
    global freeze_mode, freeze_image
    while True:
        with freeze_lock:
            if freeze_mode and freeze_image is not None:
                frame = freeze_image.copy()
            else:
                ret, frame = camera.read()
                if not ret or frame is None:
                    frame = cv2.zeros((480, 640, 3), dtype="uint8")

        ret2, jpeg = cv2.imencode(".jpg", frame)
        if ret2:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        time.sleep(0.03)


@app.route("/")
def index():
    return render_template(
        "index.html"
    )  # Your template showing <img src="/video_feed">


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/last_sample")
def last_sample():
    with last_sample_data_lock:
        data = last_sample_data.copy()
        data["sample_image_url"] = (
            url_for("serve_sample", filename=data["sample_image_filename"])
            if data["sample_image_filename"]
            else None
        )
    return jsonify(data)


@app.route("/samples/<filename>")
def serve_sample(filename):
    return send_from_directory(SAMPLE_DIR, filename)


if __name__ == "__main__":
    # Start automated capture in background thread
    threading.Thread(
        target=automated_capture_loop, args=(2,), daemon=True
    ).start()  # every 2 minutes
    app.run(host="0.0.0.0", port=5000, debug=True)
