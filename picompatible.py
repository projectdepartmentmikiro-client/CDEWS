import os
import time
import requests
import csv
import re
import threading
import numpy as np
import cv2
from flask import (
    Flask,
    Response,
    render_template,
    url_for,
    send_from_directory,
    jsonify,
)

app = Flask(__name__)
camera_qr = cv2.VideoCapture(0)
camera_sample = cv2.VideoCapture(1)

for cam in [camera_qr, camera_sample]:
    cam.set(3, 640)  # width
    cam.set(4, 480)  # height

DATA_DIR = "data"
QR_CROP_DIR = os.path.join(DATA_DIR, "qr_crops")
SAMPLE_DIR = os.path.join(DATA_DIR, "samples")
for folder in [DATA_DIR, QR_CROP_DIR, SAMPLE_DIR]:
    os.makedirs(folder, exist_ok=True)

CSV_FILE = os.path.join(DATA_DIR, "qr_data.csv")
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "qr_crop_filename",
                "qr_text",
                "sample_image_filename",
                "collector",
                "species",
                "location",
                "notes",
                "egg_count",
            ]
        )

qr_detector = cv2.QRCodeDetector()
scanned_qrs_lock = threading.Lock()
scanned_qrs = set()

last_qr_lock = threading.Lock()
last_qr_data = {
    "timestamp": "",
    "qr_crop_filename": "",
    "qr_text": "",
    "sample_image_filename": "",
    "collector": "",
    "species": "",
    "location": "",
    "notes": "",
    "egg_count": "",
}

sample_image_lock = threading.Lock()
sample_image = None  

qr_display_lock = threading.Lock()
qr_display_mode = "live"
freeze_qr_image = None


def automated_capture(interval_hours=4):
    """
    Automatically capture a sample every interval_hours.
    """
    while True:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        sample_filename = safe_filename("auto_sample")
        sample_path = os.path.join(SAMPLE_DIR, sample_filename)
        ret, frame = camera_sample.read()
        if ret and frame is not None:
            if cv2.imwrite(sample_path, frame):
                with sample_image_lock:
                    global sample_image
                    sample_image = frame.copy()
                print(f"[INFO] Automated capture saved: {sample_filename}")
                # Optionally send to Colab
                threading.Thread(target=send_sample_to_colab, args=(sample_path,), daemon=True).start()
                # Update last_qr_data without QR info
                with last_qr_lock:
                    last_qr_data.update({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "qr_crop_filename": "",
                        "qr_text": "",
                        "sample_image_filename": sample_filename,
                        "collector": "",
                        "species": "",
                        "location": "",
                        "notes": "",
                        "egg_count": "",
                    })
            else:
                print(f"[WARN] Failed to write automated sample: {sample_path}")
        else:
            print("[WARN] No frame captured for automated sample")

        time.sleep(interval_hours * 3600)  # wait for next interval

def safe_filename(prefix: str, text: str = "") -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if not text:
        return f"{prefix}_{timestamp}.jpg"
    safe = re.sub(r"[^\w\-_. ]", "_", text).strip().replace(" ", "_")
    return f"{prefix}_{safe}_{timestamp}.jpg"


def send_sample_to_colab(sample_path: str):
    """
    Send captured image to Colab for egg counting.
    """
    colab_url = "https://4866911d8b21.ngrok-free.app/predict"  
    try:
        with open(sample_path, "rb") as f:
            files = {"file": f}
            response = requests.post(colab_url, files=files, timeout=10)
        if response.status_code == 200:
            data = response.json()
            egg_count = data.get("egg_count", 0)
            print(f"[INFO] Colab counted {egg_count} eggs")
            with last_qr_lock:
                last_qr_data["egg_count"] = egg_count
        else:
            print(f"[WARN] Colab returned status {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Failed to send sample to Colab: {e}")


def capture_sample_after_delay(sample_path: str, delay: int = 3):
    """
    Capture sample from microscope camera after delay and update global sample_image.
    """
    global sample_image
    time.sleep(delay)
    for _ in range(8):
        ret, frame = camera_sample.read()  
        if ret and frame is not None:
            if cv2.imwrite(sample_path, frame):
                with sample_image_lock:
                    sample_image = frame.copy()
                # Send to Colab asynchronously
                threading.Thread(
                    target=send_sample_to_colab, args=(sample_path,), daemon=True
                ).start()
            else:
                print(f"[WARN] Failed to write sample image: {sample_path}")
            return
        time.sleep(0.2)
    print(f"[WARN] No valid frame captured for {sample_path}")

def generate_frames_qr():
    global freeze_qr_image, qr_display_mode, sample_image
    while True:
        ret, frame = camera_qr.read()  
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        with qr_display_lock:
            mode = qr_display_mode

        if mode == "live":
            data, bbox, _ = qr_detector.detectAndDecode(frame)
            if bbox is not None and len(bbox) > 0:
                bbox = bbox.astype(int).reshape(-1, 2)
                for i in range(len(bbox)):
                    pt1 = tuple(bbox[i])
                    pt2 = tuple(bbox[(i + 1) % len(bbox)])
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

                if data:
                    with scanned_qrs_lock:
                        if data not in scanned_qrs:
                            scanned_qrs.add(data)
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            x_min, y_min = np.min(bbox, axis=0)
                            x_max, y_max = np.max(bbox, axis=0)
                            h, w = frame.shape[:2]
                            x_min, y_min, x_max, y_max = map(
                                int,
                                [
                                    max(0, x_min),
                                    max(0, y_min),
                                    min(w, x_max),
                                    min(h, y_max),
                                ],
                            )

                            qr_crop = frame[y_min:y_max, x_min:x_max]
                            crop_filename = safe_filename("qr_crop", data)
                            crop_path = os.path.join(QR_CROP_DIR, crop_filename)
                            if cv2.imwrite(crop_path, qr_crop) is False:
                                print(f"[WARN] Failed to write QR crop: {crop_path}")

                            with qr_display_lock:
                                qr_display_mode = "freeze"
                                freeze_qr_image = (
                                    qr_crop.copy() if qr_crop is not None else None
                                )

                            sample_filename = safe_filename("sample", data)
                            sample_path = os.path.join(SAMPLE_DIR, sample_filename)

                            with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                                writer = csv.writer(f)
                                writer.writerow(
                                    [
                                        timestamp,
                                        crop_filename,
                                        data,
                                        sample_filename,
                                        "",
                                        "",
                                        "",
                                        "",
                                        "",
                                    ]
                                )

                            with last_qr_lock:
                                last_qr_data.update(
                                    {
                                        "timestamp": timestamp,
                                        "qr_crop_filename": crop_filename,
                                        "qr_text": data,
                                        "sample_image_filename": sample_filename,
                                        "collector": "",
                                        "species": "",
                                        "location": "",
                                        "notes": "",
                                        "egg_count": "",
                                    }
                                )

                            threading.Thread(
                                target=capture_sample_after_delay,
                                args=(sample_path, 3),
                                daemon=True,
                            ).start()

                            def unfreeze_after_delay():
                                time.sleep(3)
                                with qr_display_lock:
                                    global qr_display_mode
                                    qr_display_mode = "live"
                                    freeze_qr_image = None

                            threading.Thread(
                                target=unfreeze_after_delay, daemon=True
                            ).start()

                    try:
                        label = data.splitlines()[0]
                    except Exception:
                        label = str(data)
                    cv2.putText(
                        frame,
                        label,
                        (
                            bbox[0][0],
                            (
                                bbox[0][1] - 10
                                if bbox[0][1] - 10 > 10
                                else bbox[0][1] + 20
                            ),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2,
                    )

            ret2, jpeg = cv2.imencode(".jpg", frame)
            if ret2:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"

        elif mode == "freeze":
            with qr_display_lock:
                img_to_show = (
                    freeze_qr_image
                    if freeze_qr_image is not None
                    else np.zeros((480, 640, 3), dtype=np.uint8)
                )
                ret2, jpeg = cv2.imencode(".jpg", img_to_show)
            if ret2:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            time.sleep(0.5)


def generate_frames_sample():
    global sample_image
    while True:
        with sample_image_lock:
            img_to_show = (
                sample_image
                if sample_image is not None
                else np.zeros((480, 640, 3), dtype=np.uint8)
            )
            ret2, jpeg = cv2.imencode(".jpg", img_to_show)
        if ret2:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        time.sleep(1)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_qr")
def video_qr():
    return Response(
        generate_frames_qr(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/video_sample")
def video_sample():
    return Response(
        generate_frames_sample(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/last_qr")
def last_qr():
    with last_qr_lock:
        data = last_qr_data.copy()
        data["qr_crop_url"] = (
            url_for("serve_qr_crop", filename=data["qr_crop_filename"])
            if data["qr_crop_filename"]
            else None
        )
        data["sample_image_url"] = (
            url_for("static", filename=f"data/samples/{data['sample_image_filename']}")
            if data["sample_image_filename"]
            else None
        )
    return jsonify(data)


@app.route("/qr_crops/<filename>")
def serve_qr_crop(filename):
    return send_from_directory(QR_CROP_DIR, filename)


if __name__ == "__main__":
    threading.Thread(target=automated_capture, args=(4,), daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=True)
