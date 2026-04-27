import cv2
from ultralytics import YOLO
from playsound import playsound
import threading
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

candidate_models = [
    BASE_DIR / "YOLOv8-Fire-and-Smoke-Detection-main" / "runs" / "detect" / "train" / "weights" / "best.pt",
    BASE_DIR / "fire.pt",
]

model_path = next((path for path in candidate_models if path.exists()), None)
if model_path is None:
    raise FileNotFoundError("No YOLO checkpoint found. Expected best.pt or fire.pt in the project folder.")

# Load the first available trained model.
print(f"Using model: {model_path}")
model = YOLO(str(model_path))

cap = cv2.VideoCapture(0)

alert_playing = False
alarm_path = BASE_DIR / "alarm.wav"

def play_alarm():
    global alert_playing
    try:
        if alarm_path.exists():
            try:
                playsound(str(alarm_path))
            except Exception:
                subprocess.run(["afplay", str(alarm_path)], check=False)
    finally:
        alert_playing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)

    fire_detected = False

    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names.get(cls, str(cls)).lower()

                # Identify fire by label name so the script stays correct even if class order changes.
                if label == "fire":
                    fire_detected = True

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "FIRE", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Play alarm once (non-blocking)
    if fire_detected and not alert_playing:
        alert_playing = True
        threading.Thread(target=play_alarm).start()

    cv2.imshow("Fire Detection (YOLOv8)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()