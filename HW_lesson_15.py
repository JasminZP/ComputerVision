import os
import cv2
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(PROJECT_DIR, "video")
OUT_DIR = os.path.join(PROJECT_DIR, "out")
os.makedirs(OUT_DIR, exist_ok=True)

USE_WEBCAM = True
VIDEO_NAME = "animals.mp4"
RESIZE_WIDTH = 960
CONF_THRESHOLD = 0.5

CAT_CLASS_ID = 15
DOG_CLASS_ID = 16

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, VIDEO_NAME))

model = YOLO("yolov8n.pt")

prev_time = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    cats = 0
    dogs = 0

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == CAT_CLASS_ID:
                cats += 1
                color = (255, 0, 0)
                label = f"Cat {conf:.2f}"
            elif cls == DOG_CLASS_ID:
                dogs += 1
                color = (0, 255, 0)
                label = f"Dog {conf:.2f}"
            else:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    total = cats + dogs

    now = time.time()
    fps = 1.0 / (now - prev_time) if now != prev_time else fps
    prev_time = now

    cv2.putText(frame, f"Cats: {cats}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Dogs: {dogs}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Total animals: {total}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("YOLO Animals", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
