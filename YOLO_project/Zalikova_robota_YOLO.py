import os
import cv2
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)

VIDEO_DIR = os.path.join(PROJECT_DIR, 'video')
OUT_DIR = os.path.join(PROJECT_DIR, 'out')
os.makedirs(OUT_DIR, exist_ok=True)

USE_WEBCAM = False

VIDEO_PATH = os.path.join(VIDEO_DIR, os.listdir(VIDEO_DIR)[0])
cap = cv2.VideoCapture(VIDEO_PATH)


model = YOLO('yolov8n.pt')
CONF_TRESHOLD = 0.5
RESIZE_WIDTH = 960

prev_time = time.time()
fps = 0.0

TRANSPORT_CLASS_IDS = {1, 2, 3, 5, 6, 7, 8}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))

    result = model(frame, conf=CONF_TRESHOLD, verbose=False)

    counts = {1: 0, 2: 0, 3: 0, 5: 0, 6: 0, 7: 0, 8: 0}

    for r in result:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls not in TRANSPORT_CLASS_IDS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            counts[cls] += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            name = model.names.get(cls, str(cls))
            cv2.putText(frame, f'{name} {conf:.2f}', (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    now = time.time()
    dt = now - prev_time
    prev_time = now
    if dt > 0:
        fps = 1.0 / dt

    y = 25
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    y += 25

    for cls_id in [2, 7, 5, 3, 1, 6, 8]:
        name = model.names.get(cls_id, str(cls_id))
        cv2.putText(frame, f'{name}: {counts[cls_id]}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 25

    cv2.imshow('YOLO', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
