import cv2
import os
import csv
import sys
import subprocess
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIDEO_DIR = os.path.join(OUTPUT_DIR, 'videos')
os.makedirs(VIDEO_DIR, exist_ok=True)

INPUT_VIDEO_PATH = os.path.join(VIDEO_DIR, 'video.mp4')
OUTPUT_VIDEO_PATH = os.path.join(VIDEO_DIR, 'output_video.mp4')


USE_WEBCAM = False
USE_YOUTUBE = True

YOUTUBE_URL = "https://www.youtube.com/live/Lxqcg1qt0XU?si=DKZ6P7lRL65PqNRU"

def get_youtube_stream_url(url: str) -> str:
    cmd = [sys.executable, "-m", "yt_dlp", "-g", url]
    out = subprocess.check_output(cmd, text=True).strip()
    return out.splitlines()[0]

if USE_WEBCAM:
    source = 0
elif USE_YOUTUBE:
    source = get_youtube_stream_url(YOUTUBE_URL)
else:
    source = INPUT_VIDEO_PATH


MODEL_PATH = "yolov8s.pt"
CONF_THRESH = 0.5
TRACKER = "bytetrack.yaml"

SAVE_VIDEO = True
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(source)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps != fps:
    fps = 30.0

writer = None
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))


ROI = (250, 420, 1720, 620)
ROI_DISTANCE_METERS = 8.0

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

def point_in_roi(cx, cy, roi):
    x1, y1, x2, y2 = roi
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)


CSV_PATH = os.path.join(OUTPUT_DIR, "speed_log.csv")
csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["track_id", "class", "enter_time_s", "exit_time_s", "duration_s", "speed_mps", "speed_kmh"])


seen_id_total = set()
seen_id_class = {}

track_state = {}


MAX_ROWS_ON_SCREEN = 10
speed_rows = []

frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1

    result = model.track(frame, conf=CONF_THRESH, tracker=TRACKER, persist=True, verbose=False)
    r = result[0]


    x1r, y1r, x2r, y2r = ROI
    cv2.rectangle(frame, (x1r, y1r), (x2r, y2r), (255, 255, 0), 2)
    cv2.putText(frame, f"ROI {ROI_DISTANCE_METERS:.1f}m", (x1r, max(20, y1r - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if r.boxes is None or len(r.boxes) == 0:

        table_x = 10
        table_y = 60
        row_h = 25

        cv2.putText(frame, "ID   CLASS      SPEED", (table_x, table_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        for idx, (rid, rcls, rkmh) in enumerate(speed_rows):
            y = table_y + (idx + 1) * row_h
            txt = f"{rid:<4} {rcls:<10} {rkmh:>6.1f} km/h"
            cv2.putText(frame, txt, (table_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('frame', frame)
        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    boxes = r.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    track_id = None
    if boxes.id is not None:
        track_id = boxes.id.cpu().numpy()

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].astype(int)
        class_id = int(cls[i])
        class_name = model.names[class_id]
        score = float(conf[i])

        if class_name not in VEHICLE_CLASSES:
            continue

        tid = -1
        if track_id is not None:
            tid = int(track_id[i])
        if tid == -1:
            continue

        seen_id_total.add(tid)
        if class_name not in seen_id_class:
            seen_id_class[class_name] = set()
        seen_id_class[class_name].add(tid)


        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        inside_now = point_in_roi(cx, cy, ROI)

        if tid not in track_state:
            track_state[tid] = {
                "inside": False,
                "enter_frame": None,
                "done": False,
                "last_speed_kmh": None,
                "class": class_name
            }

        st = track_state[tid]


        if inside_now and (st["inside"] is False) and (st["done"] is False):
            st["inside"] = True
            st["enter_frame"] = frame_index


        measuring = st["inside"] and (st["done"] is False)

        if (not inside_now) and st["inside"] and (st["done"] is False):
            st["inside"] = False

            enter_f = st["enter_frame"]
            exit_f = frame_index

            if enter_f is not None and exit_f > enter_f:
                duration_s = (exit_f - enter_f) / fps
                if duration_s > 0.05:
                    speed_mps = ROI_DISTANCE_METERS / duration_s
                    speed_kmh = speed_mps * 3.6

                    st["last_speed_kmh"] = speed_kmh
                    st["done"] = True

                    enter_time_s = enter_f / fps
                    exit_time_s = exit_f / fps

                    csv_writer.writerow([tid, st["class"],
                                         f"{enter_time_s:.3f}", f"{exit_time_s:.3f}",
                                         f"{duration_s:.3f}",
                                         f"{speed_mps:.3f}", f"{speed_kmh:.2f}"])
                    csv_file.flush()


                    speed_rows.append((tid, st["class"], float(f"{speed_kmh:.1f}")))
                    speed_rows = speed_rows[-MAX_ROWS_ON_SCREEN:]


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)


        if st["last_speed_kmh"] is not None:
            speed_text = f"{st['last_speed_kmh']:.1f} km/h"
            cv2.putText(frame, speed_text, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif measuring:
            cv2.putText(frame, "measuring...", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


        label = f"{class_name} Id {tid}"
        cv2.putText(frame, label, (x1, min(frame_height - 10, y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    total = len(seen_id_total)
    cv2.putText(frame, f'unique vehicles {total}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    table_x = 10
    table_y = 60
    row_h = 25

    cv2.putText(frame, "ID   CLASS      SPEED", (table_x, table_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for idx, (rid, rcls, rkmh) in enumerate(speed_rows):
        y = table_y + (idx + 1) * row_h
        txt = f"{rid:<4} {rcls:<10} {rkmh:>6.1f} km/h"
        cv2.putText(frame, txt, (table_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('frame', frame)
    if writer is not None:
        writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if writer is not None:
    writer.release()
csv_file.close()
cv2.destroyAllWindows()

print(f"Done. CSV saved to: {CSV_PATH}")
if SAVE_VIDEO:
    print(f"Video saved to: {OUTPUT_VIDEO_PATH}")
