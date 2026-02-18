import cv2
import time
import numpy as np
import os
from ultralytics import YOLO


PROJECT_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(PROJECT_DIR, "video")
VIDEO_PATH = os.path.join(VIDEO_DIR, '1.mp4')
MODEL_PATH = "yolov8n.pt"
CONF_THRES = 0.35

FALL_DY_PX = 35
FALL_DT_SEC = 0.7

LYING_RATIO = 1.25
STAND_RATIO = 0.95
RESET_MOVE_SEC = 1.5

STILL_MOVE_PX = 10
DANGER_AFTER_FALL_SEC = 8

FORGET_TRACK_SEC = 2.0

def centroid(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def dist(a, b):
    return float(np.hypot(a[0]-b[0], a[1]-b[1]))

def wh_ratio(x1, y1, x2, y2):
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return w / h

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video source")

    model = YOLO(MODEL_PATH)

    last_seen = {}
    last_pos = {}
    last_pos_time = {}
    last_ratio = {}
    fall_time = {}
    fall_confirmed = {}
    last_move_time = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()

        results = model.track(
            frame,
            conf=CONF_THRES,
            iou=0.5,
            classes=[0],
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            for b in results[0].boxes:
                if b.id is None:
                    continue

                tid = int(b.id.item())
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cx, cy = centroid(x1, y1, x2, y2)
                ratio = wh_ratio(x1, y1, x2, y2)

                last_seen[tid] = now

                if tid not in last_pos:
                    last_pos[tid] = (cx, cy)
                    last_pos_time[tid] = now
                    last_ratio[tid] = ratio
                    last_move_time[tid] = now
                    fall_confirmed[tid] = False

                dmove = dist((cx, cy), last_pos[tid])
                if dmove >= STILL_MOVE_PX:
                    last_move_time[tid] = now

                lying_now = ratio >= LYING_RATIO
                standing_now = ratio <= STAND_RATIO

                dt = now - last_pos_time[tid]
                dy = cy - last_pos[tid][1]
                ratio_prev = last_ratio[tid]

                last_pos[tid] = (cx, cy)
                last_pos_time[tid] = now
                last_ratio[tid] = ratio

                sudden_down = (dt <= FALL_DT_SEC and dy >= FALL_DY_PX)
                became_lying = (lying_now and ratio_prev < LYING_RATIO)

                if (not fall_confirmed.get(tid, False)) and (sudden_down or became_lying):
                    fall_time[tid] = now
                    fall_confirmed[tid] = True

                danger = False
                if fall_confirmed.get(tid, False):
                    t_fall = fall_time.get(tid, now)
                    since_fall = now - t_fall
                    still_for = now - last_move_time.get(tid, now)

                    if since_fall >= DANGER_AFTER_FALL_SEC and lying_now and still_for >= DANGER_AFTER_FALL_SEC * 0.8:
                        danger = True

                if fall_confirmed.get(tid, False) and not danger:
                    moved_recently = (now - last_move_time.get(tid, now)) <= RESET_MOVE_SEC
                    if standing_now and moved_recently:
                        fall_confirmed[tid] = False
                        fall_time.pop(tid, None)

                color = (255, 255, 0)
                labels = [f"ID {tid}"]

                if fall_confirmed.get(tid, False):
                    color = (0, 165, 255)
                    labels.append("FALL!?")

                if danger:
                    color = (0, 0, 255)
                    labels.append("ALERT!")

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 4, color, -1)

                ytxt = y1 - 8
                for line in labels:
                    cv2.putText(frame, line, (x1, max(20, ytxt)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    ytxt -= 20

        dead = [tid for tid, t in last_seen.items() if (now - t) > FORGET_TRACK_SEC]
        for tid in dead:
            last_seen.pop(tid, None)
            last_pos.pop(tid, None)
            last_pos_time.pop(tid, None)
            last_ratio.pop(tid, None)
            fall_time.pop(tid, None)
            fall_confirmed.pop(tid, None)
            last_move_time.pop(tid, None)

        cv2.imshow("Fall Monitor", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
