import cv2
import numpy as np
import shutil
import os

PROJECT_DIR = os.path.dirname(__file__)

IMAGES_DIR = os.path.join(PROJECT_DIR, "images")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

OUT_DIR = os.path.join(PROJECT_DIR, "output")
PEOPLE_DIR = os.path.join(OUT_DIR, "people")
N0_PEOPLE_DIR = os.path.join(OUT_DIR, "no_people")

os.makedirs(PEOPLE_DIR, exist_ok= True)
os.makedirs(N0_PEOPLE_DIR, exist_ok= True)

print("PROJECT_DIR:", PROJECT_DIR)
print("IMAGES_DIR:", IMAGES_DIR)
print("OUT_DIR:", OUT_DIR)
print("PEOPLE_DIR:", PEOPLE_DIR)
print("NO_PEOPLE_DIR:", N0_PEOPLE_DIR)


PROTOTXT_DIR = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.prototxt")
MODEL_DIR = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.caffemodel")


net = cv2.dnn.readNetFromCaffe(PROTOTXT_DIR, MODEL_DIR)

CLASSES = [
"background",
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor"
]

PERSON_CLASS_ID = CLASSES.index("person")

CONF_TRASHHOLD = 0.6


def detect_person(image):
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image, 0.007843, size=(300, 300), mean=(127.5, 127.5, 127.5)
    )

    net.setInput(blob)
    detections = net.forward()

    best_conf = 0.0
    best_box = None

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        class_id = int(detections[0, 0, i, 1])

        if class_id == PERSON_CLASS_ID and confidence > CONF_TRASHHOLD:
            box = detections[0, 0, i, 3:7]
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            if confidence > best_conf:
                best_conf = confidence
                best_box = (x1, y1, x2, y2)

    found = best_box is not None
    return found, best_box, best_conf



allowed_text = (".jpg", ".png", ".jpeg", ".bmp")
files = os.listdir(IMAGES_DIR)

count_people = 0
count_no_people = 0

for file in files:
    if not file.lower().endswith(allowed_text):
        continue
    in_path = os.path.join(IMAGES_DIR, file)

    img = cv2.imread(in_path)
    if img is None:
        print("Не можу прочитати файл:", in_path)
        continue

    found, bext_box, best_conf = detect_person(img)
    if found:
        count_people += 1
        out_path = os.path.join(PEOPLE_DIR, file)
        shutil.copyfile(in_path, out_path)
        print("Copied to:", out_path)

        boxed = img.copy()
        x1, y1, x2, y2 = bext_box
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(boxed, f'Person:{best_conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        boxed_path = os.path.join(PEOPLE_DIR, "boxed_" + file)
        cv2.imwrite(boxed_path, boxed)
    else:
        count_no_people += 1
        out_path = os.path.join(N0_PEOPLE_DIR, file)
        shutil.copyfile(in_path, out_path)

print(count_people)
print(count_no_people)