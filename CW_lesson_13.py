import cv2
import numpy
import os
import shutil

project_dir = os.path.dirname(__file__)

image_dir = os.path.join(project_dir, "images")

models_dir = os.path.join(project_dir, "models")

out_dir = os.path.join(project_dir, "out")

people_dir = os.path.join(out_dir, "people")
no_people_dir = os.path.join(out_dir, "nopeopledir")

os.makedirs(out_dir, exist_ok=True)
os.makedirs(people_dir, exist_ok=True)
os.makedirs(no_people_dir, exist_ok=True)

cascade_patch = os.path.join(models_dir, 'haarcascade_frontalface_default.xml')

face_cascade = cv2.CascadeClassifier(cascade_patch)

if face_cascade.empty():
    print("No face pavel detected")
    exit()

def face_detect(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    return faces
