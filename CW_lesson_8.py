import cv2
import numpy as np


#face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_smile.xml')



cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30)) # масштабування кількість перевірок обличчя чи ні
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 20, minSize=(20, 20))
        smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 20, minSize=(20, 20))

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ey, ew),(ex + ew, ey + eh) , (0, 255, 0), 2)
        for ex1, ey1, ew1, eh1 in smile:
            cv2.rectangle(roi_color, (ey1, ew1), (ex1 + ew1, ey1 + eh1), (0, 0, 255), 2)


        cv2.putText(frame, f'Faces detected: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    cv2.imshow('Pavlo face tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()