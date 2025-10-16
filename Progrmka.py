import cv2
import numpy as np
import random

cap = cv2.VideoCapture(0)

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (400, 400))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 400:
            cv2.rectangle(frame, (0, 0), (400, 400), (0, 255, 0), 2)
            cv2.putText(frame, "Found", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                points.append((cx, cy))
        else:
            cv2.rectangle(frame, (0, 0), (400, 400), (0, 0, 255), 2)
            cv2.putText(frame, "Not!", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Pavlo", frame)
    cv2.imshow("Pavlo1", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




cap.release()
cv2.destroyAllWindows()