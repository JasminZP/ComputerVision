import cv2
import numpy as np


cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()
grey1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
grey1 = cv2.convertScaleAbs(grey1, alpha=1.5, beta=15)


lower = np.array([129, 0, 0])
upper = np.array([179, 196, 255])
mask = cv2.inRange(frame1, lower, upper)
img = cv2.bitwise_and(frame1, frame1, mask=mask)


while True:
    ret, frame2 = cap.read()
    if not ret:
        print("Can't receive frame Prosti")
        break

    grey2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    grey2 = cv2.convertScaleAbs(grey2, alpha=1.5, beta=15)

    diff = cv2.absdiff(grey1, grey2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (255, 0, 0), 2)
            perimeter = cv2.arcLength(cnt, True)
            M = cv2.moments(cnt)
            area = cv2.contourArea(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                x, y, w, h = cv2.boundingRect(cnt)

                approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, True)

                if len(approx) == 3:
                    shape = "Triangle"
                if len(approx) == 4:
                    shape = "Quadratic"
                if len(approx) > 12:
                    shape = "Kolo/Oval"
                else:
                    shape = "Inshe"

                cv2.drawContours(frame2, [approx], -1, (0, 255, 0), 2)
                cv2.circle(frame2, (cx, cy), 4, (0, 255, 0), 1)
                cv2.putText(frame2, f'area: {area}, shape: {shape}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 0, 255), 1)
                cv2.putText(frame2, f'Center x: {cx}, y: {cy}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 0, 255), 1)


    cv2.imshow('Pavlo', grey2)
    cv2.imshow('Pavlo', frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break






cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()