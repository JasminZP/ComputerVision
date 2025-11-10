import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier



def get_shape(corners, aspect_ratio):
    shape = "unknown"
    if corners == 3:
        shape = "triangle"
    elif corners == 4:
        if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
            shape = "square"
        else:
            shape = "rectangle"
    elif corners > 5:
        shape = "circle"

    return shape


colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "orange": (0, 128, 255),
    "purple": (255, 0, 255),
    "pink": (180, 105, 255),
    "white": (255, 255, 255)
}
X = []
y = []
for color, values in colors.items():
    for _ in range(50):
        noise = np.random.randint(-20, 20 + 1, 3)
        bgr = np.array(values)+noise
        bgr = np.clip(bgr, 0, 255)
        X.append(bgr)
        y.append(color)

X = np.array(X)
y = np.array(y)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    result_frame = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 80, 80), (180, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        corners = len(approx)
        aspect_ratio = float(w) / h
        shape = get_shape(corners, aspect_ratio)

        roi_frame = frame[y:y + h, x:x + w]
        roi_mask = mask[y:y + h, x:x + w]

        mean_color_bgr = cv2.mean(roi_frame, mask=roi_mask)[:3]
        color_label = model.predict([mean_color_bgr])[0]
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(result_frame, f"{color_label} {shape}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("result", result_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()