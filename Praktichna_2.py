import cv2
import numpy as np



img = cv2.imread("images/img6.jpg")
img = cv2.resize(img, (500, 660))
img_copy = img.copy()


img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

img = cv2.GaussianBlur(img, (3, 7), 2)


lower_red = np.array([0, 161, 28])
lower_blue = np.array([0, 160, 0])
lower_green = np.array([0, 66, 0])
upper_red = np.array([255, 255, 255])
upper_blue = np.array([179, 255, 182])
upper_green = np.array([179, 255, 255])



mask_red = cv2.inRange(img, lower_red, upper_red)
mask_blue = cv2.inRange(img, lower_blue, upper_blue)
mask_green = cv2.inRange(img, lower_green, upper_green)

mask_total = cv2.bitwise_or(mask_red, mask_blue)
mask_total = cv2.bitwise_or(mask_total, mask_green)

img = cv2.bitwise_and(img, img, mask=mask_total)



contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 120:
        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)

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

            cv2.drawContours(img_copy, [approx], -1, (0, 255, 0), 2)
            cv2.circle(img_copy, (cx, cy), 4, (0, 255, 0), 1)
            cv2.putText(img_copy, f'area: {area}, shape: {shape}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,0.4, (0, 0, 255), 1)
            cv2.putText(img_copy, f'Center x: {cx}, y: {cy}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX,0.4, (0, 0, 255), 1)


blue_contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in blue_contours:
    area = cv2.contourArea(cnt)
    if area > 4800 and area < 5000:
        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(cnt)

            cv2.putText(img_copy, "Kolir: Blue", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX,0.4, (0, 0, 255), 1)
red_contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in red_contours:
    area = cv2.contourArea(cnt)
    if area > 10000:
        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(cnt)

            cv2.putText(img_copy, "Kolir: Red", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

green_contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in green_contours:
    area = cv2.contourArea(cnt)
    if area > 4000 and area < 10000:
        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(cnt)

            cv2.putText(img_copy, "Kolir: Green", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


cv2.imwrite("images/result.png", img_copy)

cv2.imshow("Original", img)
cv2.imshow("Praktichna", img_copy)
cv2.waitKey()
cv2.destroyAllWindows()