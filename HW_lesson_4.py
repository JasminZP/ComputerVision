import cv2
import numpy as np


img = cv2.imread('images/img3.1.jpg')
scale = 2
img = cv2.resize(img, (int(img.shape[1] // scale), int(img.shape[0] // scale)))
print(img.shape)

img_copy =  img.copy()
img_copy_color = img.copy()



img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
img_copy = cv2.GaussianBlur(img_copy, (5, 5), 2)

#підсилення контрасту
img_copy = cv2.equalizeHist(img_copy)
img_copy = cv2.Canny(img_copy, 170, 250)

#крайній зовнішній контур
cont, hierarchy = cv2.findContours(img_copy, cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)

#малювання контурів прямоткунікиів та тексту

for cnt in cont:
    area = cv2.contourArea(cnt)
    if area > 30:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.drawContours(img_copy_color, [cnt], -1, (0, 255, 0), 2)
        cv2.rectangle(img_copy_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text_y = y - 5 if y - 5 > 10 else y + 15
        text = f'x:{x}, y:{y}, s:{int(area)}'
        cv2.putText(img_copy_color, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


cv2.imshow('theblackone', img_copy_color)
# cv2.imshow('theblackone', img)
cv2.imshow('theblackone_copy', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()