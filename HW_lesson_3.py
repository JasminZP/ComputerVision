import cv2
import numpy as np

img = cv2.imread('images/img2.png')
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

cv2.rectangle(img, (150, 65), (350, 325), (122, 44, 110), 2)
cv2.putText(img, "Zinchuk Pavlo", (170, 350), cv2.FONT_HERSHEY_PLAIN, 1.5, (191, 84, 174), 2)


cv2.imshow('Pavlo', img)

cv2.waitKey(0)
cv2.destroyAllWindows()