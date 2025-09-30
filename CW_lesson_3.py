import cv2
import numpy as np

img = np.zeros((512,400,3), np.uint8)
#rgb = bgr
# img[:] = 168, 50, 149

# img[100:150, 200:280] = 168, 50, 149

cv2.rectangle(img, (100, 100), (200, 200), (168, 50, 149), -2)

cv2.line(img, (100, 100), (200, 200), (124, 78, 41), 2)
cv2.line(img, (0, img.shape[0]//2), (img.shape[1], img.shape[0]//2), (124, 78, 41), 2)
cv2.line(img, (img.shape[1]//2, 0), (img.shape[1]//2, img.shape[0]), (124, 78, 41), 2)



cv2.circle(img, (200, 200), 120, (1, 523, 200), 1)
cv2.putText(img, "Pavel I fill tea again", (60, 300), cv2.FONT_HERSHEY_PLAIN, 1.5, (25, 43, 120), 2)






cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()