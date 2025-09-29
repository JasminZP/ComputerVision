import cv2
import numpy as np

# image = cv2.imread('images/img1.jpg')
# image = cv2.resize(image, (800, 400))
# image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))
# image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
# image = cv2.flip(image, 1)
# image = cv2.GaussianBlur(image, (9, 9), 5)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# print(image.shape)
# image = cv2.Canny(image, 250, 250)
# a = [
#      [0,0,0],
#      [0,0,0],
#      [0,0,0]
#  ]
# kernel = np.ones((2, 2), np.uint8)
# image = cv2.dilate(image, kernel, iterations = 1)
# image = cv2.erode(image, kernel, iterations = 1)
#
#
# cv2.imshow('Pavlo', image)
# cv2.imshow("image", image[0:200, 0:400])


video = cv2.VideoCapture("video/vid1.mp4")
while True:
    mistake, frame = video.read()
    frame = cv2.resize(frame, (800, 600))
    cv2.imshow("video", frame)
    if cv2.waitKey(1) & 0xff == ord('s'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()