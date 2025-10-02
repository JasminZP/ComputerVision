import cv2


img = cv2.imread("images/silver_grey.png")
img2 = cv2.imread("images/img.png")
img3 = cv2.imread("images/qr.png")

img3 = cv2.resize(img3, (100, 100))
img2 = cv2.resize(img2, (110, 130))
img = cv2.resize(img, (600, 400))
cv2.rectangle(img, (15, 15), (585, 385), (94, 72, 58), 2)
cv2.rectangle(img, (448, 218), (552, 322), (94, 72, 58), 2)
cv2.putText(img, "Pavlo Zinchuk", (200, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
cv2.putText(img, "Computer Vision Student  ", (200, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
cv2.putText(img, "Email: pashka.zinchuk@gmail.com", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (61, 79, 55))
cv2.putText(img, "Phone: +380957323923", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (61, 79, 55))
cv2.putText(img, "23/09/2009", (200, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (61, 79, 55))
cv2.putText(img, "OpenCV Business Card", (100, 370), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))


r, t = img3.shape[:2]
x1, y1 = 450, 220
img[y1:y1+r, x1:x1+t] = img3

h, w = img2.shape[:2]
x, y = 35, 35
img[y:y+h, x:x+w] = img2
cv2.imshow("result", img)
cv2.imwrite("business_card.png", img)


cv2.waitKey(0)
cv2.destroyAllWindows()