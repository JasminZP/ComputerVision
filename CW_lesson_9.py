import cv2
import numpy as np

net = cv2.dnn.readNetFromCaffe('data/MobileNet/mobilenet_deploy.prototxt', 'data/MobileNet/mobilenet.caffemodel')
#Model zavantajuemo

#Zjituvati spisok vsi klassi
classes = []
with open("data/MobileNet/synset.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)

#Zobrajenya
im = cv2.imread("images/MobileNet/image.jpg")
im = cv2.resize(im, (700, 500))
#Adaptacia
blob = cv2.dnn.blobFromImage(
    cv2.resize(im, (224, 224)),
    1.0 / 127.5,
    (224, 224),
    (127.5, 127.5, 127.5)
)

#v mereju
net.setInput(blob)
preds = net.forward()

#index classa z naibilsheu imovirnistue
idx = preds[0].argmax()



label = classes[idx] if idx < len(classes) else "unknown"
confidence = float(preds[0][idx]) * 100
print("Class :", label)
print("Confidence :", confidence, "%")

text = f'{label}: {int(confidence)}%'
cv2.putText(im, text, (10, 25), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)


cv2.imshow("Image", im)
cv2.waitKey(0)
cv2.destroyAllWindows()