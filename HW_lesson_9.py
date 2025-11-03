import cv2
import numpy as np

# Завантаження моделі MobileNet
net = cv2.dnn.readNetFromCaffe(
    'data/MobileNet/mobilenet_deploy.prototxt',
    'data/MobileNet/mobilenet.caffemodel'
)

# Зчитуємо список класів
classes = []
with open("data/MobileNet/synset.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)


im1 = cv2.imread("images/MobileNet/1.jpg")
im2 = cv2.imread("images/MobileNet/5090.jpg")
im3 = cv2.imread("images/MobileNet/cat.jpg")
im4 = cv2.imread("images/MobileNet/image.jpg")

images = [("1.jpg", im1), ("5090.jpg", im2), ("cat.jpg", im3), ("image.jpg", im4)]
results = []


def classify_image(name, img):
    if img is None:
        print(f"⚠️ Не вдалося відкрити {name}")
        return "unknown"

    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (224, 224)),
        1.0 / 127.5,
        (224, 224),
        (127.5, 127.5, 127.5)
    )

    net.setInput(blob)
    preds = net.forward()
    idx = preds[0].argmax()

    label = classes[idx] if idx < len(classes) else "unknown"
    confidence = float(preds[0][idx]) * 100

    print(f"\n📸 Файл: {name}")
    print(f"➡️ Клас: {label}")
    print(f"✅ Впевненість: {confidence:.2f}%")

    text = f'{label}: {int(confidence)}%'
    img_show = cv2.resize(img, (400, 300))
    cv2.putText(img_show, text, (10, 25), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    cv2.imshow(name, img_show)

    results.append(label)
    return label


label1 = classify_image("1.jpg", im1)
label2 = classify_image("5090.jpg", im2)
label3 = classify_image("cat.jpg", im3)
label4 = classify_image("image.jpg", im4)


labels = [label1, label2, label3, label4]
unique_labels = set(labels)

print("\n📊 Таблиця частоти класів:")
print("-" * 40)
print(f"{'Клас':30} | Кількість")
print("-" * 40)
for label in unique_labels:
    print(f"{label:30} | {labels.count(label)}")
print("-" * 40)

cv2.waitKey(0)
cv2.destroyAllWindows()
