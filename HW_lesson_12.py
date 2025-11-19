import numpy as np
import tensorflow as tf
import keras as keras
from keras import layers, models
from keras.preprocessing.image import load_img, img_to_array




# 1. Завантаження датасету
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/train",
    image_size=(128, 128),
    batch_size=30,
    label_mode="categorical"
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/test",
    image_size=(128, 128),
    batch_size=30,
    label_mode="categorical"
)

# Отримання списку класів автоматично
class_names = train_ds.class_names
num_classes = len(class_names)
print("Класи:", class_names)

# 2. Нормалізація
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))


# 3. Модель CNN + додано extra Conv2D блок
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # EXTRA BLOCK — доданий за умовою
    layers.Conv2D(96, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 4. Навчання моделі
history = model.fit(train_ds, epochs=50, validation_data=test_ds)

# 5. Оцінка
test_loss, test_acc = model.evaluate(test_ds)
print("\nТочність на тестових даних:", test_acc)


# 6. Передбачення для одного зображення
img = load_img("data/test/apple/3_100.jpg", target_size=(128, 128))
img_array = img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predict_index = np.argmax(prediction[0])

print("Ймовірності:", prediction[0])
print("Передбачений клас:", class_names[predict_index])
