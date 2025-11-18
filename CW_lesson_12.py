import numpy as np
import tensorflow as tf #neironka rozchirena
from tensorflow import keras
from tensorflow.keras import layers, models #shari
from tensorflow.keras.preprocessing.image import image
from tensorflow.python.layers.normalization import normalization

train_ds = tf.keras.preprocessing.image_dataset_from_directory("data/train",
                                                               image_size=(128, 128),
                                                               batch_size=30,
                                                               label_mode="categorical")
test_ds = tf.keras.preprocessing.image_dataset_from_directory("data/test",
                                                               image_size=(128, 128),
                                                               batch_size=30,
                                                               label_mode="categorical")

normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(train_ds, epochs=50, validation_data=test_ds)

test_loss, test_acc = model.evaluate(test_ds)
print('\nTest accuracy:', test_acc)

class_name = ["cars", "cats", "dogs"]

img = image.load_img("data/test/", target_size=(128, 128))

image_array = image.img_to_array(img)
image_array - image_array // 255.0
image_array = np.expand_dims(image_array, axis=0)
prediction = model.predict(image_array)
predict_index = np.argmax(prediction[0])
print(f'Predicted: {prediction[0]}')
print(f'Viznacheno: {class_name[predict_index]}')