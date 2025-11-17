import numpy as np
import pandas as pd #robota z csv tablicami
import tensorflow as tf #neironka rozchirena
from keras import Sequential
from tensorflow import keras
from tensorflow.keras import layers #shari
from sklearn.preprocessing import LabelEncoder #text -> chisla
import matplotlib.pyplot as plt #grafiki


df = pd.read_csv("data/figures.csv")
# print(df.head())

#NAZVI FIGURE NA CIIFRI


encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])


X = df[["area", "perimeter", "corners"]]
y = df["label_enc"]

#model

model = keras.Sequential([
    layers.Dense(8, activation="relu", input_shape=(3,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(8, activation="softmax"),
])


#kompilacia modeli
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#learning
history = model.fit(X, y, epochs=300, verbose=0)

#vizualizacia navchanya


plt.plot(history.history['loss'], label="loss")
plt.plot(history.history['accuracy'], label="accuracy")
plt.xlabel("Epoxa")
plt.ylabel("Znachenya")
plt.title("Znachenya")
plt.legend()
plt.show()


test = np.array([[25, 20, 0]])
pred = model.predict(test)

print(f'Imovirnist kojnogo klassu {pred}')
print(f'Model viznajila {encoder.inverse_transform([np.argmax(pred)])}')