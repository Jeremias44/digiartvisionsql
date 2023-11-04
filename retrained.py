import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import pandas as pd
import numpy as np

df = pd.read_csv('Streamlit/data')


# Define una función para convertir la cadena de texto en lista de números
def parse_vector(vector_str):
    # Elimina los corchetes y divide por comas
    values = vector_str.strip('[]').split(', ')
    # Convierte los valores a float32
    return [np.float32(value) for value in values]

# Aplica la función a la columna "Vectores" y crea una nueva columna "VectorNum"
df['VectorNum'] = df['Vectores'].apply(parse_vector)
# Hace array de 28,28
df['VectorNum'] = df['VectorNum'].apply(lambda x: np.array(x).reshape(28, 28))
# Convierte la columna 'VectorNum' en una lista de numpy.ndarray
train_vectors = df['VectorNum'].to_list()
# Convierte la lista en un numpy.ndarray de forma (n, 28, 28), donde n es el número de filas
train_vectors = np.array(train_vectors) # Ahora, train_vectors contiene una lista de numpy.ndarray

labels = df['Etiqueta'].tolist()
labels = np.array(labels)


loadedModel = keras.models.load_model("Streamlit/retrained_model.h5")
# Compila el modelo antes de entrenarlo
loadedModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Entrenar el modelo
loadedModel.fit(np.array(train_vectors), labels, epochs=6)

# Una vez que haya terminado el entrenamiento, guardar el modelo nuevamente si es necesario
loadedModel.save("Streamlit/retrained_model.h5")
