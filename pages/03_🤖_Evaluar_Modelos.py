import streamlit as st
from tensorflow import keras
from keras.models import load_model
from PIL import Image
import pandas as pd
import numpy as np
from streamlit_drawable_canvas import st_canvas
import psycopg2
import os
from dotenv import load_dotenv
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
conn = psycopg2.connect(DATABASE_URL)
query = "SELECT * FROM datos"
df = pd.read_sql_query(query, conn)
conn.close()


# Definir una función para convertir la lista anidada en un numpy.ndarray
def parse_vector(vector_list):
    return np.array(vector_list).reshape(28, 28).astype(np.float32)

df['VectorNum'] = df['vector'].apply(parse_vector)
train_vectors = df['VectorNum'].to_list()
train_vectors = np.array(train_vectors)
labels = df['etiqueta'].tolist()
labels = np.array(labels)

imagen = Image.open('media/modelos2.png')
st.sidebar.image(imagen, caption='', use_column_width=True)
model = st.sidebar.selectbox("Modelo a Evaluar", ("model_retrained.h5","model_mnist.h5","model_mix.h5"), index=0)
loaded_model = load_model(model)


predictions = loaded_model.predict(train_vectors)
predicted_labels = np.argmax(predictions, axis=1)
confusion = confusion_matrix(labels, predicted_labels)
class_totals = confusion.sum(axis=1, keepdims=True)
confusion_percentages = confusion / class_totals
class_names = [str(i) for i in range(10)]

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_percentages, annot=True, fmt='.1%', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Etiquetas Predichas')
plt.ylabel('Etiquetas Correctas')
plt.title('Matriz de Confusión retrained_model (Porcentajes)')
st.pyplot(plt)