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

st.title("🤖 Evaluación de Modelos 🤖")

st.info("En la siguiente matriz podrás ver la efectividad y precisión del modelo elegido")
st.markdown("La línea diagonal que va de arriba hacia abajo y de izquierda a derecha corresponde a las etiquetas correctamente predichas por el modelo")

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_percentages, annot=True, fmt='.1%', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Etiquetas Predichas')
plt.ylabel('Etiquetas Correctas')
plt.title(f'Matriz de Confusión {model} (Porcentajes)')
st.pyplot(plt)
st.info(f"Si observamos la fila 0, columna 0, vemos el porcentaje de veces que el modelo acertó correctamente la etiqueta 0. ({round(confusion_percentages[0,0]*100, 2)}%)")
st.info(f"Si recorremos la fila 0, podemos ver el porcentaje de veces que el modelo predijo un número distinto de 0 cuando el dibujo era un 0. ({round(confusion_percentages[0,1:10].sum()*100, 2)}%)")
st.info(f"Si recorremos la columna 0, haciendo cálculos auxiliares podemos obtener el porcentaje de veces que el modelo predijo 0 cuando el dibujo era distinto de 0. ({round((confusion_percentages[1:10,0].sum()*100) * (100/(confusion_percentages[0,0]*100)), 2)}%)")
st.markdown("Del mismo modo se puede seguir recorriendo la matriz para visualizar las predicciones, los aciertos y los errores en cada una de las filas correspondientes a cada etiqueta")

st.sidebar.markdown("### Contactá con el autor del sitio")
st.sidebar.markdown("[Jeremías Pombo en LinkedIn](https://www.linkedin.com/in/jeremiaspombo/)")
st.sidebar.markdown("### Visitá el repositorio del proyecto")
st.sidebar.markdown("[Repositorio de GitHub](https://github.com/Jeremias44/digiartvisionsql)")
