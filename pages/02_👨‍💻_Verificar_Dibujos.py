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

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
conn = psycopg2.connect(DATABASE_URL)

query = "SELECT * FROM datos"
df = pd.read_sql_query(query, conn)

st.sidebar.info(f"### En esta página podrás contribuir eliminando aquellos dibujos que considerás que fueron cargados con la etiqueta incorrecta, que fueron subidos sin información, o aquellos en los que el ojo humano no puede precisar exactamente de qué número se trata")
etiqueta = st.sidebar.number_input('Para comenzar, ¿Cuál es el número que deseás verificar?:', 0, 9)
limite = st.sidebar.selectbox("¿Cuántos dibujos deseás verificar?", (10,50,500,'todos'), index=3)
st.sidebar.markdown(f"### Cantidad de Dibujos Registrados en la Base de Datos: {len(df)}")
next_power_of_10 = 10 ** len(str(len(df)))
st.sidebar.info(f"## Gracias por contribuir a la limpieza y carga adecuada de los datos! 🏅")
st.sidebar.markdown(f"## ¡¿Querés contribuir al desafío de alcanzar los {next_power_of_10} registros?! 😃")

st.sidebar.markdown("### Contactá con el autor del sitio")
st.sidebar.markdown("[Jeremías Pombo en LinkedIn](https://www.linkedin.com/in/jeremiaspombo/)")
st.sidebar.markdown("### Visitá el repositorio del proyecto")
st.sidebar.markdown("[Repositorio de GitHub](https://github.com/Jeremias44/digiartvisionsql)")

df = df.iloc[::-1].reset_index(drop=True)
df = df[df['etiqueta'] == etiqueta ]
if limite != 'todos':
    df = df.head(limite)

st.subheader("Dibujos almacenados en la base de datos")
for index, row in df.iterrows():
    st.write(f"Dibujo {row['id']}:")
    vector = np.array(row['vector'])
    vector = vector * 255
    vector = vector.reshape(28, 28).astype('uint8')
    image = Image.fromarray(vector)
    st.image(image, caption=f"Etiqueta: {row['etiqueta']}", width=140)

    if st.button(f"Eliminar Dibujo {row['id']}"):
        cursor = conn.cursor()
        id_to_delete = row['id']
        delete_query = "DELETE FROM datos WHERE id = %s"
        cursor.execute(delete_query, (id_to_delete,))
        conn.commit()
        cursor.close()
        st.write(f"Dibujo {row['id']} eliminado de la base de datos.")
