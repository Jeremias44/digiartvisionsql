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

def save_data(vector, etiqueta):
    cursor = conn.cursor()
    insert_query = "INSERT INTO datos (vector, etiqueta) VALUES (ARRAY[%s], %s)"
    data = (vector, etiqueta)
    cursor.execute(insert_query, data)
    conn.commit()
    cursor.close()

st.sidebar.title("Navegar Páginas")
seleccion = st.sidebar.selectbox("Ir a la página:", ("Página Principal", "Ver Dibujos"))

if seleccion == "Página Principal":
    st.title('Modelo para reconocer números del 0 al 9 📚🚀💡👨‍💻')
    st.subheader('Este modelo se encuentra en proceso de entrenamiento 🏋️‍♂️ Podés jugar las veces que quieras y estarás ayudando a entrenarlo! 💪')
    st.write("## Para comenzar dibujá un número del 0 al 9")

    background_color = st.sidebar.selectbox("Color del fondo", ("black","blue"), index=0)
    stroke_width = st.sidebar.selectbox("Ancho del trazo", (20,30,40), index=1)

    current_directory = os.path.dirname(__file__)
    image_path = os.path.join(current_directory, 'media', 'modelos2.png')
    imagen = Image.open(image_path)
    st.sidebar.image(imagen, caption='', use_column_width=True)
    model = st.sidebar.selectbox("Modelo a Utilizar", ("model_retrained.h5","model_mnist.h5","model_mix.h5"), index=0)

    st.sidebar.markdown("### Contactá con el autor del sitio")
    st.sidebar.markdown("[Jeremías Pombo en LinkedIn](https://www.linkedin.com/in/jeremiaspombo/)")
    st.sidebar.markdown("### Visitá el repositorio del proyecto")
    st.sidebar.markdown("[Repositorio de GitHub](https://github.com/Jeremias44/digiartvisionsql)")

    loaded_model = load_model(model)

    canvas = st_canvas(
        fill_color="black",  # Color de relleno de las formas
        stroke_width=stroke_width,  # Ancho del trazo
        stroke_color="white",  # Color del trazo
        background_color=background_color,  # Color de fondo del canvas
        width=280,  # Ancho del lienzo
        height=280,  # Alto del lienzo
        drawing_mode="freedraw",  # Modo de dibujo
        key="canvas",
    )

    if st.checkbox('Iniciar Predicciones'):
        image = Image.fromarray(canvas.image_data.astype('uint8'))
        st.write('Dibujá y borrá las veces que quieras')
        scaled_image = image.resize((28, 28))
        scaled_image = scaled_image.convert('L')
        scaled_image = np.array(scaled_image)
        input_image = scaled_image.reshape(1, 28, 28, 1)
        input_image = input_image / 255.0 
        predicted_number = loaded_model.predict(input_image)
        predicted_number = np.argmax(predicted_number)
        st.image(scaled_image, caption='Imagen procesada', width=140)
        st.subheader(f'Valor detectado: {predicted_number}')
        input_image_flat = input_image.reshape(-1)
        input_image_flat = input_image_flat.astype(np.float32)

        etiqueta = st.number_input("Verificá que la etiqueta sea la correcta antes de guardarla. En caso de que sea incorrecta, por favor corregila (0,9):", 0, 9, predicted_number)
        vector = input_image_flat.tolist()

    if st.button('Guardar Etiqueta'):
        st.write('Gracias por ayudar a reentrenar el modelo')
        st.write("## ¡Excelente trabajo! 🏅")
        st.write('Si hacés click en la papelera podés dibujar nuevamente y seguir entrenando el modelo 😃')
        save_data(vector, etiqueta)
    
if seleccion == "Ver Dibujos":
    query = "SELECT * FROM datos"
    df = pd.read_sql_query(query, conn)

    st.sidebar.markdown(f"### En esta página podrás contribuir eliminando aquellos dibujos que considerás que fueron cargados con la etiqueta incorrecta, que fueron subidos sin información, o aquellos en los que el ojo humano no puede precisar exactamente de qué número se trata")
    etiqueta = st.sidebar.number_input('Para comenzar, ¿Cuál es el número que deseás verificar?:', 0, 9)
    limite = st.sidebar.selectbox("¿Cuántos dibujos deseás verificar?", (10,50,500,'todos'), index=3)
    st.sidebar.markdown(f"### Cantidad de Dibujos Registrados en la Base de Datos: {len(df)}")
    next_power_of_10 = 10 ** len(str(len(df)))
    st.sidebar.markdown(f"## Gracias por contribuir a la limpieza y carga adecuada de los datos! 🏅  Lleguemos a los {next_power_of_10} Registros! 😃")


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
            with st.beta_expander("¿Está seguro de que desea eliminar el dibujo? Este proceso no tiene retorno"):
                confirm = st.button("Confirmar eliminación")
                if confirm:
                    cursor = conn.cursor()
                    id_to_delete = row['id']
                    delete_query = "DELETE FROM datos WHERE id = %s"
                    cursor.execute(delete_query, (id_to_delete,))
                    conn.commit()
                    cursor.close()
                    st.write(f"Dibujo {row['id']} eliminado de la base de datos.")
                else:
                    st.write("Eliminación cancelada")
