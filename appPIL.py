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
import json

# Se obtiene la URL de la base de datos desde las variables de entorno
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
# Se establece una conexión a la base de datos
conn = psycopg2.connect(DATABASE_URL)

def save_data(vector, etiqueta):
    # Se abre un cursor para interactuar con la base de datos
    cursor = conn.cursor()
    # Se define la consulta de actualización SQL para asignar la calificación
    insert_query = "INSERT INTO datos (vector, etiqueta) VALUES (ARRAY[%s], %s)"
    # Se crea una tupla con los datos a actualizar en la consulta
    data = (vector, etiqueta)
    # Se ejecuta la consulta de actualización con los datos proporcionados
    cursor.execute(insert_query, data)
    # Se confirma los cambios en la base de datos
    conn.commit()
    # Se cierra el cursor
    cursor.close()

st.sidebar.title("Navegación")
seleccion = st.sidebar.selectbox("Ir a la página:", ("Inicio", "Ver Dibujos"))

if seleccion == "Inicio":
    st.title('Modelo para reconocer números del 0 al 9 📚🚀💡👨‍💻')
    st.subheader('Este modelo se encuentra en proceso de entrenamiento 🏋️‍♂️ Podés jugar las veces que quieras y estarás ayudando a entrenarlo! 💪')
    st.write("## Para comenzar dibujá un número del 0 al 9")

    background_color = st.sidebar.selectbox("Color del fondo", ("black","blue"), index=0)
    stroke_width = st.sidebar.selectbox("Ancho del trazo", (20,30,40), index=1)
    # Se obtiene la ruta absoluta del directorio actual y luego de la imagen
    current_directory = os.path.dirname(__file__)
    image_path = os.path.join(current_directory, 'media', 'modelos2.png')
    imagen = Image.open(image_path)
    st.sidebar.image(imagen, caption='', use_column_width=True)
    model = st.sidebar.selectbox("Modelo a Utilizar", ("model_retrained.h5","model_mnist.h5","model_mix.h5"), index=0)

    st.sidebar.markdown("### Contactá con el autor del sitio")
    st.sidebar.markdown("[Jeremías Pombo en LinkedIn](https://www.linkedin.com/in/jeremiaspombo/)")
    st.sidebar.markdown("### Visitá el repositorio del proyecto")
    st.sidebar.markdown("[Repositorio de GitHub](https://github.com/Jeremias44/digiartvisionsql)")
    # Se carga el modelo desde el archivo .h5
    loaded_model = load_model(model)

    # Se crea un lienzo en blanco
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
        # Se obtiene la imagen dibujada en el lienzo
        image = Image.fromarray(canvas.image_data.astype('uint8'))
        st.write('Dibujá y borrá las veces que quieras')
        # Se escala la imagen a 28x28 píxeles
        scaled_image = image.resize((28, 28))
        # Se convierte la imagen a escala de grises
        scaled_image = scaled_image.convert('L')
        # Se asegura que los valores de los píxeles estén en el rango [0, 255]
        scaled_image = np.array(scaled_image)
        # Se agrega una dimensión de lote y cambia el formato de la imagen
        input_image = scaled_image.reshape(1, 28, 28, 1)  # 1 es el tamaño del lote, 28x28x1 es la forma de entrada
        # Se normalizan los valores al rango [0, 1]
        input_image = input_image / 255.0 
        # Se realiza la predicción con el modelo
        predicted_number = loaded_model.predict(input_image)
        # La predicción es un arreglo de probabilidades, se puede obtener el número predicho tomando el índice con mayor probabilidad
        predicted_number = np.argmax(predicted_number)
        # Se visualiza la imagen procesada
        st.image(scaled_image, caption='Imagen procesada', width=140)
        # Ahora, `predicted_number` contiene el número predicho por el modelo
        st.subheader(f'Valor detectado: {predicted_number}')
        # Se aplana la imagen a un arreglo unidimensional
        input_image_flat = input_image.reshape(-1)
        # Se convierte el arreglo a float32
        input_image_flat = input_image_flat.astype(np.float32)

        # Se crea la variable label, es un integer con la etiqueta predicha
        etiqueta = st.number_input("Verificá que la etiqueta sea la correcta antes de guardarla. En caso de que sea incorrecta, por favor corregila (0,9):", 0, 9, predicted_number)
        # Se crea la variable vector, que tiene el arreglo de vectores dentro de una lista    
        vector = input_image_flat.tolist()


    if st.button('Guardar Etiqueta'):
        st.write('Gracias por ayudar a reentrenar el modelo')
        st.write("## ¡Excelente trabajo! 🏅")
        st.write('Si hacés click en la papelera podés dibujar nuevamente y seguir entrenando el modelo 😃')
        save_data(vector, etiqueta)
    
if seleccion == "Ver Dibujos":
    conn = psycopg2.connect(DATABASE_URL)
    # Definir la consulta SQL para seleccionar todos los datos de la tabla "datos"
    query = "SELECT * FROM datos"
    # Utilizar pandas para ejecutar la consulta y cargar los resultados en un DataFrame
    df = pd.read_sql_query(query, conn)
    st.sidebar.markdown(f"### Cantidad de Dibujos Registrados en la Base de Datos: {len(df)}")
    next_power_of_10 = 10 ** len(str(len(df)))
    st.sidebar.markdown(f"# Lleguemos a los {next_power_of_10} Registros!")

    st.title("Dibujos almacenados en la base de datos")
    # Recorrer el DataFrame y mostrar los dibujos
    etiqueta = st.number_input('¿Qué valores de etiqueta deseás verificar?:', 0, 9)
    for index, row in df.iterrows():
        if row['etiqueta'] == etiqueta:
            st.write(f"Dibujo {index + 1}:")
            
            # Obtener el arreglo de vectores de la columna "vector"
            vector = np.array(row['vector'])
            
            # Deshacer la normalización (multiplicar por 255)
            vector = vector * 255
            
            # Cambiar la forma del arreglo a 28x28 píxeles
            vector = vector.reshape(28, 28).astype('uint8')
            
            # Crear una imagen a partir del arreglo
            image = Image.fromarray(vector)
            
            # Mostrar la imagen en Streamlit
            st.image(image, caption=f"Etiqueta: {row['etiqueta']}", width=140)

            # Agregar una opción para eliminar el dibujo
            if st.button(f"Eliminar Dibujo {index + 1}"):
                # Conectarse nuevamente a la base de datos
                conn = psycopg2.connect(DATABASE_URL)
                cursor = conn.cursor()

                # Convierte el vector en una cadena JSON
                vector_str = ",".join(map(str, vector))
                # Define la consulta SQL para eliminar el registro por el vector
                delete_query = "DELETE FROM datos WHERE vector = ARRAY[%s]"

                # Ejecuta la consulta SQL con la cadena del vector
                cursor.execute(delete_query, (vector_str,))

                # Confirma los cambios en la base de datos
                conn.commit()

                # Cerrar el cursor y la conexión
                cursor.close()
                
                st.write(f"Dibujo {index + 1} eliminado de la base de datos.")

                # Cierra la conexión a la base de datos cuando hayas terminado
                conn.close()