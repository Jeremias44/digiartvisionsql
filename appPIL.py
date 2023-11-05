import streamlit as st
from tensorflow import keras
from keras.models import load_model
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
import psycopg2
import os
from dotenv import load_dotenv


# Se obtiene la URL de la base de datos desde las variables de entorno
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
# Establece una conexión a la base de datos
conn = psycopg2.connect(DATABASE_URL)

def save_data(vector, label):
    # Abre un cursor para interactuar con la base de datos
    cursor = conn.cursor()
    # Define la consulta de actualización SQL para asignar la calificación
    insert_query = "INSERT INTO datos (vector, etiqueta) VALUES (ARRAY[%s], %s)"
    # Crea una tupla con los datos a actualizar en la consulta
    data = (vector, label)
    # Ejecuta la consulta de actualización con los datos proporcionados
    cursor.execute(insert_query, data)
    # Confirma los cambios en la base de datos
    conn.commit()
    # Cierra el cursor
    cursor.close()

   
st.title('Modelo para reconocer números del 0 al 9 📚🚀💡👨‍💻')
st.subheader('Este modelo se encuentra en proceso de entrenamiento 🏋️‍♂️ Podés jugar las veces que quieras y estarás ayudando a entrenarlo! 💪')
st.write("## Para comenzar dibujá en el lienzo un número del 0 al 9")

st.sidebar.title("Opciones de Dibujo")
background_color = st.sidebar.selectbox("Color del fondo", ("black","blue"), index=0)
stroke_width = st.sidebar.selectbox("Ancho del trazo", (20,30,40), index=1)

# Modelo a Utilizar
st.sidebar.title("Modelos a Utilizar")
st.sidebar.info('model_retrained.h5 es un modelo que se entrena únicamente con los dibujos realizados por los usuarios de esta app')
st.sidebar.info('model_mnist.h5 es un modelo preentrenado con un dataset muy utilizado llamado MNIST')
st.sidebar.info('model_mix.h5 se entrena con los datos provenientes de ambas fuentes')
model = st.sidebar.selectbox("Modelo", ("model_retrained.h5","model_mnist.h5","model_mix.h5"), index=0)

st.sidebar.markdown("### Contacta con el autor del sitio")
st.sidebar.markdown("[Jeremías Pombo en LinkedIn](https://www.linkedin.com/in/jeremiaspombo/)")
st.sidebar.markdown("### Visita el repositorio del proyecto")
st.sidebar.markdown("[Repositorio de GitHub](https://github.com/Jeremias44/digiartvisionsql)")

# Carga el modelo desde el archivo .h5
loaded_model = load_model(model)

# Crea un lienzo en blanco
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
    # Obtiene la imagen dibujada en el lienzo
    image = Image.fromarray(canvas.image_data.astype('uint8'))
    #image = canvas.image_data.astype(np.uint8)
    st.write('Dibujá y borrá las veces que quieras')

    # Escala la imagen a 28x28 píxeles
    scaled_image = image.resize((28, 28))
    # Convierte la imagen a escala de grises
    scaled_image = scaled_image.convert('L')
    # Asegura que los valores de los píxeles estén en el rango [0, 255]
    scaled_image = np.array(scaled_image)
    # Agrega una dimensión de lote y cambia el formato de la imagen
    input_image = scaled_image.reshape(1, 28, 28, 1)  # 1 es el tamaño del lote, 28x28x1 es la forma de entrada
    # Normaliza la imagen si es necesario
    input_image = input_image / 255.0  # Normaliza los valores al rango [0, 1]
    # Realiza la predicción con el modelo
    predicted_number = loaded_model.predict(input_image)
    # La predicción es un arreglo de probabilidades, se puede obtener el número predicho tomando el índice con mayor probabilidad
    predicted_number = np.argmax(predicted_number)
    # Visualiza la imagen procesada
    st.image(scaled_image, caption='Imagen procesada', width=140)
    # Ahora, `predicted_number` contiene el número predicho por el modelo
    st.subheader(f'Valor detectado: {predicted_number}')
    # Aplana la imagen a un arreglo unidimensional
    input_image_flat = input_image.reshape(-1)
    # Convierte el arreglo a float32
    input_image_flat = input_image_flat.astype(np.float32)

    label = st.number_input("Verificá que la etiqueta sea la correcta antes de guardarla. En caso de que sea incorrecta, por favor corregila (0,9):", 0, 9, predicted_number)     
    vector = input_image_flat.tolist()


if st.button('Guardar Etiqueta'):
    st.write('Gracias por ayudar a reentrenar el modelo')
    st.write("## ¡Excelente trabajo! 🏅")
    st.write('Si hacés click en la papelera podés dibujar nuevamente un número y seguir entrenando el modelo 😃')
    save_data(vector, label)