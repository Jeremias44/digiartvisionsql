import streamlit as st
from tensorflow import keras
from keras.models import load_model
from PIL import Image
import pandas as pd
import numpy as np
from streamlit_drawable_canvas import st_canvas
from pgadmin_connect_render import *

crear_tabla_predicciones()
   
st.title('Modelo para reconocer números del 0 al 9 📚🚀💡👨‍💻')
st.subheader('Este modelo se encuentra en proceso de entrenamiento 🏋️‍♂️ Podés jugar las veces que quieras y estarás ayudando a entrenarlo! 💪')
st.write("## Para comenzar dibujá en el lienzo un número del 0 al 9")

drawing = False
# Crear una barra lateral para opciones
drawing_mode = "freedraw"
background_color="black"
stroke_width=20
st.sidebar.title("Opciones de Dibujo")
drawing_mode = st.sidebar.selectbox("Modo de Dibujo", ("freedraw", "line"), index=0)
background_color = st.sidebar.selectbox("Color del fondo", ("black","blue"), index=0)
stroke_width = st.sidebar.selectbox("Ancho del trazo", (20,30,40), index=1)
st.sidebar.title("Modelo a Utilizar")
st.sidebar.markdown('retrained_model.h5 es un modelo entrenado únicamente con los dibujos realizados por los usuarios de esta app')
st.sidebar.markdown('model.h5 es un modelo preentrenado con un dataset muy usado llamado MNIST')
st.sidebar.markdown('mix_model.h5 utiliza los datos provenientes de ambas fuentes')
model = st.sidebar.selectbox("Modelo", ("Streamlit/retrained_model.h5","Streamlit/model.h5","Streamlit/mix_model.h5"), index=0)

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
    drawing_mode=drawing_mode,  # Modo de dibujo
    key="canvas",
)

if st.checkbox('Iniciar Predicciones'):
    # Obtén la imagen dibujada en el lienzo
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

    label = st.number_input("Verificá que la etiqueta sea la correcta antes de registrarla. En caso de que sea incorrecta, por favor corregila (0,9):", 0, 9, predicted_number)     
    vector = [input_image_flat.tolist()]
    etiqueta = [label]


if st.button("Registrar Etiqueta"):
    st.write('Gracias por ayudar a reentrenar el modelo')
    st.write("## ¡Excelente trabajo! 🏅")
    st.write('Si hacés click en la papelera podés hacer un nuevo dibujo y seguir entrenando al modelo 😃')
    guardar_data(vector, etiqueta)


