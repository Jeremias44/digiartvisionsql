# DigiArtVision

# Modelo para reconocer dígitos 📚🚀💡👨‍💻

DigiArtVision es una aplicación de reconocimiento de números del 0 al 9 que utiliza distintos modelos en proceso de entrenamiento continuo. Los usuarios pueden dibujar números en un lienzo y contribuir al entrenamiento de los modelos mediante feedback, además de comparar los resultados de los distintos modelos.

# Estructura del Repositorio

## Modelos

En la carpeta raíz de este repositorio, encontrarás los tres modelos en formato .h5, disponibles para su uso y comparación:

- **model_retrained.h5**: Este modelo se entrena continuamente a medida que los usuarios utilizan la aplicación. Utiliza únicamente los dibujos realizados en la app como datos de entrenamiento.

- **model_mnist.h5**: Es un modelo preentrenado que no se vuelve a entrenar con nuevos datos. Utiliza el conjunto de datos `MNIST` para su entrenamiento inicial.

- **model_mix.h5**: Este modelo utiliza los pesos del modelo mnist preentrenado y en las últimas capas de la red neuronal se reentrena con los nuevos datos obtenidos desde la aplicación. Combina el conocimiento del modelo preentrenado con los nuevos datos.

## Aplicación Streamlit

La aplicación `appPIL.py` está desplegada desde un repositorio en `GitHub` a través de `Streamlit` y se encuentra configurada con las variables de entorno necesarias. El código de la aplicación establece la conexión con la base de datos `PostgreSQL`, donde se guardan los registros, que incluyen los vectores de los dibujos y las etiquetas correspondientes. La base de datos `PostgresSQL` está desplegada en `Render`, lo que significa que está disponible las 24 horas del día para acceder a los datos almacenados y cargar nuevos registros.

### Importación de librerías

```python
import streamlit as st
from tensorflow import keras
from keras.models import load_model
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
import psycopg2
import os
from dotenv import load_dotenv
```

Este bloque de código se encarga de importar las bibliotecas y módulos necesarios para el funcionamiento de la aplicación. Aquí se describen las principales funciones de cada uno:

- `streamlit`: Es el marco de trabajo utilizado para crear aplicaciones web interactivas en Python.

- `keras` y `load_model`: Importa la biblioteca Keras, que se utiliza para cargar modelos de redes neuronales preentrenados y realizar predicciones.

- `PIL` (Pillow): Se utiliza para trabajar con imágenes y, en este caso, para procesar las imágenes dibujadas por los usuarios.

- `numpy`: Proporciona soporte para trabajar con matrices y arreglos multidimensionales, es esencial para el procesamiento de imágenes y datos numéricos.

- `streamlit_drawable_canvas`: Es una extensión de Streamlit que permite a los usuarios dibujar en un lienzo interactivo.

- `psycopg2`: Se utiliza para conectarse a la base de datos PostgreSQL y realizar operaciones de base de datos.

- `os`: Proporciona funcionalidades para interactuar con el sistema operativo, en este caso, se utiliza para cargar variables de entorno.

- `dotenv`: Permite cargar las variables de entorno desde un archivo `.env`.

### Conexión a la Base de Datos PostgreSQL

```python
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
conn = psycopg2.connect(DATABASE_URL)

def save_data(vector, label):
    cursor = conn.cursor()
    insert_query = "INSERT INTO datos (vector, etiqueta) VALUES (ARRAY[%s], %s)"
    data = (vector, label)
    cursor.execute(insert_query, data)
    conn.commit()
    cursor.close()
```

En esta sección del código, se establece la conexión con `PostgreSQL`, y se define una función para guardar datos en ella. A continuación, se detallan las acciones realizadas:

- **Obtención de la URL de la Base de Datos**: 
  - `load_dotenv()`: Esta función se utiliza para cargar variables de entorno desde un archivo `.env`. En este caso, se espera que en el archivo `.env` se defina una variable llamada "DATABASE_URL", que contiene la URL de la base de datos PostgreSQL.
  - `DATABASE_URL = os.getenv("DATABASE_URL")`: Aquí se obtiene la URL de la base de datos desde las variables de entorno y se almacena en la variable "DATABASE_URL". Esta URL suele contener información necesaria para la conexión, como el nombre de usuario, la contraseña, el host y el nombre de la base de datos.

- **Establecimiento de la Conexión a la Base de Datos**:
  - `conn = psycopg2.connect(DATABASE_URL)`: Se utiliza la biblioteca `psycopg2` para establecer una conexión a la base de datos PostgreSQL utilizando la URL obtenida anteriormente. Esto crea una conexión que se utilizará para interactuar con la base de datos.

- **Función para Guardar Datos**:
  - `def save_data(vector, label)`: Se define una función llamada "save_data" que toma dos argumentos, "vector" y "label".
  - `cursor = conn.cursor()`: Aquí se crea un cursor que se utilizará para ejecutar consultas en la base de datos. Un cursor es una especie de "puntero" que permite interactuar con la base de datos.
  - `insert_query = "INSERT INTO datos (vector, etiqueta) VALUES (ARRAY[%s], %s)"`: Se define una consulta SQL para insertar datos en la tabla "datos" de la base de datos. Los valores "vector" y "etiqueta" se insertan como parámetros en la consulta para evitar problemas de seguridad y prevenir inyecciones SQL.
  - `data = (vector, label)`: Se crea una tupla llamada "data" que contiene los valores de "vector" y "label" que se desean insertar en la base de datos.
  - `cursor.execute(insert_query, data)`: Aquí se ejecuta la consulta de inserción con los datos proporcionados en la tupla "data".
  - `conn.commit()`: Se confirman los cambios en la base de datos, lo que significa que los datos se guardan de manera permanente.
  - `cursor.close()`: Finalmente, se cierra el cursor, liberando los recursos utilizados para la consulta.

Esta sección del código es fundamental para guardar los datos generados por los usuarios en la base de datos PostgreSQL, lo que permite entrenar y mejorar el modelo de reconocimiento de números.

### Interfaz de Usuario y Configuración del Modelo

```python
st.title('Modelo para reconocer números del 0 al 9 📚🚀💡👨‍💻')
st.subheader('Este modelo se encuentra en proceso de entrenamiento 🏋️‍♂️ Podés jugar las veces que quieras y estarás ayudando a entrenarlo! 💪')
st.write("## Para comenzar dibujá en el lienzo un número del 0 al 9")

st.sidebar.title("Opciones de Dibujo")
background_color = st.sidebar.selectbox("Color del fondo", ("black","blue"), index=1)
stroke_width = st.sidebar.selectbox("Ancho del trazo", (20,30,40), index=1)
st.sidebar.title("Modelo a Utilizar")
st.sidebar.markdown('model_retrained.h5 es un modelo que se entrena únicamente con los dibujos realizados por los usuarios de esta app')
st.sidebar.markdown('model_mnist.h5 es un modelo preentrenado con un dataset muy utilizado llamado MNIST')
st.sidebar.markdown('model_mix.h5 se entrena con los datos provenientes de ambas fuentes')
model = st.sidebar.selectbox("Modelo", ("model_retrained.h5","model_mnist.h5","model_mix.h5"), index=0)

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
```

Esta sección del código se encarga de la interfaz de usuario y la configuración del modelo. A continuación, se detallan las acciones realizadas:

- **Título y Subtítulo**:
  - `st.title('Modelo para reconocer números del 0 al 9 📚🚀💡👨‍💻')`: Se establece un título principal para la aplicación web, que indica el propósito del modelo. El título incluye emojis para darle un toque visual y amigable.
  - `st.subheader('Este modelo se encuentra en proceso de entrenamiento 🏋️‍♂️ Podés jugar las veces que quieras y estarás ayudando a entrenarlo! 💪')`: Se agrega un subtítulo que describe el estado de entrenamiento del modelo y anima a los usuarios a participar en su mejora.

- **Instrucciones Iniciales**:
  - `st.write("## Para comenzar dibujá en el lienzo un número del 0 al 9")`: Se proporciona una instrucción inicial clara para los usuarios, invitándolos a comenzar a dibujar un número del 0 al 9 en el lienzo.

- **Configuración de la Barra Lateral**:
  - `st.sidebar.title("Opciones de Dibujo")`: Se establece un título en la barra lateral que indica las opciones relacionadas con el dibujo.
  - `background_color = st.sidebar.selectbox("Color del fondo", ("black","blue"), index=1)`: Los usuarios pueden seleccionar el color de fondo para el lienzo a través de un cuadro de selección en la barra lateral.
  - `stroke_width = st.sidebar.selectbox("Ancho del trazo", (20,30,40), index=1)`: Se permite a los usuarios elegir el ancho del trazo que desean utilizar para dibujar.
  - `st.sidebar.title("Modelo a Utilizar")`: Se agrega un título en la barra lateral que indica las opciones relacionadas con el modelo a utilizar.
  - Se proporciona información sobre tres modelos disponibles y la opción de seleccionar uno de ellos.

- **Carga del Modelo**:
  - `model = st.sidebar.selectbox("Modelo", ("model_retrained.h5","model_mnist.h5","model_mix.h5"), index=0)`: Se permite a los usuarios elegir el modelo que desean utilizar desde una lista desplegable en la barra lateral.
  - `loaded_model = load_model(model)`: El modelo seleccionado se carga desde un archivo .h5 para su uso en la aplicación.

- **Creación del Lienzo**:
  - `canvas = st_canvas(...)`: Se crea un lienzo en blanco en la interfaz de usuario que permitirá a los usuarios dibujar el número que deseen. Se especifican propiedades como el color de relleno, el ancho del trazo y el tamaño del lienzo.

### Proceso de Predicción y Etiquetado

```python
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

    label = st.number_input("Verificá que la etiqueta sea la correcta antes de guardarla. En caso de que sea incorrecta, por favor corregila (0,9):", 0, 9, predicted_number)     
    vector = input_image_flat.tolist()

if st.button('Guardar Etiqueta'):
    st.write('Gracias por ayudar a reentrenar el modelo')
    st.write("## ¡Excelente trabajo! 🏅")
    st.write('Si hacés click en la papelera podés dibujar nuevamente un número y seguir entrenando el modelo 😃')
    save_data(vector, label)
```

Esta sección del código se encarga de la predicción de números dibujados por el usuario en el lienzo y permite al usuario etiquetar el número antes de guardarlo. A continuación, se detallan las acciones realizadas:

- **Checkbox para Iniciar Predicciones**:
  - `if st.checkbox('Iniciar Predicciones'):`: Se utiliza un cuadro de verificación (checkbox) que permite al usuario activar el proceso de predicción una vez que ha dibujado un número en el lienzo.

- **Obtención de la Imagen Dibujada**:
  - `image = Image.fromarray(canvas.image_data.astype('uint8'))`: Se convierte la información de la imagen dibujada en el lienzo en un objeto de imagen. La imagen se representa en formato uint8.
  - `st.write('Dibujá y borrá las veces que quieras')`: Se muestra un mensaje indicando al usuario que puede seguir dibujando y borrando en el lienzo.

- **Preprocesamiento de la Imagen**:
  - `scaled_image = image.resize((28, 28))`: La imagen se escala para tener dimensiones de 28x28 píxeles, que es la entrada esperada por el modelo.
  - `scaled_image = scaled_image.convert('L')`: La imagen se convierte a escala de grises para que tenga una sola capa de píxeles en lugar de múltiples canales de color.
  - `scaled_image = np.array(scaled_image)`: La imagen se convierte en un arreglo NumPy para su procesamiento.
  - `input_image = scaled_image.reshape(1, 28, 28, 1)`: Se ajusta la forma de la imagen para que sea compatible con el modelo, agregando una dimensión de lote y configurando las dimensiones a 28x28x1.

- **Predicción con el Modelo**:
  - `predicted_number = loaded_model.predict(input_image)`: Se realiza la predicción del número dibujado utilizando el modelo cargado. La salida es un arreglo de probabilidades.
  - `predicted_number = np.argmax(predicted_number)`: Se determina el número predicho tomando el índice del valor más alto en el arreglo de probabilidades.

- **Visualización de la Imagen y Resultado**:
  - `st.image(scaled_image, caption='Imagen procesada', width=140)`: Se muestra la imagen procesada en la interfaz de usuario con una leyenda.
  - `st.subheader(f'Valor detectado: {predicted_number}')`: Se muestra el número predicho en la interfaz de usuario como resultado.

- **Etiquetado del Número y Preparación de Datos para Guardar**:
  - `input_image_flat = input_image.reshape(-1)`: La imagen procesada se aplana a un arreglo unidimensional.
  - `input_image_flat = input_image_flat.astype(np.float32)`: El arreglo se convierte a tipo de dato float32 para su almacenamiento.
  - `label = st.number_input("Verificá que la etiqueta sea la correcta antes de guardarla. En caso de que sea incorrecta, por favor corrégila (0,9):", 0, 9, predicted_number)`: Se permite al usuario verificar y, si es necesario, corregir la etiqueta del número dibujado. La etiqueta se almacena en la variable `label`, y la representación de la imagen en `input_image_flat`.

- **Botón para Guardar Etiqueta**:
  - `if st.button('Guardar Etiqueta'):`: Se presenta un botón que permite al usuario guardar la etiqueta asociada al número dibujado.

- **Mensaje de Agradecimiento**:
  - Después de guardar la etiqueta, se muestra un mensaje agradeciendo al usuario por ayudar a reentrenar el modelo y se anima al usuario a seguir dibujando.

## Entorno Virtual

Para gestionar las dependencias de la aplicación y garantizar la reproducibilidad, se utiliza un entorno virtual llamado `venvSQL`. En el archivo `requirements.txt`, encontrarás las librerías y paquetes necesarios para ejecutar la aplicación y conectar con la base de datos.

## Proceso de Reentrenamiento

El modelo se reentrena ejecutando el archivo `retrain.ipynb`. Este archivo accede a la base de datos PostgreSQL, realiza un preprocesamiento de los datos almacenados y, posteriormente, pasa estos datos a los modelos correspondientes para llevar a cabo el proceso de reentrenamiento. Al final del archivo `retrain.ipynb`, encontrarás un código que evalúa los tres modelos en distintas métricas para medir su rendimiento.

```python
import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

- **Importación de Librerías**:
  - `import psycopg2`: Se importa la librería `psycopg2`, que se utiliza para interactuar con la base de datos PostgreSQL y recuperar los datos para el reentrenamiento.
  - `import pandas as pd`: Se importa la librería `pandas` para el manejo de datos, lo que facilita la manipulación de los datos recuperados de la base de datos.
  - `from dotenv import load_dotenv`: Se importa la función `load_dotenv` de la librería `dotenv`, que se utiliza para cargar las variables de entorno que contienen información sensible, como las credenciales de la base de datos.
  - `import os`: Se importa la librería `os` para trabajar con rutas de archivos y cargar las variables de entorno.
  - `import numpy as np`: Se importa la librería `numpy` con el alias `np` para operaciones matemáticas y manipulación de arreglos numéricos.
  - `import tensorflow as tf`: Se importa la librería `tensorflow` con el alias `tf` para el entrenamiento y reentrenamiento del modelo de reconocimiento.
  - `from tensorflow import keras`: Se importa la sublibrería `keras` de `tensorflow` para trabajar con modelos de redes neuronales.

Este conjunto de librerías proporciona las herramientas necesarias para cargar datos desde la base de datos PostgreSQL, realizar el preprocesamiento de datos, y ejecutar el proceso de reentrenamiento del modelo. A medida que avances en el código, estas librerías se utilizarán para llevar a cabo estas tareas de manera eficiente.

La importación de estas librerías establece las bases para el reentrenamiento del modelo en función de los nuevos datos proporcionados por los usuarios de la aplicación. Más adelante en el código, se realizarán tareas como la extracción de datos desde la base de datos, el preprocesamiento de imágenes y el ajuste del modelo para mejorar su capacidad de reconocimiento de números.

### Carga de Datos y Preprocesamiento

```python
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
conn = psycopg2.connect(DATABASE_URL)
query = "SELECT * FROM datos"
df = pd.read_sql_query(query, conn)
conn.close()

def parse_vector(vector_list):
    return np.array(vector_list).reshape(28, 28).astype(np.float32)

df['VectorNum'] = df['vector'].apply(parse_vector)
train_vectors = df['VectorNum'].to_list()
train_vectors = np.array(train_vectors)
labels = df['etiqueta'].tolist()
labels = np.array(labels)
```

En esta parte del código, se realiza la carga de datos desde `PostgreSQL` y el preprocesamiento necesario para entrenar o reentrenar el modelo de reconocimiento de números. A continuación, se explican los pasos realizados:

- **Cargar Variables de Entorno**: Primero, se cargan las variables de entorno desde un archivo `.env` utilizando `load_dotenv()`. Esto es comúnmente utilizado para almacenar información sensible, como las credenciales de la base de datos.

- **Obtener la URL de la Base de Datos**: Se obtiene la URL de la base de datos PostgreSQL desde las variables de entorno mediante `os.getenv("DATABASE_URL")`. Esto proporciona la ubicación y las credenciales necesarias para conectarse a la base de datos.

- **Establecer Conexión a la Base de Datos**: Se establece una conexión a la base de datos utilizando `psycopg2.connect(DATABASE_URL)`. Esto permite acceder a la base de datos y recuperar los datos almacenados.

- **Consultar y Cargar Datos en un DataFrame**: Se define una consulta SQL para seleccionar todos los datos de la tabla "datos" y se ejecuta utilizando Pandas. Los resultados se almacenan en un DataFrame llamado `df`, lo que facilita el manejo y procesamiento de los datos.

- **Cierre de la Conexión a la Base de Datos**: Una vez que los datos se han cargado en el DataFrame, se cierra la conexión a la base de datos con `conn.close()` para liberar recursos.

- **Transformación de Datos**: Luego, se define una función llamada `parse_vector` que se aplica a la columna "vector" del DataFrame. Esta función convierte la lista anidada en un arreglo NumPy y redimensiona los datos a una forma de (28, 28) para que coincidan con el formato de las imágenes.

- **Creación de Arreglos NumPy para Entrenamiento**: Se crea un arreglo NumPy llamado `train_vectors` que contiene los datos preprocesados de las imágenes en un formato adecuado para el entrenamiento del modelo. Además, se crea un arreglo `labels` que almacena las etiquetas asociadas a las imágenes.

Este conjunto de pasos permite cargar los datos desde la base de datos, preprocesar las imágenes y etiquetas, y prepararlos para ser utilizados en el proceso de entrenamiento o reentrenamiento del modelo de reconocimiento de números. Estos datos son esenciales para mejorar y adaptar el modelo a nuevas muestras recopiladas desde la aplicación.

### Creación y Compilación del Modelo de Red Neuronal

Esta parte del código tiene como objetivo la creación y entrenamiento del modelo de red neuronal. Utiliza la biblioteca TensorFlow y Keras para llevar a cabo estas tareas. A continuación, se describen en detalle cada parte del código:

### Creación del Modelo

```python
retrained_model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])
```

- **keras.Sequential**: Se crea un modelo secuencial que es una secuencia lineal de capas de red neuronal.

- **layers.Flatten**: La primera capa, Flatten, transforma la imagen 2D (28x28 píxeles) en un arreglo unidimensional de 784 elementos. Esto es necesario para aplanar la imagen y proporcionarla como entrada a la red neuronal.

- **layers.Dense(128, activation='relu')**: Esta es una capa oculta con 128 neuronas. La función de activación 'relu' (unidad lineal rectificada) se utiliza para introducir no linealidad en la red.

- **layers.Dropout(0.2)**: Se aplica un regularizador llamado "Dropout" con una tasa del 20%. Esto ayuda a prevenir el sobreajuste al desactivar aleatoriamente un 20% de las neuronas durante cada iteración de entrenamiento.

- **layers.Dense(10, activation='softmax')**: La capa de salida tiene 10 neuronas, correspondientes a las 10 clases posibles (números del 0 al 9). La función de activación 'softmax' se utiliza para obtener probabilidades de pertenencia a cada clase.


```python
retrained_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

- **optimizer='adam'**: Se utiliza el optimizador Adam, que es un algoritmo de optimización que ajusta automáticamente la tasa de aprendizaje durante el entrenamiento. Es ampliamente utilizado en tareas de entrenamiento de redes neuronales.

- **loss='sparse_categorical_crossentropy'**: La función de pérdida se establece en 'sparse_categorical_crossentropy'. Esta función de pérdida es comúnmente utilizada en problemas de clasificación multiclase.

- **metrics=['accuracy']**: La métrica de evaluación se establece en 'accuracy' para realizar un seguimiento de la precisión del modelo durante el entrenamiento.

```python
retrained_model.fit(np.array(train_vectors), labels, epochs=6)
retrained_model.save("model_retrained.h5")
```

- **np.array(train_vectors)**: Los datos de entrenamiento se proporcionan como un arreglo NumPy que contiene las imágenes aplanadas.

- **labels**: Las etiquetas correspondientes a las imágenes de entrenamiento.

- **epochs=6**: El modelo se entrena durante 6 épocas, lo que significa que pasa por todo el conjunto de datos de entrenamiento 6 veces. Durante el entrenamiento, el modelo ajusta sus pesos y bias para minimizar la función de pérdida.

- **retrained_model.save("model_retrained.h5")**: # Una vez que se haya terminado el entrenamiento, se guarda el modelo

El archivo `retrain.ipynb` continúa de manera similar con el entrenamiento de los restantes modelos `mix_model.h5` y `mnist_model.h5`. Luego pasa a la sección de Comparativa y Evaluación de Modelos.





## Carpeta Media

Por último, en la carpeta `/media` se encuentran imágenes y capturas de pantalla de la aplicación en funcionamiento, con ejemplos de dibujos y etiquetas de predicciones


***
## Proyecto DigiArtVisionSQL

* [Jeremías Pombo en LinkedIn](https://www.linkedin.com/in/jeremiaspombo/)

* [Repositorio de GitHub](https://github.com/Jeremias44/digiartvisionsql)

* [Acceder a la Aplicación en Streamlit](https://digiartvisionsql.streamlit.app/)

## Contacto
Si tienes alguna pregunta o sugerencia, no dudes en contactarme a través de mi correo electrónico: jeremiaspombo@outlook.com.

## Código Abierto
Este proyecto es de código abierto, lo que significa que estás invitado a contribuir, realizar mejoras o utilizarlo en tus propios proyectos. ¡Espero que disfrutes de DigiArtVisionSQL!