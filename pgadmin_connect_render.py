import os
import streamlit as st
import psycopg2

# Se obtiene la URL de la base de datos desde las variables de entorno
DATABASE_URL = 'postgres://digiartvisionsql_postrgressql_user:QbcjiMJNMV7nxMragK1Dxyyam2SMgvPn@dpg-cl36t99novjs73bcvcjg-a.oregon-postgres.render.com/digiartvisionsql_postrgressql'
#DATABASE_URL = os.getenv(DATABASE_URL)

# Inicializa la conexión a la base de datos
@st.cache_resource
def init_connection():
    # Crea una conexión utilizando la URL de la base de datos
    return psycopg2.connect(DATABASE_URL)

# Establece una conexión a la base de datos
conn = init_connection()

'''
def crear_tabla_datos():
    # Abre un cursor para interactuar con la base de datos
    cursor = conn.cursor()
    # Define la consulta SQL para crear la tabla si no existe
    create_table_query = """
    CREATE TABLE IF NOT EXISTS datos (
        vector REAL[],
        etiqueta INTEGER
    );
    """
    # Ejecuta la consulta para crear la tabla
    cursor.execute(create_table_query)
    # Confirma los cambios en la base de datos
    conn.commit()
    # Cierra el cursor
    cursor.close()
'''

def guardar_data(vector, etiqueta):
    # Abre un cursor para interactuar con la base de datos
    cursor = conn.cursor()
    # Define la consulta de actualización SQL para asignar la calificación
    insert_query = "INSERT INTO datos (vector, etiqueta) VALUES (ARRAY[%s], %s)"
    # Crea una tupla con los datos a actualizar en la consulta
    data = (vector, etiqueta)
    # Ejecuta la consulta de actualización con los datos proporcionados
    cursor.execute(insert_query, data)
    # Confirma los cambios en la base de datos
    conn.commit()
    # Cierra el cursor
    cursor.close()