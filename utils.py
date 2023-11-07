import streamlit as st

def save_data(vector, label):
    # Se abre un cursor para interactuar con la base de datos
    cursor = conn.cursor()
    # Se define la consulta de actualización SQL para asignar la calificación
    insert_query = "INSERT INTO datos (vector, etiqueta) VALUES (ARRAY[%s], %s)"
    # Se crea una tupla con los datos a actualizar en la consulta
    data = (vector, label)
    # Se ejecuta la consulta de actualización con los datos proporcionados
    cursor.execute(insert_query, data)
    # Se confirma los cambios en la base de datos
    conn.commit()
    # Se cierra el cursor
    cursor.close()