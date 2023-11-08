import streamlit as st
from PIL import Image

st.info('# 👁️ ------  DIGI ART VISION ------ 👁️')

st.markdown("## Te damos la bienvenida a DigiArtVision, una app de interacción con modelos de computer vision 🤖")

st.info("Desde el menú lateral se accede a la sección principal donde podrás dibujar ✏️, ver predicciones 🚀💡 y alimentar la base de datos 👨‍💻.")
st.info("También podés acceder a la sección Verificar Dibujos 📝, en la que podrás contribuir 🏅 evaluando los últimos dibujos subidos a la aplicación.")
st.info("Además, inauguramos la nueva sección Evaluar Modelos 🤖, en la que podrás observar la precisión de las predicciones de los modelos, entre otras métricas.")

imagen = Image.open('media/portada.png')
st.image(imagen, caption='', use_column_width=True)

st.sidebar.markdown("### Contactá con el autor del sitio")
st.sidebar.markdown("[Jeremías Pombo en LinkedIn](https://www.linkedin.com/in/jeremiaspombo/)")
st.sidebar.markdown("### Visitá el repositorio del proyecto")
st.sidebar.markdown("[Repositorio de GitHub](https://github.com/Jeremias44/digiartvisionsql)")

