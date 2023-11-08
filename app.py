import streamlit as st

st.title('DIGI ART VISION')

# Coloca el enlace al video de YouTube entre corchetes y paréntesis
video_url = "https://youtu.be/UIxxBsHQznw"

# Utiliza st.markdown para mostrar el enlace como un hipervínculo
st.markdown(f"[## Presentación]({video_url})")

st.markdown(f"[## Acceder a Página Pincipal]('https://digiartvisionsql.streamlit.app/P%C3%A1gina_Principal')")
st.markdown(f"[## Acceder a Verificar Dibujos]('https://digiartvisionsql.streamlit.app/Verificar_Dibujos')")

st.sidebar.markdown("### Contactá con el autor del sitio")
st.sidebar.markdown("[Jeremías Pombo en LinkedIn](https://www.linkedin.com/in/jeremiaspombo/)")
st.sidebar.markdown("### Visitá el repositorio del proyecto")
st.sidebar.markdown("[Repositorio de GitHub](https://github.com/Jeremias44/digiartvisionsql)")