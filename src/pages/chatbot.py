import streamlit as st
import os, base64
from model.llm_api import invoke_basic_chain

def render_or_update_model_info(model_name):
    """
    Renders or updates the model information on the webpage.
    """
    # Leer y aplicar estilos CSS
    css_path = os.path.join(os.path.dirname(__file__), '..', 'design', 'assistant', 'styles.css')
    with open(css_path, 'r', encoding='utf-8') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

    # Codificar imagen como base64
    image_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'malaga-logo-removebg.png')
    with open(image_path, 'rb') as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode()
    image_data_uri = f"data:image/png;base64,{img_base64}"

    # Renderizar HTML con imagen embebida
    html_path = os.path.join(os.path.dirname(__file__), '..', 'design', 'assistant', 'content.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        html_template = f.read()
    html = html_template.format(logo=image_data_uri, model=model_name)

    st.markdown(html, unsafe_allow_html=True)

def reset_chat_history():
    """Resets the chat history."""
    if "messages" in st.session_state:
        st.session_state.messages = []

# Inicializar configuración del modelo
model_options = ["Gemini-2.5"]

if "model" not in st.session_state:
    st.session_state.model = model_options[0]
    st.session_state.temperature = 0
    st.session_state.messages = []

with st.sidebar:
    st.title("Configuración de modelo")

    st.session_state.model = st.selectbox("Elige un modelo:", model_options, index=0)
    st.session_state.temperature = st.slider('Selecciona una temperatura:', 0.0, 1.0, 0.0, 0.01, format="%.2f")

    if st.button("Clear Chat 🧹", use_container_width=True):
        reset_chat_history()

render_or_update_model_info(st.session_state.model)

# Mostrar historial de mensajes
for role, content in st.session_state.messages:
    if isinstance(content, list):
        for item in content:
            if item["type"] == "file":
                with st.chat_message(role, avatar="📄"):
                    with st.status(item['filename']):
                        st.markdown(f"**Nombre:** {item['filename']}")
                        st.markdown(f"**Tipo:** {item['mime_type']}")
                        st.markdown(f"**Tamaño:** {len(item['data'])} bytes")
    else:
        with st.chat_message(role):
            st.markdown(content)

# Entrada del usuario
prompt = st.chat_input(
    placeholder="¿En qué puedo ayudarte?"
    # accept_file="multiple",
    # file_type=["pdf"]
)

if prompt:
    text_input = prompt

    with st.chat_message("user"):
        st.markdown(text_input)

    # Procesar con LLM (solo texto)
    with st.chat_message("assistant"):
        response = invoke_basic_chain(
            input_text=text_input,
            chat_history=st.session_state.messages,
            streaming=True
        )
        st.write_stream(response)

    # Guardar historial
    st.session_state.messages.append(("user", text_input))
    st.session_state.messages.append(("assistant", invoke_basic_chain.response))
