import streamlit as st
import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

chatbot = st.Page("./pages/chatbot.py", title="Chatbot", icon="🗓️")
pg = st.navigation(
    {
    #    "Información": [welcome, doc],
       #"Chatbot": [assistant],
       #"Panel principal": [intro_page],
       "Agent": [chatbot] 
    }
    )

st.set_page_config(
    page_title="ATC",
    page_icon="🛠️",
    # layout="wide"
)

def get_base64_of_bin_file(bin_file):
    """Devuelve la cadena base64 de un archivo binario."""
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

BASE_DIR = Path(__file__).resolve().parent

logo_atc = "atc-logo-completo-removebg.png"
logo_path = BASE_DIR / "images" / logo_atc

logo_atc_small = "atc-logo-completo-removebg-small.png"
logo_small_path = BASE_DIR / "images" / logo_atc_small

st.logo(logo_path, 
        link = "https://www3.gobiernodecanarias.org/tributos/atc/",
        icon_image = logo_small_path, 
        size = "large")

pg.run()