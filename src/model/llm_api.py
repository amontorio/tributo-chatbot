from functools import lru_cache
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
import os
import streamlit as st
import mimetypes
import base64
import httpx

# -*- coding: utf-8 -*-
"""
LLM API for Azure and GCP
This module provides functions to interact with LLMs hosted on Azure and GCP.
It includes functions to invoke basic and vision chains, upload files to Azure Blob Storage and GCP, and handle streaming responses.
"""

# azure_endpoint = str(os.getenv("AZURE_ENDPOINT", "")).rstrip("/")
api_key = os.getenv("GOOGLE_API_KEY", "")

print(api_key)

@st.cache_data(show_spinner=True, ttl=3600)
def pdf_file_to_base64_string(filepath):
    with open(filepath, "rb") as f:
        pdf_bytes = f.read()
        base64_bytes = base64.b64encode(pdf_bytes)
        base64_string = base64_bytes.decode("utf-8")
        return base64_string

pdf_data = pdf_file_to_base64_string("src/docs/ICIO - DocumentoNormativa789.pdf")

@lru_cache()
def get_llm():
    LLM_MODEL = "gemini-2.5-flash-preview-05-20"
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0,
        max_tokens=None,
        max_retries=2,
        http_client=httpx.Client(verify=False)
    )
    return llm

llm = get_llm()

# Chain para clasificación tributaria cacheada
@st.cache_resource(show_spinner=False)
def get_tax_classification_chain(): 
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Debes responder únicamente con 'sí' o 'no'. ¿La siguiente pregunta trata sobre información tributaria en algún sentido?"),
        ("human", "{input}")
    ])
    return prompt | llm | StrOutputParser()

# Función para determinar si la pregunta tiene que ver con información tributaria usando un LLM
@st.cache_data(show_spinner=False, ttl=600)
def is_tax_related_question(question: str) -> bool:
    chain = get_tax_classification_chain()
    result = chain.invoke({"input": question}).strip().lower()
    return result.startswith("sí")

# Cache para chains con y sin archivo
@st.cache_resource(show_spinner=False)
def get_cached_chains():

    base_messages = [
        ("system", """
        Eres un asistente experto en ordenanzas fiscales municipales, especializado en el Impuesto sobre Construcciones, Instalaciones y Obras (ICIO).
        Tu tarea es responder de forma clara, precisa y conforme a la normativa vigente, las preguntas que te realicen los ciudadanos o técnicos municipales sin hacer mención al documento adjunto.
        
        Intenta ceñirte a responder solo pregunta del usuario. Sólo da mucho detalle y remite al artículo correspondiente de la ordenanza fiscal o legislación aplicable si te preguntan explícitamente. 
        Si una pregunta no se puede responder sin conocer detalles específicos de la ordenanza local, indícalo claramente y sugiere consultar directamente con el Ayuntamiento correspondiente.
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]

    # Chain sin archivo
    prompt_sin_file = ChatPromptTemplate.from_messages(base_messages)
    chain_sin_file = prompt_sin_file | llm | StrOutputParser()

    # Chain con archivo
    messages_con_file = base_messages + [
        (
            "human",
            [
                {
                    "type": "file",
                    "source_type": "base64",
                    "filename": "ICIO - DocumentoNormativa789.pdf",
                    "data": pdf_data,
                    "mime_type": "application/pdf",
                }
            ],
        )
    ]
    prompt_con_file = ChatPromptTemplate.from_messages(messages=messages_con_file)
    chain_con_file = prompt_con_file | llm | StrOutputParser()

    return {"with_file": chain_con_file, "without_file": chain_sin_file}

def invoke_basic_chain(input_text, chat_history, streaming=True):
    include_file = is_tax_related_question(input_text)
    chains = get_cached_chains()
    chain = chains["with_file"] if include_file else chains["without_file"]

    if include_file:
        with st.spinner("Analizando documentación..."):
            if streaming:
                res = ""
                for chunk in chain.stream({"input": input_text, "chat_history": chat_history}):
                    res += chunk
                    yield chunk
                invoke_basic_chain.response = res
            else:
                res = chain.invoke({"input": input_text, "chat_history": chat_history})
                return res
    else:
        if streaming:
            res = ""
            for chunk in chain.stream({"input": input_text, "chat_history": chat_history}):
                res += chunk
                yield chunk
            invoke_basic_chain.response = res
        else:
            res = chain.invoke({"input": input_text, "chat_history": chat_history})
            return res
