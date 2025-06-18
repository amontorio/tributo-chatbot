import streamlit as st
import os, base64
from model.llm_api import invoke_basic_chain, invoke_web_chain
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from langchain_tavily import TavilySearch
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
#
from operator import itemgetter

from langchain_openai.chat_models import ChatOpenAI

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from datetime import datetime
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: list[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.
store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]
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
    if "memory" in st.session_state:
        st.session_state.memory.messages = []

# Inicializar configuración del modelo
model_options = ["Gemini-2.5"]

if "model" not in st.session_state:
    st.session_state.model = model_options[0]
    st.session_state.temperature = 0

if "web_mode" not in st.session_state:
    st.session_state.web_mode = False
if "memory" not in st.session_state:
    st.session_state.memory = ChatMessageHistory()
with st.sidebar:
    st.title("Configuración de modelo")

    st.session_state.model = st.selectbox("Elige un modelo:", model_options, index=0)
    st.session_state.temperature = st.slider('Selecciona una temperatura:', 0.0, 1.0, 0.0, 0.01, format="%.2f")

    if st.button("Clear Chat 🧹", use_container_width=True):
        reset_chat_history()

    st.session_state.web_mode  = st.toggle("Modo Web", value=False)
    #st.write("Modo Web:", "Activado" if st.session_state.web_mode else "Desactivado")
render_or_update_model_info(st.session_state.model)

# Mostrar historial de mensajes
for msg in st.session_state.memory.messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)

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
        if not st.session_state.web_mode:
            response = invoke_basic_chain(
                input_text=text_input,
                chat_history=st.session_state.memory.messages,
                streaming=True
            )
            st.write_stream(response)
            # Guardar historial
            st.session_state.memory.add_user_message(text_input)
            st.session_state.memory.add_ai_message(invoke_basic_chain.response)

        else:
            st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            tavily_search_tool = TavilySearch(
                max_results=5,
                topic="general",
                country="spain",
            )
            #system_prompt = hub.pull("hwchase17/react")
            system_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """Perform the task to the best of your ability.
                        You is here to assist.

                        Current date: {current_date}

                        TOOLS:
                        ------
                        You has access to the following tools:
                        {tools}
                        To use a tool, please use the following format:
                        ```
                        Thought: Do I need to use a tool? Yes
                        Action: the action to take, should be one of [{tool_names}]
                        Action Input: the input to the action
                        Observation: the result of the action
                        ```
                        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
                        ```
                        Thought: Do I need to use a tool? No
                        Final Answer: [your response here]
                        ```
                        you always MUST use the format:
                        Thought: you should always think about what to do
                        Action Input: the input to the action
                        Observation: the result of the action
                        ... (this Thought/Action/Action Input/Observation can repeat N times)
                        Thought: I now know the final answer
                        Final Answer: the final answer to the original input question

                        Your answer must always contain points:
                        Thought, Final Answer or Thought, Action, Action Input

                        Your answer can`t contain both points 'Final Answer' and 'Action'

                        New input: {input}
                        {agent_scratchpad}
                        """
                    ),
                    MessagesPlaceholder("chat_history", optional=True),
                    ("human", "Pregunta: {input}"),
                    ("ai", "{agent_scratchpad}"),
                ]
            )
            tools = [tavily_search_tool]
            LLM_MODEL = "gemini-2.5-flash-preview-05-20"
            llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL,
                temperature=0,
                max_tokens=None,
                max_retries=2,
            )
            
            agent = create_react_agent(llm, tools, system_prompt)
            agent_executor = AgentExecutor(agent=agent,
                                           tools=tools,
                                           verbose=True,
                                           handle_parsing_errors=True)
            agent_with_chat_history = RunnableWithMessageHistory(
                agent_executor,
                #get_session_history=st.session_state.memory,  # Use a fixed session_id for simplicity
                lambda session_id: st.session_state.memory,  # Associates memory with session_id
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="output" 
            )
            response = agent_with_chat_history.invoke(
                input={"input": prompt, "current_date": datetime.now().strftime("%Y-%m-%d: %H:%M:%S")},
                config={"configurable": {"session_id": "test-session"},
                        "callbacks": [st_callback]}
            )
            #st.write(response)
            st.write(response["output"])
            #st.session_state.memory.add_user_message(prompt)
            #st.session_state.memory.add_ai_message(response["output"])
    
