from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token=hf_token
)
chat_model = ChatHuggingFace(llm=llm)

# Variable donde se almacenan los chats
store = {}

# Funcion que obtiene el historial de mensajes de una sesion
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(chat_model, get_session_history)


while True:
    sesion = input("Ingrese id de sesion: ")
    if sesion == "exit":
        break

    config = {"configurable": {"session_id": sesion}}
    while True:
        user_input = input("You: ")
        response = with_message_history.invoke(
            HumanMessage(content=user_input),
            config=config
        )
        print(f"IA: {response.content}")

        if response.content == "¡Hasta luego!":
            break