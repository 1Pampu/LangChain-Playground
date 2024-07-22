from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace

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

chat_historial = []
while True:
    user_input = input("You: ")
    messages = HumanMessage(content=user_input)
    chat_historial.append(messages)
    response = chat_model.invoke(chat_historial)
    chat_historial.append(response)
    print(f"IA: {response.content}")
    if response.content == "¡Hasta luego!":
        break