import gradio as gr
from app.compile_agent import agent
from io import BytesIO
from PIL import Image
from langchain.messages import HumanMessage, AIMessage
import asyncio

def chat(message, history):
    # messages = []

    # for item in history:
    #     if item["role"] == "user":
    #         messages.append(HumanMessage(content=item["content"]))
    #     elif item["role"] == "assistant":
    #         messages.append(AIMessage(content=item["content"]))
    #     else:
    #         pass

    # messages.append(HumanMessage(content=message)) # Importante
    
    response = agent.invoke( {"messages": [{"role": "user", "content": message}]},
    {"configurable": {"thread_id": "1"}},)
    text = response["messages"][-1].content
    return text

async def ChatInterfaceInit():
    demo = gr.ChatInterface(fn=chat, examples=["hello", "hola", "merhaba"], title="BOT DE DEMOSTRACIÓN")
    demo.launch()