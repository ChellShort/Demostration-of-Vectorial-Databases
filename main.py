from app.compile_agent import agent
from io import BytesIO
from PIL import Image
from langchain.messages import HumanMessage
from gradio_interface.main import ChatInterfaceInit
import asyncio

# img = Image.open(BytesIO(agent.get_graph(xray=True).draw_mermaid_png()))
# img.save("saved_image.png")

# messages = [HumanMessage(content="Cual es el reglamento de la hora de comida?")]
# messages = agent.invoke({"messages": messages})
# for m in messages["messages"]:
#     m.pretty_print()
    
if __name__ == "__main__":
    asyncio.run(ChatInterfaceInit())