from app.compile_agent import agent
# from IPython.display import Image, display
from io import BytesIO
from PIL import Image

img = Image.open(BytesIO(agent.get_graph(xray=True).draw_mermaid_png()))
img.save("saved_image.png")


# Invoke
from langchain.messages import HumanMessage
messages = [HumanMessage(content="What's the rujles in the workplace? perform a vectorized search")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()