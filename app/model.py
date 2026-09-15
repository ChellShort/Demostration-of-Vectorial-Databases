
from langchain.chat_models import init_chat_model
from langchain_aws import ChatBedrockConverse
from app.core.config import QWEN3_ID, BEDROCK_API_KEY
from app.tools.add import add
from app.tools.multiply import multiply
from app.tools.divide import divide
from app.nodes.consult_qdrant import vector_search

model = ChatBedrockConverse(
        model_id=QWEN3_ID,
        api_key = BEDROCK_API_KEY,
        region_name="us-east-2",
        max_tokens=1048
    )

# Augment the LLM with tools
tools = [add, multiply, divide, vector_search]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)
