from langchain.messages import SystemMessage, AIMessage
from app.model import model_with_tools

def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    
    response = model_with_tools.invoke(
                    [
                        SystemMessage(
                            content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                        )
                    ]
                    + state["messages"]
                )
    response = AIMessage(
            content=response.content,
            additional_kwargs=response.additional_kwargs,
            response_metadata=response.response_metadata,
            tool_calls=response.tool_calls
        )
    
    return {
        "messages": [
            response
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }