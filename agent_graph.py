import json
import re
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai
import os

# 1. Setup LLM, Faiss
llm = ChatOpenAI(
    model_name=("MODEL_NAME","provider-2/gpt-oss-20b"), 
    base_url=("OPENAI_BASE_URL","https://api.a4f.co/v1"),
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=600
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")

db = FAISS.load_local(
    FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Logic: Manual Tool Trigger
SYSTEM_PROMPT = """You are a helpful assistant. 
If you need to search company policies or documents to answer, you must output EXACTLY:
ACTION: query_knowledge_base("your search query")

Otherwise, answer the user directly."""

@retry(
    retry=retry_if_exception_type(openai.RateLimitError),
    wait=wait_exponential(multiplier=1, min=20, max=70),
    stop=stop_after_attempt(3)
)
def call_model(state: AgentState):
    print("---CALLING LLM---")
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def tool_node(state: AgentState):
    print("---EXECUTING TOOL---")
    last_message = state["messages"][-1].content
    match = re.search(r'query_knowledge_base\("([^"]+)"\)', last_message)
    if match:
        query = match.group(1)
        docs = db.similarity_search(query, k=3)
        content = "\n\n".join([d.page_content for d in docs])
        return {"messages": [HumanMessage(content=f"Observation from Documents: {content}")]}
    return {"messages": [HumanMessage(content="No data found.")]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1].content
    if "ACTION: query_knowledge_base" in last_message:
        return "tools"
    return END

# 3. Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

agent = workflow.compile()