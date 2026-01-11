from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from agent_graph import agent

app = FastAPI(title="LangGraph AI Agent with RAG")

class AskRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask(req: AskRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    inputs = {"messages": [HumanMessage(content=req.query)]}
    config = {"recursion_limit": 10}

    result = agent.invoke(inputs, config=config)
    final_answer = result["messages"][-1].content

    return {"answer": final_answer}

@app.get("/health")
def health():
    return {"status": "ok"}
