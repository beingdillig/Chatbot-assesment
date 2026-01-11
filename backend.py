from fastapi import FastAPI, Request
from langchain_core.messages import HumanMessage
from main import agent

app = FastAPI()

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    user_query = data.get("query")
    
    # Initialize the state with a HumanMessage
    inputs = {"messages": [HumanMessage(content=user_query)]}
    
    try:
        # Run the agent
        config = {"recursion_limit": 10}
        result = agent.invoke(inputs, config=config)
        
        # Get the very last message from the LLM
        final_answer = result["messages"][-1].content
        return {"answer": final_answer}
        
    except Exception as e:
        return {"error": str(e)}