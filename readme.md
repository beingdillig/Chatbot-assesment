# AI Policy Assistant (RAG + Agentic Workflow)

A production-ready AI agent designed to answer company policy questions using a Retrieval-Augmented Generation (RAG) approach. The system uses a cyclic graph to autonomously decide when to consult internal documentation versus answering directly.

---

## 🏗 Architecture Overview

The system is built as a **stateful cyclic graph** using LangGraph. This allows the agent to reason, act, and observe in a loop until a satisfactory answer is generated.



### Workflow:
1.  **User Input:** Received via FastAPI.
2.  **Agent Node (LLM):** Evaluates the query. If it requires policy data, it outputs a specific `ACTION` string.
3.  **Conditional Router:** Detects the `ACTION` pattern via Regex.
4.  **Tool Node (FAISS):** Performs similarity search on the local vector store.
5.  **Observation:** The retrieved text is fed back to the Agent to synthesize a final response.

---

## 🛠 Tech Stack

* **Frameworks:** FastAPI, LangGraph, LangChain
* **LLM:** OpenAI (GPT-based via custom API gateway)
* **Vector Database:** FAISS (Local Vector Store)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Resilience:** Tenacity (Exponential backoff retry logic)
* **Server:** Uvicorn

---

## 🚀 Setup Instructions

### Local Development

1. **Clone the repository and install dependencies:**
   ```bash
   pip install fastapi uvicorn langchain langchain-openai langchain-huggingface faiss-cpu langgraph tenacity
2. **Configure Environment Variables:**
    ```bash
    export OPENAI_API_KEY="your_api_key"

3. **Initialize the Vector Store: Place your .txt policy files in a /data folder, then run:**
    ```bash
    python vectordb.py
4. **Start the API:**
    ```bash
    uvicorn backend:app --reload