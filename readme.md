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

Understood. I have moved the subscription disclaimer directly into the Deployment & DevOps section to keep all cloud-related context in one place. This keeps the top of your README focused on the project's purpose while providing the necessary context as soon as the reader looks for deployment info.

Here is the finalized, raw Markdown:

Markdown

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

## ☁️ Deployment & DevOps (Internship Exposure)
**Note on Deployment**: Due to Azure subscription limitations, I could not finalize a live deployment for this project. However, it is architected to be "cloud-ready."

During my recent internships, including a specialized DevOps Internship, I gained foundational exposure to modern cloud practices. While I am still growing in this area, I have implemented the following to ensure portability:

**Docker Basics:** I have hands-on experience containerizing Python applications (FastAPI) to ensure consistent performance across different environments.

**Azure Familiarity:** Through my internship projects, I have navigated the Azure Portal, worked with App Services, and understand how to manage images in the Azure Container Registry (ACR).

**Deployment Mindset:** I am familiar with CI/CD pipeline concepts and have worked in teams where automated deployments were part of the development lifecycle.

---
## 🧠 Design Decisions
**Custom Regex-Based Routing:** Since the 3rd-party OpenAI proxy used for this project does not natively support function calling, I architected a custom tool-calling trigger. The system prompt instructs the LLM to use a specific string format, which is then parsed via Regex to execute local tools.

**LangGraph for Control Flow:** Chosen over standard chains to allow for cyclic logic and robust error handling during the reasoning phase.

**Local Embeddings:** Using HuggingFaceEmbeddings locally reduces latency and API costs compared to calling external embedding endpoints.
