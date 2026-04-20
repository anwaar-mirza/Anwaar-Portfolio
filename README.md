<div align="center">

# 🤖 Anwaar's Personal AI Assistant
### RAG-Powered Portfolio Chatbot with FastAPI Backend

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-00B388?style=for-the-badge)](https://pinecone.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Qwen2.5_7B-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)

<p align="center">
  <b>An intelligent, RAG-based conversational AI that answers questions about Anwaar Rasool's professional background — skills, projects, experience, and availability — using a Pinecone vector database, LangChain memory chains, and a Qwen2.5-7B LLM served via HuggingFace Inference Endpoints.</b>
</p>

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Chatbot Deep Dive](#-chatbot-deep-dive)
- [FastAPI Backend](#-fastapi-backend)
- [Setup & Installation](#-setup--installation)
- [Environment Variables](#-environment-variables)
- [API Reference](#-api-reference)
- [How RAG Works Here](#-how-rag-works-here)

---

## 🧠 Overview

This project is a **production-ready AI assistant** that acts as an interactive portfolio agent. Instead of a static CV, visitors can *chat* with an LLM that has deep knowledge of Anwaar's professional background — powered by **Retrieval-Augmented Generation (RAG)**.

**Key capabilities:**
- Answers natural language questions about skills, experience, and projects
- Maintains **conversation history** across a session (context-aware follow-ups)
- Retrieves relevant knowledge chunks from a **Pinecone vector database** before answering
- Served via a **FastAPI REST endpoint** consumed by the portfolio frontend

---

## 🏗️ Architecture

```
User Query (Frontend)
        │
        ▼
┌─────────────────────┐
│    FastAPI Server    │  ← GET /chat?query=...
│    (main.py)         │
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│              PersonalChatBot (RAG Pipeline)          │
│                                                     │
│  ┌─────────────┐    ┌──────────────────────────┐   │
│  │  User Query  │───▶│  History-Aware Retriever  │   │
│  └─────────────┘    │  (Contextualizes query     │   │
│                     │   using chat history)       │   │
│                     └────────────┬───────────────┘   │
│                                  │                   │
│                                  ▼                   │
│                     ┌────────────────────────┐       │
│                     │   Pinecone Vector DB    │       │
│                     │   (Knowledge Base)      │       │
│                     │   k=5 similar chunks    │       │
│                     └────────────┬───────────┘       │
│                                  │                   │
│                                  ▼                   │
│                     ┌────────────────────────┐       │
│                     │  Stuff Documents Chain  │       │
│                     │  + Retrieval Prompt     │       │
│                     └────────────┬───────────┘       │
│                                  │                   │
│                                  ▼                   │
│                     ┌────────────────────────┐       │
│                     │  Qwen2.5-7B-Instruct   │       │
│                     │  (HuggingFace Endpoint) │       │
│                     └────────────┬───────────┘       │
│                                  │                   │
│  ┌───────────────────────────────▼───────────────┐  │
│  │   RunnableWithMessageHistory (Session Memory)  │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
         │
         ▼
  {"response": "..."}  ← JSON back to frontend
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | Qwen2.5-7B-Instruct (HuggingFace Endpoint) | Text generation & reasoning |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Query & document vectorization |
| **Vector DB** | Pinecone | Storing & retrieving knowledge chunks |
| **RAG Framework** | LangChain | Chains, retrievers, memory, prompts |
| **Memory** | `ChatMessageHistory` + `RunnableWithMessageHistory` | Per-session conversation history |
| **Backend API** | FastAPI | REST endpoint serving the chatbot |
| **Config** | python-dotenv | Secure environment variable management |
| **Frontend** | HTML/CSS/JS | Portfolio UI (served via FastAPI StaticFiles) |

---

## 📁 Project Structure

```
anwaars-personal-assistant/
│
├── main.py                  # FastAPI app — routes, CORS, lifespan
├── PersonalChatBot.py       # Core RAG chatbot class
├── prompts.py               # System & contextualization prompts
├── .env                     # API keys (never commit this)
├── .gitignore
├── requirements.txt
│
└── static/
    └── portfolio.html       # Frontend — served at root "/"
```

---

## 🤖 Chatbot Deep Dive

The `PersonalChatBot` class in `PersonalChatBot.py` encapsulates the full RAG pipeline. Here is what happens step by step when a user sends a message:

### 1. Model — `create_model()`
```python
HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct", max_new_tokens=512, temperature=0.7)
```
Uses HuggingFace's serverless inference endpoint for **Qwen2.5-7B-Instruct** — a capable open-source instruction-tuned LLM. Wrapped in `ChatHuggingFace` for LangChain compatibility.

### 2. Embeddings — `create_embeddings()`
```python
HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```
Converts both the user's query and stored knowledge chunks into dense vector representations for semantic similarity search.

### 3. Vector Store & Retriever — `create_vector_store()`
```python
PineconeVectorStore(index=self.index, embedding=self.embeddings).as_retriever(search_kwargs={"k": 5})
```
Connects to the `anwaars-knowledge-base` Pinecone index and fetches the **top 5 most semantically relevant chunks** for each query.

### 4. History-Aware Retriever — `create_history_retriever()`
```python
create_history_aware_retriever(llm=self.model, retriever=self.retriever, prompt=self.contextualize_prompt)
```
Before searching Pinecone, this step **rephrases the user's query** using the conversation history — so follow-up questions like *"Tell me more about that"* still retrieve the right context.

### 5. Document Chain — `create_doc_chain()`
```python
create_stuff_documents_chain(llm=self.model, prompt=self.retrieval_prompt)
```
Takes the retrieved Pinecone chunks, stuffs them into the prompt context, and passes everything to the LLM to generate a grounded answer.

### 6. Retrieval Chain — `crete_ret_chain()`
```python
create_retrieval_chain(self.history_retriever, self.doc_chain)
```
Combines the history-aware retriever and the document chain into one end-to-end pipeline.

### 7. Final Chain with Memory — `create_final_chain()`
```python
RunnableWithMessageHistory(
    self.retrival_chain,
    self.my_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)
```
Wraps the full pipeline with **session-based message history**. Each user session gets a unique `session_id` (UUID), and conversation turns are stored in memory so the bot remembers context across the full session.

### 8. Invocation — `invoke_chain(query)`
```python
response = self.chain.invoke({"input": query}, config={"configurable": {"session_id": self.session_id}})
return response['answer']
```
The single public method — takes a raw query string and returns the LLM's answer string.

---

## ⚡ FastAPI Backend

`main.py` exposes the chatbot through a clean REST API:

```
GET  /          → Serves portfolio.html (StaticFiles)
GET  /health    → Server health check
GET  /chat      → Main chatbot endpoint
```

### Lifespan — Bot Initialization
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bot = PersonalChatBot()   # Initialized ONCE on startup
    yield
```
The `PersonalChatBot` is instantiated **once** when the server starts (not on every request). This means the Pinecone connection, embeddings model, and LangChain chains are all loaded once — keeping response times fast.

### CORS — Cross-Origin Access
```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```
Allows the portfolio frontend (served from any domain) to call the API without browser CORS blocks.

### Chat Endpoint
```python
@app.get("/chat", response_model=ChatResponse)
def chat(query: str = Query(...)):
    result = app.state.bot.invoke_chain(query)
    return ChatResponse(response=result)
```

**Request:**
```
GET /chat?query=What are your top skills?
```

**Response:**
```json
{
  "response": "Anwaar's top skills include Python, web scraping with Selenium and Playwright, ETL pipeline design, and Generative AI using LangChain and RAG..."
}
```

---

## 🚀 Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/anwaar-mirza/anwaars-personal-assistant.git
cd anwaars-personal-assistant
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create `.env` file
```bash
cp .env.example .env
# Fill in your API keys (see Environment Variables section)
```

### 5. Run the server
```bash
uvicorn main:app --reload
```

### 6. Open in browser
```
http://127.0.0.1:8000          → Portfolio + Chatbot UI
http://127.0.0.1:8000/docs     → FastAPI Interactive API Docs
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
HF_TOKEN=hf_your_huggingface_token_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

| Variable | Where to get it |
|---|---|
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `PINECONE_API_KEY` | [app.pinecone.io](https://app.pinecone.io) → API Keys |

> ⚠️ **Never commit your `.env` file.** It is already in `.gitignore`.

---

## 📡 API Reference

### `GET /health`
Returns server status.

**Response:**
```json
{ "status": "ok", "service": "Anwaar's Personal Assistant" }
```

---

### `GET /chat`

**Query Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | `string` | ✅ Yes | The user's natural language question |

**Success Response `200`:**
```json
{ "response": "string" }
```

**Error Responses:**

| Code | Reason |
|---|---|
| `400` | Empty query string |
| `500` | Bot/LLM internal error |

---

## 🔍 How RAG Works Here

```
"What scraping tools does Anwaar use?"
          │
          ▼
  [Contextualize with history]
  → No prior history, query unchanged
          │
          ▼
  [Embed query with MiniLM-L6-v2]
  → [0.23, -0.11, 0.87, ...]  (384-dim vector)
          │
          ▼
  [Pinecone similarity search — top 5 chunks]
  → "Expert in Selenium, Playwright, Requests..."
  → "Anti-bot handling with proxy rotation..."
  → "500+ scraping projects delivered..."
  → "BeautifulSoup for HTML parsing..."
  → "Scrapy familiarity for large crawls..."
          │
          ▼
  [Stuff chunks into prompt + send to Qwen2.5-7B]
          │
          ▼
  "Anwaar uses Selenium for dynamic JS sites,
   Playwright for modern browser automation,
   Requests for lightweight scraping, and
   BeautifulSoup for HTML parsing..."
```

This approach ensures the LLM **never hallucinates** about Anwaar — it only answers based on what's in the knowledge base.

---

<div align="center">

**Built by [M. Anwaar Rasool](https://github.com/anwaar-mirza) — Python Data Engineer | ETL | GenAI | Lahore, Pakistan**

*Open to remote collaboration and freelance projects — muhammadanwaarrasool@gmail.com*

</div>