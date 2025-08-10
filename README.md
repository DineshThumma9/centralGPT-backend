# 🧠 CentralGPT Backend

**One Nexus Point to access and chat with multiple LLM providers — Mistral, Ollama, Groq, TogetherAI —  with GitHub & PDF RAG capabilities.**

Supports memory, chat history, persistent messages, and file retrieval from GitHub.

---

## 🛠 Tech Stack

| Layer                | Tech                      |
| -------------------- | ------------------------- |
| **Framework**        | FastAPI                   |
| **LLM Framework**    | LangChain                 |
| **Validation**       | Pydantic                  |
| **Database**         | Redis, PostgreSQL, Qdrant |
| **Containerization** | Docker                    |
| **Deployment**       | Azure App Service         |
| **Git Integration**  | PyGithub                  |

---

## 🚀 Features

* 🔁 Chat with **multiple LLM providers**
* 💾 Persistent chat history (PostgreSQL, Redis)
* 🧠 Session memory support
* 🔄 Dynamic model switching (Ollama, Groq, Mistral, etc.)
* 📂 **GitHub Integration**: Fetch files from any public/private repository using username , repo name,branch and commit
* 📄 **PDF & Document RAG**: Upload PDFs and retrieve context-aware answers
* 🔍 Vector search via Qdrant for fast retrieval

---

## 🧩 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/DineshThumma9/centralGPT-backend.git
cd centralGPT-backend
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```
DATABASE_URL=
REDIS_URL=
QDRANT_URL=
QDRANT_API_KEY=
GROQ_API_KEY=
COHERE_API_KEY=
HUGGINGFACE_HUB_TOKEN=
GITHUB_TOKEN=
```

### 4. Frontend Setup (Required)

Clone and set up the frontend:

```bash
git clone https://github.com/DineshThumma9/centralGPT.git
```

Follow the frontend repo instructions to configure and run it.

---

## 🧪 Development (Local)

```bash
uvicorn src.main:app --reload 
```

---
