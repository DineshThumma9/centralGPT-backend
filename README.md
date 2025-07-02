# 🧠 CentralGPT Backend

**One Nexus Point to access and chat with multiple LLM providers — Mistral, Ollama, Groq, TogetherAI.**  
Supports memory, chat history, and persistent messages.

---

## 🛠 Tech Stack

| Layer               | Tech                        |
|---------------------|-----------------------------|
| **Framework**       | FastAPI                     |
| **LLM Framework**   | LangChain                   |
| **Validation**      | Pydantic                    |
| **Database**        | Redis, PostgreSQL, Qdrant   |
| **Containerization**| Docker                      |
| **Deployment**      | Railway                     |

---

## 🚀 Features

- 🔁 Chat with **multiple LLM providers**
- 💾 Persistent chat history (PostgreSQL, Redis)
- 🧠 Session memory support
- 🔄 Dynamic model switching (Ollama, Groq, Mistral, etc.)

---

## 🧩 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/DineshThumma9/centralGPT-backend.git
cd centralGPT-backend
```

## 🧩 Setup Instructions

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```
### 3. Configure Environment Variables
Create a .env file in the root directory:
```
DATABASE_URL=
REDIS_URL=
QDRANT_URL=
QDRANT_API_KEY=
API_URL=
```

### 4. Frontend Setup (Required)
Clone and set up the backend:
```
git clone https://github.com/DineshThumma9/centralGPT.git
```
Follow the frontend repo instructions to configure and run it.

### 🧪 Development Local
```
uvicorn src.main:app --reload 
```
