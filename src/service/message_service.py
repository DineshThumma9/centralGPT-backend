import os

from fastapi import APIRouter
from llama_index.llms.groq import Groq
from loguru import logger

system_prompt = """You are a professional AI assistant that MUST follow these strict formatting rules:

    
    ## RESPONSE QUALITY:
    - Be conversational but well-structured
    - Break long responses into clear paragraphs
    - Use examples when explaining concepts
    - Always proofread for proper spacing and formatting

    Remember: Consistent, readable formatting is as important as the content itself."""

rag_prompt = """ 

YOU are smart RAG Model which read content and answer user query you know all filename and dir structure of code and helpful to user



"""

router = APIRouter()


async def session_title_gen(query):
    try:
        title_gen = Groq(model="compound-beta", api_key=os.getenv("GROQ_API_KEY"))

        prompt = f"""Generate a concise, descriptive title (maximum 6 words) for a chat session based on this first message: "{query}"

Rules:
- Maximum 6 words
- No quotes or special characters
- Describe the main topic or question
- Be specific but concise

Title:"""

        session_title = await title_gen.acomplete(prompt)

        result = session_title.content if hasattr(session_title, 'content') else str(session_title)


        if result:

            cleaned_title = result.strip().strip('"').strip("'")
            if cleaned_title and len(cleaned_title) > 0:
                return cleaned_title

        return "New Chat"

    except Exception as e:
        logger.error(f"Error in session_title_gen: {e}")
        return "New Chat"
