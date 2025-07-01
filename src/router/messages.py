import asyncio
import json
import os
from typing import Dict, List

from fastapi import APIRouter
from fastapi import Depends, HTTPException, Body
from fastapi.responses import StreamingResponse
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
from loguru import logger
from pydantic import BaseModel
from sqlmodel import Session as DBSession

from src.db.dbs import get_db, add_msg_to_dbs
from src.router.auth import get_current_user
from src.router.setup import get_llm_instance

system_prompt = """You are a professional AI assistant that MUST follow these strict formatting rules:

    ## TEXT FORMATTING:
    - Always use proper spacing between words and sentences
    - End sentences with periods followed by ONE space
    - Use double line breaks (\\n\\n) between paragraphs
    - Never compress words together or create run-on sentences

    ## CODE FORMATTING:
    - ALWAYS wrap code in proper fenced code blocks with language specification:
      ```python
      # Your code here
      ```
    - For inline code, use single backticks: `code`
    - Never mix inline code with code blocks
    - Always include the language name after the opening ```

    ## LIST FORMATTING:
    - Use proper markdown lists with consistent spacing:
      - Bullet point 1
      - Bullet point 2

      Or numbered:
      1. First item
      2. Second item

    - Always put ONE space after the bullet/number
    - Each list item on its own line
    - Add empty line before and after lists

    ## STRUCTURE REQUIREMENTS:
    - Use proper headers: # ## ### 
    - like for Main Topic #
    - for subtopics ## 
    - Add empty lines before and after headers
    - Use **bold** and *italic* correctly use bold for keywords etc
    - For tables, use proper markdown table format

    ## RESPONSE QUALITY:
    - Be conversational but well-structured
    - Break long responses into clear paragraphs
    - Use examples when explaining concepts
    - Always proofread for proper spacing and formatting

    Remember: Consistent, readable formatting is as important as the content itself."""

router = APIRouter()


class MsgRequest(BaseModel):
    session_id: str
    isFirst: bool = False
    msg: str


class MessageInfo(BaseModel):
    message_id: str
    session_id: str
    content: str
    sender: str
    timestamp: str


class qdrant_convert(BaseModel):
    point_id: str
    vector: List[float]
    payload: Dict
    collection_name: str


def conversion_for_qdrant(msg: MessageInfo, collection_name: str):
    msg_id = msg.message_id  # str, not tuple

    vector = msg.content
    payload = {
        "content": msg.content,
        "sender": msg.sender,
        "timestamp": msg.timestamp
    }

    return qdrant_convert(point_id=msg_id, vector=vector, payload=payload, collection_name=collection_name)


async def session_title_gen(query):
    try:
        title_gen = ChatGroq(model_name="compound-beta", api_key=os.getenv("GROQ_API_KEY"))

        prompt = f"""Generate a concise, descriptive title (maximum 6 words) for a chat session based on this first message: "{query}"

Rules:
- Maximum 6 words
- No quotes or special characters
- Describe the main topic or question
- Be specific but concise

Title:"""

        session_title = await title_gen.ainvoke(prompt)

        result = session_title.content if hasattr(session_title, 'content') else str(session_title)

        # Clean and validate the result
        if result:
            # Remove quotes and clean up
            cleaned_title = result.strip().strip('"').strip("'")
            if cleaned_title and len(cleaned_title) > 0:
                return cleaned_title

        return "New Chat"

    except Exception as e:
        logger.error(f"Error in session_title_gen: {e}")
        return "New Chat"


@router.post("/simple-stream")
async def message_stream(
        body: MsgRequest = Body(...),
        db: DBSession = Depends(get_db),
        user=Depends(get_current_user)
):
    current_model = get_llm_instance(db=db, user=user)
    if not current_model:
        raise HTTPException(status_code=401, detail="No model found")

    session_id = body.session_id
    redis_key_prefix = f"chat_session:{session_id}"

    try:
        # Set up memory for context
        chat_history = RedisChatMessageHistory(
            session_id=redis_key_prefix,
            url=os.getenv("REDIS_URL"),
            key_prefix="langchain:chat_history:",
            ttl=3600
        )

        memory = ConversationBufferMemory(
            chat_memory=chat_history,
            return_messages=True,
            memory_key="history"
        )

        # Get conversation history
        history_messages = memory.chat_memory.messages

        # Build full conversation for the model
        messages = []
        messages.append({"role": "system", "content": system_prompt})

        # Add history
        for msg in history_messages:
            if hasattr(msg, 'type'):
                role = "user" if msg.type == "human" else "assistant"
                messages.append({"role": role, "content": msg.content})

        # Add current message
        messages.append({"role": "user", "content": body.msg})

        add_msg_to_dbs(body.msg, session_id, db)

        # Generate title if first message
        title = ""
        if body.isFirst:
            try:
                title = await session_title_gen(body.msg)
                if not title or title.strip() == "":
                    title = "New Chat"
            except Exception as e:
                logger.error(f"Title generation error: {e}")
                title = "New Chat"

        async def stream_response():
            full_response = ""

            try:
                yield f"data: {json.dumps({'type': 'start', 'content': ''})}\n\n"

                # Stream directly from model
                async for chunk in current_model.astream(messages):
                    if hasattr(chunk, "content") and chunk.content:
                        token = chunk.content
                        full_response += token
                        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                        await asyncio.sleep(0.01)

                yield f"data: {json.dumps({'type': 'done', 'content': full_response})}\n\n"

                # Send title after completion
                if title:
                    yield f"data: {json.dumps({'type': 'title', 'content': title})}\n\n"

                # Save to memory and database
                memory.chat_memory.add_user_message(body.msg)
                memory.chat_memory.add_ai_message(full_response)
                add_msg_to_dbs(full_response, session_id, db, isUser=False)

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    except Exception as e:
        logger.error(f"Streaming setup error: {e}")
        raise HTTPException(status_code=500, detail="Streaming failed")
