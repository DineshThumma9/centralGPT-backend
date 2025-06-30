import asyncio
import json
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

from src.db.dbs import get_db
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
        title_gen = ChatGroq(model_name="compound-beta")
        session_title = await title_gen.ainvoke(
            f"You are a llm which help to generate meaningful session title for a chatgpt like app so here is title and generate a session title in one line for query:{query}")

        result = session_title.content if hasattr(session_title, 'content') else str(session_title)
        print(f"Session title result: {result}, type: {type(result)}")
        logger.info(f"Session title result: {result}, type: {type(result)}")
        return str(result)
    except Exception as e:
        print(f"Error in session_title_gen: {e}")
        return "New Chat00000"


@router.post("/simple-stream")
async def message_stream(
        body: MsgRequest = Body(...),
        db: DBSession = Depends(get_db),
        user=Depends(get_current_user)

):
    current_model = get_llm_instance(
        db=db,
        user=user
    )
    if not current_model:
        raise HTTPException(status_code=401, detail="No model found")

    session_id = body.session_id
    if not session_id:
        raise HTTPException(status_code=422, detail="Missing session_id")

    redis_key_prefix = f"chat_session:{session_id}"

    try:

        chat_history = RedisChatMessageHistory(
            session_id=redis_key_prefix,
            url="redis://localhost:6379",
            key_prefix="langchain:chat_history:",
            ttl=3600
        )

        memory = ConversationBufferMemory(
            chat_memory=chat_history,
            return_messages=True,
            memory_key="history"
        )

        from src.db.dbs import add_msg_to_dbs
        add_msg_to_dbs(body.msg, session_id, db)

        title = ""
        if body.isFirst:
            try:
                title_result = await session_title_gen(body.msg)
                print(f"Title result type: {type(title_result)}")
                print(f"Title result: {title_result}")

                if hasattr(title_result, 'content'):
                    title = str(title_result.content)
                else:
                    title = str(title_result)

                logger.info(f"Final title: {title}")
            except Exception as e:
                logger.error(f"Title generation error: {e}")
                title = "New Chat000"

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        chain = LLMChain(
            llm=current_model,
            prompt=prompt,
            memory=memory,
            verbose=True
        )

        async def stream_response():
            full_response = ""
            buffer = ""

            try:

                yield f"data: {json.dumps({'type': 'start', 'content': ''})}\n\n"

                formatted_messages = prompt.format_messages(input=body.msg)

                async for chunk in current_model.astream(formatted_messages):
                    if hasattr(chunk, "content") and chunk.content:
                        token = chunk.content
                        full_response += token
                        buffer += token

                        if ' ' in buffer or '\n' in buffer:
                            yield f"data: {json.dumps({'type': 'token', 'content': buffer})}\n\n"
                            buffer = ""

                        await asyncio.sleep(0.01)

                if buffer:
                    yield f"data: {json.dumps({'type': 'token', 'content': buffer})}\n\n"

                yield f"data: {json.dumps({'type': 'done', 'content': full_response})}\n\n"

                yield f"data: {json.dumps({'type': 'title', 'content': title})}\n\n"

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
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )

    except Exception as e:
        logger.error(f"Streaming setup error: {e}")
        raise HTTPException(status_code=500, detail="Streaming failed")
