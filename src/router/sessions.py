import datetime
import json
from typing import Optional, List, Dict
from uuid import UUID

from dotenv import load_dotenv
from fastapi import APIRouter, Depends
from fastapi import Query
from langchain_groq import ChatGroq
from loguru import logger
from pydantic import BaseModel
from sqlmodel import Session as DBSession
from sqlmodel import select

from src.db.dbs import get_db
from src.db.redis_client import redis
from src.models.schema import Message, SenderRole
from src.models.schema import Session as SessionModel
from src.models.schema import User
from src.router.auth import get_current_user
from src.router.setup import llm_instances


from fastapi.responses import StreamingResponse
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.memory.chat_message_histories import RedisChatMessageHistory

import asyncio



logger.add("logs/api.log", rotation="1 MB", retention="10 days", level="INFO")

logger.info("Server started")

router = APIRouter()

load_dotenv()


class MsgRequest(BaseModel):
    session_id: str
    isFirst : bool = False
    msg: str



class SessionInfo(BaseModel):
    session_id: str
    user_id: str
    model: Optional[str]
    title: str = "New Chat"
    created_at: str
    updated_at: Optional[str] = None


class MessageInfo(BaseModel):
    message_id: str
    session_id: str
    content: str
    sender: str
    timestamp: str


class TitleUpdateRequest(BaseModel):
    title: str


class TitleResponse(BaseModel):
    title: str


class SessionResponse(BaseModel):
    session_id: str


@router.post("/new", response_model=SessionResponse)
async def create_new_session(user: User = Depends(get_current_user), db: DBSession = Depends(get_db)):
    try:

        try:
            new_session = SessionModel(
                user_id=user.userid,
                title="New Chat",
                model="default",
            )
        except Exception as model_error:
            raise HTTPException(status_code=422, detail=f"Model creation failed: {str(model_error)}")

        db.add(new_session)
        db.commit()
        db.refresh(new_session)

        return SessionResponse(
            session_id=str(new_session.session_id)
        )

    except HTTPException:
        raise
    except Exception as e:
        try:
            db.rollback()
        except:
            pass

        raise HTTPException(
            status_code=500,
            detail=f"Session creation failed: {str(e)}"
        )


class qdrant_convert(BaseModel):
    point_id: str
    vector: List[float]
    payload: Dict
    collection_name: str


from sentence_transformers import SentenceTransformer


def conversion_for_qdrant(msg: MessageInfo, collection_name: str):
    msg_id = msg.message_id  # str, not tuple

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    vector = embed_model.encode(msg.content)
    payload = {
        "content": msg.content,
        "sender": msg.sender,
        "timestamp": msg.timestamp
    }

    return qdrant_convert(point_id=msg_id, vector=vector, payload=payload, collection_name=collection_name)


from fastapi import Request, Body, Depends, HTTPException
from uuid import uuid4
from datetime import datetime


@router.get("/history/{session_id}")
async def get_chat_history(session_id: str, db: DBSession = Depends(get_db)):
    """Get all messages in a chat session"""
    messages = db.query(Message).filter(
        Message.session_id == session_id
    ).order_by(Message.timestamp).all()
    return messages



@router.post("/simple")
async def message(
        request: Request,
        db: DBSession = Depends(get_db),
        body: MsgRequest = Body(...)
):
    current_model = getattr(request.app.state, "current_model", None)
    if not (current_model and hasattr(current_model, "invoke")):
        raise HTTPException(status_code=401, detail="No valid model found")

    session_id = body.session_id
    if not session_id:
        raise HTTPException(status_code=422, detail="Missing current_session_id in request body")

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

        if body.isFirst:
            title = session_title_gen(body.msg)


        add_msg_to_dbs(body.msg, session_id, db)

        from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
        from langchain.chains import LLMChain

        # Define system prompt (markdown-safe)
        # Replace your system prompt in sessions.py with this:

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
        - Add empty lines before and after headers
        - Use **bold** and *italic* correctly
        - For tables, use proper markdown table format

        ## RESPONSE QUALITY:
        - Be conversational but well-structured
        - Break long responses into clear paragraphs
        - Use examples when explaining concepts
        - Always proofread for proper spacing and formatting
        
        dont forget to ask or suggest follow up question user can ask about the topic
        like asked what is ollama 
        at last  ask 
        would you like to set up ollama
        wanna learn more about local models
        
        feel free to ask questions

        Remember: Consistent, readable formatting is as important as the content itself."""

        # Prompt template setup
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        # LLM chain with prompt
        chain = LLMChain(
            llm=current_model,
            prompt=prompt,
            memory=memory,
            verbose=True,
        )

        # Invoke chain with user input
        response = chain.invoke({
            "input": body.msg
        })

        response_text = response.get('response', str(response)) if isinstance(response, dict) else str(response)

        assistant_msg_info = add_msg_to_dbs(response_text, session_id, db, False)

        return assistant_msg_info

    except Exception as e:
        logger.error(f"Error in message processing for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")



class SessionTitle(BaseModel):
    title:str


async def session_title_gen(query):
    try:
        title_gen = ChatGroq(model_name="compound-beta")
        session_title = await title_gen.ainvoke(
            f"You are a llm which help to generate meaningful session title for a chatgpt like app so here is title and generate a session title in one line for query:{query}")

        # Ensure we return a string
        result = session_title.content if hasattr(session_title, 'content') else str(session_title)
        print(f"Session title result: {result}, type: {type(result)}")
        return str(result)  # Force string conversion
    except Exception as e:
        print(f"Error in session_title_gen: {e}")
        return "New Chat00000"

        # Return a fallback title
@router.post("/simple-stream")
async def message_stream(
        request: Request,
        db: DBSession = Depends(get_db),
        body: MsgRequest = Body(...)
):
    current_model = getattr(request.app.state, "current_model", None)
    if not current_model:
        raise HTTPException(status_code=401, detail="No model found")

    session_id = body.session_id
    if not session_id:
        raise HTTPException(status_code=422, detail="Missing session_id")

    redis_key_prefix = f"chat_session:{session_id}"

    try:
        # Initialize memory and history
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

        # Save user message to DB
        from src.db.dbs import add_msg_to_dbs
        add_msg_to_dbs(body.msg, session_id, db)

        # Generate title BEFORE the streaming function
        title = ""
        if body.isFirst:
            try:
                title_result = await session_title_gen(body.msg)
                print(f"Title result type: {type(title_result)}")
                print(f"Title result: {title_result}")

                # Ensure it's a string
                if hasattr(title_result, 'content'):
                    title = str(title_result.content)
                else:
                    title = str(title_result)

                logger.info(f"Final title: {title}")
            except Exception as e:
                logger.error(f"Title generation error: {e}")
                title = "New Chat"  # Fallback title



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

        # Construct prompt with history
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
            buffer = ""  # Buffer for word-boundary streaming


            try:
                # Send start signal
                yield f"data: {json.dumps({'type': 'start', 'content': ''})}\n\n"

                formatted_messages = prompt.format_messages(input=body.msg)

                async for chunk in current_model.astream(formatted_messages):
                    if hasattr(chunk, "content") and chunk.content:
                        token = chunk.content
                        full_response += token
                        buffer += token

                        # Send tokens in word boundaries for better UX
                        if ' ' in buffer or '\n' in buffer:
                            yield f"data: {json.dumps({'type': 'token', 'content': buffer})}\n\n"
                            buffer = ""

                        # Small delay to prevent overwhelming the client
                        await asyncio.sleep(0.01)

                # Send any remaining buffered content
                if buffer:
                    yield f"data: {json.dumps({'type': 'token', 'content': buffer})}\n\n"

                # Send completion signal
                yield f"data: {json.dumps({'type': 'done', 'content': full_response})}\n\n"


                yield f"data: {json.dumps({'type': 'title', 'content': title})}\n\n"


                # Update memory and save to DB
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
@router.patch("/{session_id}/title", response_model=TitleResponse)
async def update_session_title(
        session_id: str,
        request: TitleUpdateRequest,
        db: DBSession = Depends(get_db)
):
    """Update session title"""
    try:
        session_uuid = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID format")

    session_query = select(SessionModel).where(SessionModel.session_id == session_uuid)
    session = db.execute(session_query).scalars().first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.title = request.title

    session.updated_at = datetime.utcnow()

    db.add(session)
    db.commit()
    db.refresh(session)

    logger.info(f"Updated title for session {session_id}: {request.title}")
    return TitleResponse(title=session.title)


@router.delete("/{session_id}")
async def delete_session(session_id: str, db: DBSession = Depends(get_db), user=Depends(get_current_user)):
    # 1. Parse & validate UUID
    try:
        sid = UUID(session_id)
    except ValueError:
        raise HTTPException(400, "Invalid session ID format")

    # 2. Fetch the Session row
    stmt = select(SessionModel).where(SessionModel.session_id == sid)
    result = db.execute(stmt)
    session_row = result.scalars().first()
    if not session_row:
        raise HTTPException(404, "Session not found")

    # 3. (Optional) Authorization check here:
    if session_row.user_id != user.userid: raise HTTPException(403)

    # 4. Delete all Messages linked to it
    msg_stmt = select(Message).where(Message.session_id == sid)
    messages = db.execute(msg_stmt).scalars().all()
    for m in messages:
        db.delete(m)

    # 5. Delete the Session row itself
    db.delete(session_row)
    db.commit()

    # 6. Redis cleanup, etc.
    try:
        redis.delete(f"chat:{session_id}")
    except:
        pass

    return True


@router.get("/getAll")
async def get_all_sessions(
        db: DBSession = Depends(get_db),
        user: User = Depends(get_current_user)
):
    """Get all sessions for the current user"""
    try:
        session_query = (
            select(SessionModel)
            .where(SessionModel.user_id == user.userid)
            .order_by(SessionModel.updated_at)
        )
        sessions = db.execute(session_query).scalars().all()

        return [
            {
                "id": str(session.session_id),
                "session_id": str(session.session_id),
                "title": session.title,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat() if session.updated_at else None
            }
            for session in sessions
        ]

    except Exception as e:
        logger.error(f"Error fetching sessions for user {user.userid}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch sessions")

