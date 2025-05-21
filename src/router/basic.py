import os
import datetime
import json
import logging
import time
from fastapi import HTTPException, Query, APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, Any
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from src.db.dbs import get_db, SessionLocal
from src.router.auth import get_current_user
from src.models.schema import User, Session as ChatSession, Message, SenderType

# Configure logging
logger = logging.getLogger("basic_router")

# Load environment variables
load_dotenv()

# Create a new router
router = APIRouter()

# Define LLM classes dictionary
llm_instances = {}


@router.post("/api/{api_key}")
def set_api_key(api_key: str):
    os.environ["GROQ_API_KEY"] = api_key
    logger.info("API key set")
    return JSONResponse(
        content={"message": "API key set successfully"},
        status_code=200
    )


@router.get("/providers")
def get_llm_providers():
    logger.info("LLM providers requested")
    return JSONResponse(
        content={"providers": ["groq", "ollama"]},
        status_code=200
    )


@router.post("/providers/{llm_prov}")
async def choose_llm_provider(llm_prov: str, request: Request):
    logger.info(f"Setting LLM provider: {llm_prov}")

    if llm_prov not in ["groq", "ollama"]:
        logger.warning(f"Invalid LLM provider requested: {llm_prov}")
        raise HTTPException(status_code=404, detail=f"LLM provider '{llm_prov}' not found")

    # Dynamically import to avoid circular imports
    if llm_prov == "groq":
        try:
            from langchain_groq import ChatGroq
            request.app.state.llm_class = ChatGroq
            logger.info("Groq provider loaded")
        except ImportError as e:
            logger.error(f"Failed to import langchain_groq: {str(e)}")
            raise HTTPException(status_code=500, detail="LangChain Groq integration not installed")
    elif llm_prov == "ollama":
        try:
            from langchain_ollama import ChatOllama
            request.app.state.llm_class = ChatOllama
            logger.info("Ollama provider loaded")
        except ImportError as e:
            logger.error(f"Failed to import langchain_ollama: {str(e)}")
            raise HTTPException(status_code=500, detail="LangChain Ollama integration not installed")

    return JSONResponse(
        content={"message": f"LLM provider '{llm_prov}' selected successfully"},
        status_code=200
    )


@router.post("/models/{model}")
async def choose_model(model: str, request: Request):
    logger.info(f"Setting model: {model}")

    llm_class = getattr(request.app.state, "llm_class", None)
    if llm_class is None:
        logger.warning("No LLM provider selected before requesting model")
        raise HTTPException(status_code=400, detail="No LLM provider selected")

    try:
        # Use a unique key for the model instance
        instance_key = f"{llm_class.__name__}_{model}"

        logger.debug(f"Creating model instance with key: {instance_key}")

        # Create the instance if it doesn't exist
        if instance_key not in llm_instances:
            llm_instances[instance_key] = llm_class(model=model)
            logger.info(f"Created new instance for {instance_key}")
        else:
            logger.info(f"Using existing instance for {instance_key}")

        # Set the current instance
        request.app.state.llm_instance = llm_instances[instance_key]
        request.app.state.current_model = model
    except Exception as e:
        logger.error(f"Failed to instantiate model {model}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to instantiate model: {str(e)}")

    return JSONResponse(
        content={"message": f"Model '{model}' instantiated successfully"},
        status_code=200
    )


@router.post("/chat/session")
async def create_chat_session(
        title: str,
        request: Request,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    if not hasattr(request.app.state, "current_model"):
        logger.warning("No model selected before creating chat session")
        raise HTTPException(status_code=400, detail="No model selected")

    logger.info(f"Creating new chat session with title: {title}")

    # Create a new chat session
    new_session = ChatSession(
        user_id=current_user.userid,
        title=title,
        model=request.app.state.current_model
    )

    try:
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        logger.info(f"Chat session created: {new_session.session_id}")
    except Exception as e:
        logger.error(f"Failed to create chat session: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

    return JSONResponse(
        content={"session_id": str(new_session.session_id), "title": new_session.title},
        status_code=201
    )


@router.post("/chat/message")
async def send_message(
        session_id: str,
        message: str,
        request: Request,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    logger.info(f"Processing message in session {session_id}")

    if not hasattr(request.app.state, "llm_instance"):
        logger.warning("No model instance selected before sending message")
        raise HTTPException(status_code=400, detail="No model instance selected")

    # Get session
    try:
        session = db.query(ChatSession).filter(
            ChatSession.session_id == session_id,
            ChatSession.user_id == current_user.userid
        ).first()

        if not session:
            logger.warning(f"Session {session_id} not found for user {current_user.userid}")
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error retrieving session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    # Save user message
    try:
        user_msg = Message(
            session_id=session_id,
            sender=SenderType.USER,
            content=message
        )
        db.add(user_msg)
        db.commit()
        logger.debug(f"User message saved for session {session_id}")
    except Exception as e:
        logger.error(f"Failed to save user message: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save message: {str(e)}")

    # Get LLM response
    try:
        logger.debug("Getting LLM response")
        start_time = time.time()
        llm_response = request.app.state.llm_instance.invoke(message)
        end_time = time.time()

        response_time_ms = (end_time - start_time) * 1000
        response_content = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
        logger.debug(f"LLM response received in {response_time_ms:.2f}ms")
    except Exception as e:
        logger.error(f"Error getting LLM response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting LLM response: {str(e)}")

    # Save assistant message
    try:
        assistant_msg = Message(
            session_id=session_id,
            sender=SenderType.ASSISTANT,
            content=response_content,
            model_response_time_ms=response_time_ms
        )
        db.add(assistant_msg)

        # Update session last updated timestamp
        session.updated_at = datetime.datetime.utcnow()
        db.commit()
        logger.debug(f"Assistant message saved for session {session_id}")
    except Exception as e:
        logger.error(f"Failed to save assistant message: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save response: {str(e)}")

    return JSONResponse(
        content={"response": response_content},
        status_code=200
    )


@router.post("/chat/stream")
async def stream_message(
        session_id: str,
        message: str,
        request: Request,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    logger.info(f"Processing streaming message in session {session_id}")

    if not hasattr(request.app.state, "llm_instance"):
        logger.warning("No model instance selected before streaming message")
        raise HTTPException(status_code=400, detail="No model instance selected")

    # Get session
    try:
        session = db.query(ChatSession).filter(
            ChatSession.session_id == session_id,
            ChatSession.user_id == current_user.userid
        ).first()

        if not session:
            logger.warning(f"Session {session_id} not found for user {current_user.userid}")
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error retrieving session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    # Save user message
    try:
        user_msg = Message(
            session_id=session_id,
            sender=SenderType.USER,
            content=message
        )
        db.add(user_msg)
        db.commit()
        logger.debug(f"User message saved for session {session_id}")
    except Exception as e:
        logger.error(f"Failed to save user message: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save message: {str(e)}")

    # Stream response and collect full response
    start_time = time.time()
    full_response = ""

    async def event_stream():
        nonlocal full_response
        try:
            logger.debug("Starting streaming response")
            for chunk in request.app.state.llm_instance.astream(message):
                chunk_content = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_response += chunk_content
                yield f"data: {json.dumps({'content': chunk_content})}\n\n"

            # Calculate response time
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            logger.debug(f"Stream completed in {response_time_ms:.2f}ms")

            # Save assistant message
            try:
                logger.debug("Saving streamed assistant message")
                assistant_msg = Message(
                    session_id=session_id,
                    sender=SenderType.ASSISTANT,
                    content=full_response,
                    model_response_time_ms=response_time_ms
                )

                # We need to use a new session since we're in a generator
                with SessionLocal() as new_db:
                    new_db.add(assistant_msg)
                    # Update session last updated timestamp
                    session_to_update = new_db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
                    if session_to_update:
                        session_to_update.updated_at = datetime.datetime.utcnow()
                    new_db.commit()
                    logger.debug(f"Streamed assistant message saved for session {session_id}")
            except Exception as e:
                logger.error(f"Failed to save streamed assistant message: {str(e)}")
                # Cannot raise HTTPException from here as we're in a generator
        except Exception as e:
            logger.error(f"Error in stream processing: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/chat/sessions")
async def get_sessions(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    logger.info(f"Getting sessions for user {current_user.userid}")

    try:
        sessions = db.query(ChatSession).filter(
            ChatSession.user_id == current_user.userid
        ).order_by(ChatSession.updated_at.desc()).all()

        return JSONResponse(
            content={
                "sessions": [
                    {
                        "id": str(session.session_id),
                        "title": session.title,
                        "model": session.model,
                        "created_at": session.created_at.isoformat(),
                        "updated_at": session.updated_at.isoformat()
                    }
                    for session in sessions
                ]
            },
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error retrieving sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/chat/messages/{session_id}")
async def get_session_messages(
        session_id: str,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    logger.info(f"Getting messages for session {session_id}")

    # Verify session belongs to user
    try:
        session = db.query(ChatSession).filter(
            ChatSession.session_id == session_id,
            ChatSession.user_id == current_user.userid
        ).first()

        if not session:
            logger.warning(f"Session {session_id} not found for user {current_user.userid}")
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error verifying session ownership: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    # Get messages
    try:
        messages = db.query(Message).filter(
            Message.session_id == session_id
        ).order_by(Message.timestamp).all()

        return JSONResponse(
            content={
                "messages": [
                    {
                        "id": str(msg.message_id),
                        "sender": msg.sender,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "response_time_ms": msg.model_response_time_ms
                    }
                    for msg in messages
                ]
            },
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error retrieving messages: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")