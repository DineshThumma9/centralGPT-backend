# src/app/router/basic.py
import os
from fastapi import HTTPException, Query, APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from src.app import app  # Import from centralized location
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

router = APIRouter()

llms = {
    "groq": ChatGroq,
    "ollama": ChatOllama,
}

@router.post("/api/{api_key}")
def get_api_key(api_key: str):
    os.environ["GROQ_API_KEY"] = api_key

@router.get("/models/{llm_prov}")
def choose_llm_provider(llm_prov: str):
    llm_class = llms.get(llm_prov)
    if not llm_class:
        raise HTTPException(status_code=404, detail=f"LLM provider '{llm_prov}' not found")

    app.state.llm_class = llm_class
    return JSONResponse(
        content={"message": f"LLM provider '{llm_prov}' selected successfully"},
        status_code=200
    )

@router.get("/models/model/{model}")
def choose_model(model: str):
    llm_class = getattr(app.state, "llm_class", None)
    if llm_class is None:
        raise HTTPException(status_code=400, detail="No LLM provider selected")

    try:
        llm_instance = llm_class(model=model)
        app.state.llm_instance = llm_instance
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to instantiate model: {str(e)}")

    return JSONResponse(
        content={"message": f"Model '{model}' instantiated successfully"},
        status_code=200
    )

@router.api_route("/chat", methods=["GET", "POST"])
def getResponse(message: str = Query(...)):
    if not hasattr(app.state, "llm_instance"):
        raise HTTPException(status_code=400, detail="No model instance selected")

    response = app.state.llm_instance.invoke(message)

    if hasattr(response, "content"):
        return JSONResponse(
            content={"response": response.content},
            status_code=200
        )
    else:
        return JSONResponse(
            content={"response": str(response)},
            status_code=200
        )

@router.post("/chat/stream")
def streamMsg(query: str):
    def event_stream():
        for chunk in app.state.llm_instance.astream(query):
            yield chunk
    return StreamingResponse(event_stream(), media_type="text/plain")

