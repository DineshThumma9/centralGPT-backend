import qdrant_client
from fastapi import APIRouter, Depends
from llama_index.core import StorageContext

from src.db import get_db
from src.router.messages import get_memory
from src.service.auth_service import get_current_user
from src.service.rag_service import GitRequest, git_handler
from src.service.set_up_service import get_llm_instance

router = APIRouter()


@router.post("/git")
def git_rag(req:GitRequest,session_id:str,context_id:str,db=Depends(get_db),user=Depends(get_current_user)):
    res = git_handler(req=req,
                         llm=get_llm_instance(db=db,user=user),
                         memory=get_memory(session_id=session_id),
                         session_id=session_id
                         ,context_id=context_id
                         ,context_type="code")
    return res