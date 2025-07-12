import logging
from typing import List

from fastapi import APIRouter, File, UploadFile, Form

from src.models.schema import Notes
from src.service.rag_service import GitRequest, git_handler, get_handler

router = APIRouter()

logger = logging.getLogger("rag")

@router.post("/git")
def git_rag(req: GitRequest, session_id: str, context_id: str):
    res = git_handler(req=req,
                      session_id=session_id
                      , context_id=context_id
                      , context_type="code")
    return res


@router.post("/upload")
async def git_rag(
    files: List[UploadFile] = File(...),
    session_id: str = Form(...),
    context_id: str = Form(...),
    context_type: str = Form(...)
):

    logger.info(f"Got Content:{files} {session_id} {context_id} {context_type}")
    res = await get_handler(
        req=files,
        session_id=session_id,
        context_id=context_id,
        context_type=context_type  # use passed value
    )
    return res
