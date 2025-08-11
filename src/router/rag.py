import logging
from typing import List, Optional

from fastapi import APIRouter, File, UploadFile, Form
from pydantic import BaseModel

from src.models.schema import git_spec
from src.service.rag_service import GitRequest, git_handler, get_handler, get_dir_struct

router = APIRouter()

logger = logging.getLogger("rag")


@router.post("/git")
async def git_rag(req: GitRequest, session_id: str, context_id: str):
    res =  await git_handler(req=req,
                      session_id=session_id
                      , context_id=context_id
                      , context_type="code")
    return res


@router.post("/upload")
async def get_rag(
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



@router.post("/tree")
async def get_tree(
        reques:git_spec
):

    logger.info(f"Request Which has recived from fonted : {reques}")
    tree = get_dir_struct(reques)
    return tree



