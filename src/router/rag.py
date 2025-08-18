# Fixed rag.py

import logging
from typing import List

from fastapi import APIRouter, File, UploadFile, Form
from fastapi import BackgroundTasks
from fastapi import HTTPException

from src.db.redis_client import redis_client
from src.models.schema import git_spec
from src.service.rag_service import GitRequest, git_handler, get_handler, get_dir_struct

router = APIRouter()

logger = logging.getLogger("rag")


@router.post("/git")
async def git_rag(req: GitRequest, session_id: str, context_id: str):
    res = await git_handler(req=req,
                      session_id=session_id
                      , context_id=context_id
                      , context_type="code")
    return res


@router.post("/upload")
async def get_rag(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(...),
        session_id: str = Form(...),
        context_id: str = Form(...),
        context_type: str = Form(...)
):
    logger.info(f"Got Content:{files} {session_id} {context_id} {context_type}")
    logger.info("In Upload Endpoint")
    res = await get_handler(
        background_tasks,
        req=files,
        session_id=session_id,
        context_id=context_id,
        context_type=context_type
    )
    return res


@router.post("/tree")
async def get_tree(
        reques: git_spec
):
    logger.info(f"Request Which has received from frontend : {reques}")
    tree = get_dir_struct(reques)
    return tree


@router.get("/status")
async def get_status(session_id: str, context_id: str, context_type: str):
    try:
        collection_name = f"{session_id}_{context_id}_{context_type}"
        key = f"collection_name:{collection_name}:status"
        status = await redis_client.get(key)

        if not session_id or not context_id or not context_type:
            raise HTTPException(status_code=400,detail="Error has occured some params are missing")




        if status is None:
            return {"status": "missing", "collection_name": collection_name}



        if isinstance(status, bytes):
            status = status.decode()

        logger.info(f"Status for {collection_name}: {status}")
        return {"status": status, "collection_name": collection_name}
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Error getting status: {e}")
        return {"error": str(e), "status": "error"}