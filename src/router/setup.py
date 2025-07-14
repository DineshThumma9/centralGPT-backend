import logging
from typing import Dict

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Body
from fastapi import Depends

from src.db import get_db
from src.models.schema import APIKEYS, UserLLMConfig, API_KEY_REQUEST
from src.router.auth import get_current_user
from src.service.set_up_service import api_providers, encrypt, llm_providers

logger = logging.getLogger("basic_router")
load_dotenv()
router = APIRouter()


@router.post("/init/")
def set_api_provider(
        req: API_KEY_REQUEST,
        current_user=Depends(get_current_user),
        db=Depends(get_db)
):
    api_provider = req.api_prov.upper().strip()
    api_key = req.api_key.strip()

    logger.info(f"API KEY AND PROVIDER ARE : {api_provider} & {api_key}")

    if api_provider not in api_providers:
        raise HTTPException(status_code=404, detail="api provider doesnt exists")

    encrypted_key = encrypt(api_key)

    existing = (
        db.query(APIKEYS).filter_by(user_id=current_user.userid, provider=api_provider).first()
    )

    if existing:
        existing.encrypted_key = encrypted_key
        logger.info("Saved / Updated new key", encrypted_key)
    else:
        new_key = APIKEYS(
            user_id=current_user.userid,
            provider=api_provider,
            encrypted_key=encrypted_key
        )

        db.add(new_key)
        logger.info(f"Added new key  {new_key}")
    db.commit()
    return {
        "message": "Succesfully key added",
        "status_code": 200
    }


@router.post("/providers")
async def choose_llm_provider(
        body: Dict = Body(...),
        db=Depends(get_db),
        user=Depends(get_current_user)
):
    provider = body.get("provider")

    logger.info(f"provider is {provider}")
    if not provider:
        raise HTTPException(status_code=400, detail="Provider is required")

    provider = provider.strip()
    if not provider:
        raise HTTPException(status_code=400, detail="Provider cannot be empty")

    if provider.lower() not in llm_providers:
        raise HTTPException(status_code=404, detail="Provider not supported")

    config = db.query(UserLLMConfig).filter_by(user_id=user.userid).first()

    if config:
        config.provider = provider.lower()
    else:
        config = UserLLMConfig(
            user_id=user.userid,
            provider=provider,
            model=""
        )

        db.add(config)
    db.commit()

    return {
        "message": f"provider choosed success fully {config}"
    }


@router.post("/models")
async def choose_model(
        body: Dict = Body(...),
        db=Depends(get_db),
        user=Depends(get_current_user)
):
    model = body.get("model")

    if not model:
        raise HTTPException(status_code=400, detail="Model is required")

    model = model.strip()
    if not model:
        raise HTTPException(status_code=400, detail="Model cannot be empty")

    config = db.query(UserLLMConfig).filter_by(user_id=user.userid).first()

    if not config:
        raise HTTPException(status_code=404, detail=f"Config not found {config}")

    config.model = model
    db.commit()

    return {
        "message": f"Model has been set to {model}"
    }
