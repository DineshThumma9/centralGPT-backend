import logging
from typing import Dict

from cryptography.fernet import Fernet;
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Body
from fastapi import Depends
from langchain_community.chat_models import ChatDeepInfra
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from langchain_together import ChatTogether
from pydantic import BaseModel

from src.db import get_db
from src.models.schema import APIKEYS, UserLLMConfig
from src.router.auth import get_current_user

logger = logging.getLogger("basic_router")
load_dotenv()
router = APIRouter()
llm_instances = {}

llm_providers = {
    "ollama": ChatOllama,
    "groq": ChatGroq,
    "mistral": ChatMistralAI,
    "deepinfra": ChatDeepInfra,
    "together": ChatTogether,

}

api_providers = {
    "GROQ",
    "OPENAI",
    "GROK",
    "TOGETHER",
    "DEEPSEEK",
    "DEEPINFRA"
    "OPENROUTER",
    "MISTRAL"
    "DEEPINFRA",
    "TOGETHERAI"

}


class API_KEY_REQUEST(BaseModel):
    api_prov: str
    api_key: str




fernet = Fernet("d3FVcotBFzBnqZ4BE0zlgji_YYZiK5hkDO3EzX9H7fs=")

def encrypt(key:str) -> str:
    return fernet.encrypt(key.encode()).decode()

def decrypt(key:str)->str:
    return fernet.decrypt(key.encode()).decode()


# Fix for the set_api_provider function
@router.post("/init/")
def set_api_provider(
        req: API_KEY_REQUEST,
        current_user=Depends(get_current_user),
        db=Depends(get_db)
):
    api_provider = req.api_prov.upper().strip()
    api_key = req.api_key.strip()

    if api_provider not in api_providers:
        raise HTTPException(status_code=404, detail="api provider doesnt exists")

    encrypted_key = encrypt(api_key)

    # FIXED: Use filter_by with keyword arguments only
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
        logger.info("Added new key  new_key")
    db.commit()
    return {
        "message": "Succesfully key added",
        "status_code": 200
    }


# You also have the same issue in other places. Here are the other fixes:

def get_api_key(provider: str, db=Depends(get_db), user=Depends(get_current_user)):
    # FIXED: Use filter_by with keyword arguments
    api_key = db.query(APIKEYS).filter_by(user_id=user.userid, provider=provider).first()
    if not api_key:
        raise HTTPException(status_code=404, detail="API KEY NOT FOUND")
    return decrypt(api_key.encrypted_key)


def get_llm_instance(db=Depends(get_db), user=Depends(get_current_user)):
    # FIXED: Use filter_by with keyword arguments
    config = db.query(UserLLMConfig).filter_by(user_id=user.userid).first()

    logger.info(config)
    if not config:
        raise HTTPException(status_code=404, detail="Config insnt Setup")

    # FIXED: Use filter_by with keyword arguments
    api_record = db.query(APIKEYS).filter_by(user_id=user.userid, provider=config.provider.upper()).first()

    if not api_record:
        raise HTTPException(status_code=404, detail="API KEY ISNT SET")

    decrypted_key = decrypt(api_record.encrypted_key)

    logger.info(f"API RECORD IS  {api_record}")
    logger.info(f"DECRPTED KEY  , {decrypted_key}")

    llm_class = llm_providers.get(config.provider.lower())

    logger.info(f"llms class not found {llm_class}")
    logger.info(f"llm decropedt key is   {decrypted_key}")

    if not llm_class:
        raise HTTPException(status_code=404, detail="NOT AN LLM CLass found")

    if not decrypted_key:
        raise HTTPException(status_code=404, detail="DECRPYTED KEY DOESNT EXIST")

    return llm_class(model=config.model, api_key=decrypted_key)


@router.post("/providers")
async def choose_llm_provider(
        body: Dict = Body(...),
        db=Depends(get_db),
        user=Depends(get_current_user)
):
    provider = body.get("provider")

    # Check if provider is None or empty

    logger.info(f"provider is {provider}")
    if not provider:
        raise HTTPException(status_code=400, detail="Provider is required")

    # Strip whitespace and convert to lowercase
    provider = provider.strip()
    if not provider:
        raise HTTPException(status_code=400, detail="Provider cannot be empty")

    if provider.lower() not in llm_providers:
        raise HTTPException(status_code=404, detail="Provider not supported")

    # FIXED: Use filter_by with keyword arguments
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

    # Check if model is None or empty
    if not model:
        raise HTTPException(status_code=400, detail="Model is required")

    # Strip whitespace
    model = model.strip()
    if not model:
        raise HTTPException(status_code=400, detail="Model cannot be empty")

    # FIXED: Use filter_by with keyword arguments
    config = db.query(UserLLMConfig).filter_by(user_id=user.userid).first()

    if not config:
        raise HTTPException(status_code=404, detail=f"Config not found {config}")

    config.model = model
    db.commit()

    return {
        "message": f"Model has been set to {model}"
    }