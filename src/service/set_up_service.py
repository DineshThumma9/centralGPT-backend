import logging

from cryptography.fernet import Fernet;
from fastapi import Depends
from fastapi import HTTPException
from llama_index.llms.deepinfra import DeepInfraLLM
from llama_index.llms.groq import Groq
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.together import TogetherLLM

from src.db import get_db
from src.models.schema import APIKEYS, UserLLMConfig
from src.router.auth import get_current_user

llm_instances = {}

llm_providers = {
    "ollama": Ollama,
    "groq": Groq,
    "mistral": MistralAI,
    "together": TogetherLLM,

}

api_providers = {
    "GROQ",
    "OPENAI",
    "GROK",
    "TOGETHER",
    "DEEPSEEK",
    "OPENROUTER",
    "MISTRAL",
    "TOGETHERAI"

}

logger = logging.getLogger("set_up_service")

fernet = Fernet("d3FVcotBFzBnqZ4BE0zlgji_YYZiK5hkDO3EzX9H7fs=")


def encrypt(key: str) -> str:
    return fernet.encrypt(key.encode()).decode()


def decrypt(key: str) -> str:
    return fernet.decrypt(key.encode()).decode()


def get_api_key(provider: str, db=Depends(get_db), user=Depends(get_current_user)):
    api_key = db.query(APIKEYS).filter_by(user_id=user.userid, provider=provider).first()
    if not api_key:
        raise HTTPException(status_code=404, detail="API KEY NOT FOUND")
    return decrypt(api_key.encrypted_key)


def get_llm_instance(db=Depends(get_db), user=Depends(get_current_user)):
    config = db.query(UserLLMConfig).filter_by(user_id=user.userid).first()

    logger.info(config)

    if not config:
        raise HTTPException(status_code=404, detail="Config is'nt Setup")

    api_record = db.query(APIKEYS).filter_by(user_id=user.userid, provider=config.provider.upper()).first()

    if not api_record:
        raise HTTPException(status_code=404, detail="API KEY ISNT SET")

    decrypted_key = decrypt(api_record.encrypted_key)

    logger.info(f"API RECORD IS  {api_record}")
    logger.info(f"DECRYPTED KEY  , {decrypted_key}")

    llm_class = llm_providers.get(config.provider.lower())

    logger.info(f"llms class not found {llm_class}")
    logger.info(f"llm decropedt key is   {decrypted_key}")

    if not llm_class:
        raise HTTPException(status_code=404, detail="NOT AN LLM CLass found")

    if not decrypted_key:
        raise HTTPException(status_code=404, detail="DECRPYTED KEY DOESNT EXIST")

    return llm_class(model=config.model, api_key=decrypted_key)
