from src.db.dbs import logger


class ChatEngineManager:
    def __init__(self):
        self.engines = {}

    def set_engine(self, context_type: str, session_id: str, context_id: str, engine):
        key = f"{session_id}_{context_id}_{context_type}"
        logger.info(f"Setting engine with key: {key}")
        self.engines[key] = engine
        return key

    def get_engine(self, session_id: str, context_type: str, context_id: str):
        key = f"{session_id}_{context_id}_{context_type}"
        return self.engines.get(key)

