import logging
from collections import Counter
from typing import Optional, List

from fastapi import HTTPException
from langchain_core.vectorstores import VectorStoreRetriever
from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.node_parser import CodeSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import BaseModel
from starlette.responses import JSONResponse

from src.db.qdrant_client import qdrant_client
from src.service.message_service import system_prompt, rag_prompt

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="llamaIndex"
)


logger = logging.getLogger("rag")

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


chat_engine = ChatEngineManager()

class GitRequest(BaseModel):
    owner: str
    repo: str
    commit: Optional[str] = None
    branch: Optional[str] = "main"
    dir_include: Optional[List[str]] = None
    dir_exclude: Optional[List[str]] = None
    file_extension_include: Optional[List[str]] = None
    file_extension_exclude: Optional[List[str]] = None





def get_documents(req:GitRequest):
    documents = GithubRepositoryReader(
        github_client=GithubClient(verbose=True),
        owner=req.owner,
        repo=req.repo,
        filter_directories=(req.dir_include or ["src"], GithubRepositoryReader.FilterType.INCLUDE),
        filter_file_extensions=([
                                    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
                                    ".json", ".ipynb", ".lock", ".md"
                                ], GithubRepositoryReader.FilterType.EXCLUDE),
        use_parser=False,
        verbose=True
    ).load_data(branch="main")


    if not documents:
        raise HTTPException(status_code=400, detail="No documents found")

    return documents


def get_nodes(documents,laguage):
    return CodeSplitter(
        language=laguage
    ).get_nodes_from_documents(documents)



def get_index(nodes,embed_model):
    return VectorStoreIndex(
        nodes=nodes,
        show_progress=True,
        embed_model=embed_model

    )


def get_embed_model():
    return HuggingFaceEmbedding(
        model_name="Qodo/Qodo-Embed-1-1.5B"
    )



def get_retriever(index,embed_model):
    return VectorIndexRetriever(
        index=index,
        embed_model=embed_model,
        similarity_top_k=5,
        node_postprocessors=[CohereRerank(top_n=3)]
    )



def build_file_tree(file_entries):
    """
    file_entries: list of dicts with keys: 'file_path' and 'url'
    Returns a nested dictionary representing the folder structure.
    """
    tree = {}

    for entry in file_entries:
        path_parts = entry["file_path"].strip("/").split("/")
        current = tree

        for i, part in enumerate(path_parts):
            if i == len(path_parts) - 1:
                # Leaf node = file, add metadata
                current[part] = {"url": entry["url"]}
            else:
                current = current.setdefault(part, {})

    return tree



def git_handler(req:GitRequest,llm,memory,session_id,context_id,context_type):
    documents = get_documents(req)
    counter = Counter()
    files_info = []
    for dic in documents:
        counter[dic.extra_info["file_name"].split(".")[1]] = counter.get(dic.extra_info["file_name"].split(".")[1],0)+1
        files_info.append(dic.extra_info)
    max_val = max(counter.values())
    language = max(key for key,val in counter.items() if val==max_val)
    tree = build_file_tree(files_info)
    nodes = get_nodes(documents,laguage=language)
    embed_model = get_embed_model()
    index = get_index(nodes,embed_model)
    retriever = get_retriever(index,embed_model=embed_model)

    engine =  ContextChatEngine.from_defaults(
        retriever=retriever,
        llm=llm,
        system_prompt=rag_prompt,
        vector_store=vector_store,
        memory=memory,


    )

    chat_engine.set_engine(session_id=session_id,context_type=context_type,context_id=context_id,engine=engine)

    return JSONResponse(
        status_code=200,
        content="Engine has been Set"
    )


