import logging
from collections import Counter
from typing import Optional, List

from dotenv import load_dotenv
from fastapi import HTTPException, UploadFile
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, StorageContext
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.llms.together import TogetherLLM
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from pydantic import BaseModel
from starlette.responses import JSONResponse

from src.models.schema import Notes

load_dotenv()

EXT_TO_LANG = {
    # Python
    "py": "python",
    # JavaScript / TypeScript
    "js": "javascript",
    "ts": "typescript",
    "jsx": "javascript",
    "tsx": "typescript",
    # HTML/CSS
    "html": "html",
    "css": "css",
    # Java
    "java": "java",
    # C/C++
    "c": "c",
    "h": "c",
    "cpp": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "hpp": "cpp",
    # C#
    "cs": "csharp",
    # PHP
    "php": "php",
    # Ruby
    "rb": "ruby",
    # Go
    "go": "go",
    # Rust
    "rs": "rust",
    # Kotlin
    "kt": "kotlin",
    "kts": "kotlin",
    # Swift
    "swift": "swift",
    # Shell / Bash
    "sh": "bash",
    "bash": "bash",
    # PowerShell
    "ps1": "powershell",
    # Scala
    "scala": "scala",
    # Dart
    "dart": "dart",
    # R
    "r": "r",
    # Julia
    "jl": "julia",
    # SQL
    "sql": "sql",
    # YAML/JSON config
    "yaml": "yaml",
    "yml": "yaml",
    "json": "json",
    # Markdown / Docs
    "md": "markdown",
    "rst": "markdown",
    # Text
    "txt": "text",
    # Docker / CI
    "dockerfile": "docker",
    "env": "text",
    # Make
    "makefile": "make",
}


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





def git_documents(req:GitRequest):
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


import tempfile
import shutil
from pathlib import Path


async def get_documents(files: List[UploadFile]):
    documents = []
    temp_dir = tempfile.mkdtemp()

    try:
        for file in files:
            # Don't seek - just read the content
            content = await file.read()

            if not content:
                logger.warning(f"Empty file: {file.filename}")
                continue

            temp_path = Path(temp_dir) / file.filename

            with open(temp_path, 'wb') as f:
                f.write(content)

        # Use SimpleDirectoryReader with proper error handling
        if temp_dir:
            try:
                documents = SimpleDirectoryReader(
                    input_dir=temp_dir,
                    recursive=True,

                ).load_data()
            except Exception as e:
                logger.error(f"Error reading documents: {e}")
                raise HTTPException(400, f"Error processing documents: {e}")

    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing files: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if not documents:
        raise HTTPException(400, "No documents were successfully processed")

    return documents




def get_nodes(documents: List[Document], is_code: bool,language:Optional[str]=None):
    """Process documents into nodes"""
    if is_code:
        from llama_index.core.node_parser import CodeSplitter
        splitter = CodeSplitter(
            language=language
        )
        return splitter.get_nodes_from_documents(documents)
    else:
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        pipeline = IngestionPipeline(transformations=[
            SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=embed_model
            )

        ])
        return pipeline.run(documents=documents)








def get_index(nodes, embed_model):

    return VectorStoreIndex(
        nodes=nodes,
        embed_model=embed_model,

    )



def get_embed_model(is_code:bool):
    return HuggingFaceEmbedding(
        model_name="nomic-ai/nomic-embed-text-v1" if is_code else "BAAI/bge-small-en-v1.5",
        trust_remote_code=True
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



def git_handler(req:GitRequest,session_id,context_id,context_type):
    documents = git_documents(req)
    counter = Counter()
    files_info = []
    for dic in documents:
        counter[dic.extra_info["file_name"].split(".")[1]] = counter.get(dic.extra_info["file_name"].split(".")[1],0)+1
        files_info.append(dic.extra_info)
    max_val = max(counter.values())
    language = max(key for key,val in counter.items() if val==max_val)
    tree = build_file_tree(files_info)
    nodes = get_nodes(documents,language=EXT_TO_LANG[language],is_code=True)
    embed_model = get_embed_model(is_code=True)
    index = get_index(nodes,embed_model)
    retriever = get_retriever(index,embed_model=embed_model)


    chat_engine.set_engine(session_id=session_id,context_type=context_type,context_id=context_id,engine=retriever)

    return JSONResponse(
        status_code=200,
        content="Engine has been Set"
    )



async def get_handler(req:List[UploadFile],session_id,context_id,context_type):
    documents = await get_documents(req)
    nodes = get_nodes(documents,is_code=False)
    embed_model = get_embed_model(is_code=False)
    index = get_index(nodes,embed_model)
    retriever = get_retriever(index,embed_model=embed_model)


    chat_engine.set_engine(session_id=session_id,context_type=context_type,context_id=context_id,engine=retriever)

    return JSONResponse(
        status_code=200,
        content="Engine has been Set For notes"
    )

