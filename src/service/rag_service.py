import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import HTTPException, UploadFile, BackgroundTasks
from github import Auth, Github
from llama_index.core import SimpleDirectoryReader, Document, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from pydantic import BaseModel
from starlette.responses import JSONResponse

from src.db.qdrant_client import get_vector_store
from src.db.redis_client import redis_client
from src.models.schema import GitRequest

load_dotenv()

logger = logging.getLogger(__name__)

auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
g = Github(auth=auth)


class FileType(BaseModel):
    name: str
    content: bytes
    type: str


EXT_MAP = {
    ".py": "Python",
    ".js": "JavaScript",
    ".java": "Java",
    ".ts": "TypeScript",
    ".c": "C",
    ".cpp": "C++",
    ".cs": "C#",
    ".go": "Go",
    ".rb": "Ruby",
    ".php": "PHP",
    ".jsx": "JavaScript",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".rs": "Rust",
    ".sh": "Shell",
    ".html": "HTML",
    ".css": "CSS",
    ".json": "JSON",
    ".md": "Markdown"
}

code_embed_model = HuggingFaceEmbedding(
    model_name="jinaai/jina-embeddings-v2-base-code",
    trust_remote_code=True,
    embed_batch_size=32,

)
notes_embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    trust_remote_code=True,
    embed_batch_size=64,

)


async def git_documents(req: GitRequest):
    repo = g.get_repo(f"{req.owner}/{req.repo}")
    langs = repo.get_languages()

    if req.branch:
        sha = repo.get_branch(req.branch).commit.sha
    elif req.commit:
        sha = repo.get_commit(req.commit).sha
    else:
        sha = repo.get_branch(repo.default_branch).commit.sha

    documents = await async_github_reader(owner=req.owner,
                                          repo=req.repo,
                                          dir_include=req.dir_include,
                                          commit=req.commit,
                                          branch=req.branch)

    try:
        read_me = repo.get_readme()
        readme = Document(
            doc_id=read_me.sha,
            text=read_me.decoded_content.decode('utf-8') if isinstance(read_me.decoded_content,
                                                                       bytes) else read_me.decoded_content,
            metadata={"path": read_me.path, "size": read_me.size}
        )
        documents.append(readme)
    except Exception as e:
        logger.warning(f"README not found or not accessible: {e}")

    if not documents:
        raise HTTPException(status_code=400, detail="No documents found")

    return {"docs": documents, "langs": langs, "sha": sha}





async def async_github_reader(owner, repo, dir_include, branch, commit):
    loop = asyncio.get_event_loop()

    return await loop.run_in_executor(None,
                                      lambda: GithubRepositoryReader(
                                          github_client=GithubClient(verbose=True),
                                          owner=owner,
                                          repo=repo,
                                          concurrent_requests=10,
                                          retries=2,
                                          logger=logger,
                                          filter_directories=(
                                              dir_include or ["src"], GithubRepositoryReader.FilterType.INCLUDE),
                                          filter_file_extensions=([
                                                                      ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
                                                                      ".json", ".ipynb", ".lock", ".md"
                                                                  ], GithubRepositoryReader.FilterType.EXCLUDE),
                                          use_parser=False,

                                          custom_folder=f"{owner}_{repo}_{branch}"
                                      ).load_data(branch=branch, commit_sha=commit)
                                      )




def get_documents(files: List[FileType]):
    temp_dir = tempfile.mkdtemp()

    try:
        for file in files:

            if not file.content:
                logger.warning(f"Empty file: {file.name}")
                continue

            temp_path = Path(temp_dir) / file.name
            with open(temp_path, 'wb') as f:
                f.write(file.content)

        if temp_dir and os.listdir(temp_dir):
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


def get_nodes(documents: List[Document], is_code: bool,
              language: Optional[str] = None):
    if is_code:

        try:

            splitter = CodeSplitter(language=language)
            nodes = splitter.get_nodes_from_documents(documents)

        except Exception as e:

            logger.error(f"Error has occured : {e}")
            pipeline = IngestionPipeline(transformations=[

                SentenceSplitter(
                    chunk_size=512,
                    chunk_overlap=128
                )

            ])
            logger.info("Due to error we are splitting using pipeline")
            nodes = pipeline.run(documents=documents)

        return nodes

    else:

        pipeline = IngestionPipeline(transformations=[
            SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=code_embed_model
            )
        ])
        nodes = pipeline.run(documents=documents)
        return nodes






def get_or_build_index(nodes, is_code: bool, session_id: str, context_id: str, context_type: str):
    """
    Builds the index and PERSISTS it to the vector store.
    """
    collection_name = f"{session_id}_{context_id}_{context_type}"
    logger.info(f"Using collection: {collection_name} for persistence.")


    vector_store = get_vector_store(collection_name)


    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=code_embed_model if is_code else notes_embed_model,
        show_progress=True,
    )
    return index




def build_tree(tree_lis):
    root = {"name": "/", "type": "tree", "children": []}
    path_map = {"/": root}

    for item in tree_lis:
        parts = item["path"].split("/")
        for i in range(1, len(parts) + 1):
            sub_path = "/".join(parts[:i])
            if sub_path not in path_map:
                parent_path = "/".join(parts[:i - 1]) or "/"
                parent = path_map[parent_path]
                node_type = "tree" if i < len(parts) else item["type"]
                node = {
                    "name": parts[i - 1],
                    "path": sub_path,
                    "type": node_type,
                    "children": [] if node_type == "tree" else None,
                }
                if node_type == "blob":
                    node["sha"] = item.get("sha")
                    node["size"] = item.get("size")
                parent["children"].append(node)
                path_map[sub_path] = node

    return root["children"]





def get_dir_struct(req):
    repo = g.get_repo(f"{req.owner}/{req.repo}")
    tree_sha = repo.default_branch
    tree = repo.get_git_tree(sha=tree_sha, recursive=True)
    lis = []
    for item in tree.tree:
        lis.append({
            "type": item.type,
            "path": item.path,
            "size": item.size,
            "sha": item.sha
        })
    return build_tree(lis)


def get_specific_files(files: List[str], owner: str, repo: str):
    return get_repo_data(owner, repo, files)


def get_repo_data(owner: str, repo: str, files: List[str]):
    from collections import defaultdict

    repo = g.get_repo(f"{owner}/{repo}")
    sha = repo.get_branch(repo.default_branch).commit.sha
    docs, langs = [], defaultdict(int)

    for fil in files:
        content = repo.get_contents(path=fil)
        ext = os.path.splitext(fil)[1].lower()
        file_lang = EXT_MAP.get(ext, "Unknown")
        langs[file_lang] += 1
        docs.append(
            Document(
                doc_id=content.sha,
                text=content.decoded_content.decode('utf-8') if isinstance(content.decoded_content,
                                                                           bytes) else content.decoded_content,
                metadata={"path": fil, "size": content.size}
            )
        )

    return {
        "docs": docs,
        "langs": langs,
        "sha": sha
    }


async def build_index(req: List[FileType], session_id, context_id, context_type):
    documents = get_documents(req)
    nodes = get_nodes(documents=documents, is_code=False)

    get_or_build_index(
        nodes=nodes,
        is_code=False,
        session_id=session_id,
        context_id=context_id,
        context_type=context_type
    )

    collection_name = f"{session_id}_{context_id}_{context_type}"
    await redis_client.set(f"collection_name:{collection_name}:status", "ready",ex=600)


async def to_files(files: List[UploadFile]):
    converted = []

    for file in files:

        try:

            content = await file.read()

            if not content:
                logger.error(f"File has no contents : {file.filename}")
                continue

            f = FileType(
                name=file.filename,
                content=content,
                type=file.content_type
            )

            converted.append(f)

            await file.seek(0)

        except Exception as e:
            logger.error(f"Error has occured some  : {e}")
            continue

    return converted


async def get_handler(background_tasks: BackgroundTasks, req: List[UploadFile], session_id, context_id, context_type):
    collection_name = f"{session_id}_{context_id}_{context_type}"
    await redis_client.set(f"collection_name:{collection_name}:status", "indexing",ex=600)
    files = await to_files(req)
    background_tasks.add_task(build_index, files, session_id, context_id, context_type)

    return JSONResponse(status_code=200, content={"message": "Documents are being indexed", "status": "indexing"})


async def git_handler(req: GitRequest, session_id, context_id, context_type):
    if req.files:
        git_dic = get_specific_files(req.files, req.owner, req.repo)
    else:
        git_dic = await git_documents(req)

    documents = git_dic["docs"]
    langs = git_dic["langs"]
    language = max(langs, key=langs.get) if langs else "Unknown"

    nodes = get_nodes(documents=documents, language=language.lower(), is_code=True)

    get_or_build_index(
        nodes=nodes,
        is_code=True,
        session_id=session_id,
        context_type=context_type,
        context_id=context_id
    )

    collection_name = f"{session_id}_{context_id}_{context_type}"
    await redis_client.set(f"collection_name:{collection_name}:status", "ready",ex=600)

    return JSONResponse(status_code=200, content={"message": "Engine has been Set", "status": "ready"})
