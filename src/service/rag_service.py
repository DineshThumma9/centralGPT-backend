
import asyncio
import logging
import os
import shutil
import tempfile
from asyncio import to_thread
from pathlib import Path
from typing import List
from typing import Optional

from dotenv import load_dotenv
from fastapi import HTTPException, UploadFile
from fastapi.logger import logger
from github import Auth, Github
from llama_index.core import SimpleDirectoryReader, Document, StorageContext
from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.redis import RedisVectorStore
from redisvl.schema import IndexSchema
from starlette.responses import JSONResponse

from src.db.redis_client import get_doc_store, get_index_store, \
    get_vector_store  # adjust import as needed
from src.models.schema import GitRequest
from src.service.chat_engine_manager import ChatEngineManager

load_dotenv()


logger = logging.getLogger(__name__)

auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
g = Github(auth=auth)



# Simple extension-to-language mapping
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
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".rs": "Rust",
    ".sh": "Shell",
    ".html": "HTML",
    ".css": "CSS",
    ".json": "JSON",
    ".md": "Markdown"
}


chat_engine = ChatEngineManager()



async def git_documents(req: GitRequest,storage_context):
    """Fixed async version of git_documents"""
    repo = g.get_repo(f"{req.owner}/{req.repo}")
    langs = repo.get_languages()

    if req.branch:
        sha = repo.get_branch(req.branch).commit.sha
    elif req.commit:
        sha = repo.get_commit(req.commit).sha
    else:
        sha = repo.get_branch(repo.default_branch).commit.sha

    # Fix: Run GitHub reader in thread pool to avoid event loop conflicts
    documents = await to_thread(
        _sync_github_reader,
        req.owner,
        req.repo,
        req.dir_include,
        req.branch,
        req.commit
    )

    # Try to add README but don't fail if it doesn't exist
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

    # Check and add new documents
    new_docs = []
    for doc in documents:
        if not await storage_context.docstore.adocument_exists(doc.doc_id):
            new_docs.append(doc)

    if new_docs:
        await storage_context.docstore.async_add_documents(new_docs)

    if not documents:
        raise HTTPException(status_code=400, detail="No documents found")

    return {"docs": documents, "langs": langs, "sha": sha}


def _sync_github_reader(owner, repo, dir_include, branch, commit):
    """Synchronous GitHub reader to run in thread"""
    return GithubRepositoryReader(
        github_client=GithubClient(verbose=True),
        owner=owner,
        repo=repo,
        filter_directories=(dir_include or ["src"], GithubRepositoryReader.FilterType.INCLUDE),
        filter_file_extensions=([
                                    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
                                    ".json", ".ipynb", ".lock", ".md"
                                ], GithubRepositoryReader.FilterType.EXCLUDE),
        use_parser=False,
        verbose=True
    ).load_data(branch=branch, commit_sha=commit)


async def get_documents(files: List[UploadFile]):
    documents = []
    temp_dir = tempfile.mkdtemp()

    try:
        for file in files:
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


async def get_nodes(documents: List[Document], is_code: bool, storage_context:StorageContext,language: Optional[str] = None):
    """Process documents into nodes - Fixed async handling"""

    # Fix: Process documents that DON'T exist yet
    new_docs = []
    for doc in documents:
        if not await storage_context.docstore.adocument_exists(doc_id=doc.doc_id):
            new_docs.append(doc)

    if not new_docs:
        # All documents already processed, load existing nodes
        all_nodes = []
        for doc in documents:
            try:
                doc_nodes = await storage_context.docstore.aget_nodes(doc.doc_id)
                all_nodes.extend(doc_nodes)
            except:
                pass
        return all_nodes

    if is_code:
        from llama_index.core.node_parser import CodeSplitter
        splitter = CodeSplitter(language=language)

        # Fix: Run node processing in thread to avoid blocking
        nodes = await to_thread(splitter.get_nodes_from_documents, new_docs)

        # Store both documents and nodes
        await storage_context.docstore.async_add_documents(new_docs)
        # Fix: Store nodes, not documents
        for node in nodes:
            await storage_context.docstore.async_add_documents([node])

        return nodes
    else:
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        pipeline = IngestionPipeline(transformations=[
            SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=embed_model
            )
        ])

        # Fix: Run pipeline in thread
        nodes = await to_thread(pipeline.run, documents=new_docs)

        # Store documents and nodes properly
        await storage_context.docstore.async_add_documents(new_docs)
        for node in nodes:
            await storage_context.docstore.async_add_documents([node])

        return nodes


def get_repo_index_id(owner, repo, commit_sha):
    return f"{owner}_{repo}_{commit_sha}_code"



async def get_or_build_index(index_id: Optional[str], nodes, is_code: bool, storage_context,language=None):
    """Async-safe index creation/loading with support for Redis/Qdrant without forcing BGSAVE"""
    embed_model = get_embed_model(is_code)

    if is_code and index_id:
        try:

            existing_structs = await storage_context.index_store.async_index_structs()
            if index_id in existing_structs:
                logger.info(f"[INDEX STORE HIT] {index_id}")
                index = await asyncio.to_thread(load_index_from_storage, storage_context, index_id)
                return index
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")

        logger.info(f"[INDEX STORE MISS] {index_id}")

        # Build new index
        index = await asyncio.to_thread(
            VectorStoreIndex,
            nodes,
            embed_model=embed_model,
            storage_context=storage_context
        )

        if index_id:
            index.set_index_id(index_id)


            if not isinstance(index.vector_store, (RedisVectorStore, QdrantVectorStore)):
                await asyncio.to_thread(index.storage_context.persist)

        return index

    else:
        return await asyncio.to_thread(
            VectorStoreIndex,
            nodes,
            embed_model=embed_model,
            storage_context=storage_context
        )



def get_embed_model(is_code: bool):
    return HuggingFaceEmbedding(
        model_name="jinaai/jina-embeddings-v2-base-code" if is_code else "BAAI/bge-small-en-v1.5",
        trust_remote_code=True
    )


def get_retriever(index, embed_model):
    return VectorIndexRetriever(
        index=index,
        embed_model=embed_model,
        similarity_top_k=5,
        node_postprocessors=[CohereRerank(top_n=3)]
    )


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

                if node_type == "blob":  # file
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




async def get_specific_files(files: List[str], owner: str, repo: str):
    """Fixed async version of get_specific_files"""
    print(f"[DEBUG] Fetching repository: {owner}/{repo}")

    # Run GitHub API calls in thread to avoid blocking
    repo_data = await to_thread(_get_repo_data, owner, repo, files)

    return repo_data


def _get_repo_data(owner: str, repo: str, files: List[str]):
    """Synchronous helper for GitHub API calls"""
    from collections import defaultdict

    repo = g.get_repo(f"{owner}/{repo}")
    sha = repo.get_branch(repo.default_branch).commit.sha
    docs, langs = [], defaultdict(int)

    for fil in files:
        print(f"[DEBUG] Fetching file: {fil}")
        content = repo.get_contents(path=fil)
        print(f"[DEBUG] Retrieved file: {fil} | SHA: {content.sha} | Size: {content.size} bytes")

        # Detect language from extension
        ext = os.path.splitext(fil)[1].lower()
        file_lang = EXT_MAP.get(ext, "Unknown")
        langs[file_lang] += 1
        print(f"[DEBUG] Detected language (by extension): {file_lang}")

        docs.append(
            Document(
                doc_id=content.sha,
                text=content.decoded_content.decode('utf-8') if isinstance(content.decoded_content,
                                                                           bytes) else content.decoded_content,
                metadata={"path": fil, "size": content.size}
            )
        )

    print(f"[DEBUG] Total docs collected: {len(docs)}")
    print(f"[DEBUG] Language breakdown: {dict(langs)}")

    return {
        "docs": docs,
        "langs": langs,
        "sha": sha
    }


async def git_handler(req: GitRequest, session_id, context_id, context_type):
    """Fixed async git handler"""
    index_namespace = f"central-gpt:code"
    embedding_dim = 768  # adjust if your embedding vector length differs

    vector_schema = IndexSchema.from_dict({
        "index": {
            "name": "centralGPT",
            "prefix": "code" ,
            "key_separator": ":",
            "storage_type": "json",
        },
        "fields": [
            {"name": "id", "type": "text"},
            {"name": "user", "type": "tag"},
            {"name": "credit_score", "type": "tag"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "algorithm": "hnsw",  # better than flat for larger data
                    "dims": embedding_dim,  # must match your embedding vector size
                    "distance_metric": "cosine",
                    "datatype": "float32"
                }
            }
        ]
    })

    storage_context = StorageContext.from_defaults(
        docstore=get_doc_store(),
        index_store=get_index_store(namespace=index_namespace),
        vector_stores=get_vector_store(vector_schema)
    )

    if req.files:
        git_dic = await get_specific_files(req.files, req.owner, req.repo)
    else:
        git_dic = await git_documents(req,storage_context)

    documents = git_dic["docs"]
    langs = git_dic["langs"]
    logger.info(f"Languages: {langs}")

    language = max(langs, key=langs.get) if langs else "python"
    logger.info(f"Primary language: {language}")

    nodes = await get_nodes(documents=documents,storage_context=storage_context, language=language.lower(), is_code=True)
    embed_model = get_embed_model(is_code=True)

    index = await get_or_build_index(
        index_id=get_repo_index_id(owner=req.owner, repo=req.repo, commit_sha=git_dic["sha"]),
        nodes=nodes,
        is_code=True,
        language=language
    )

    retriever = get_retriever(index, embed_model=embed_model)

    chat_engine.set_engine(
        session_id=session_id,
        context_type=context_type,
        context_id=context_id,
        engine=retriever
    )

    return JSONResponse(
        status_code=200,
        content="Engine has been Set"
    )


async def get_handler(req: List[UploadFile], session_id, context_id, context_type):

    """Fixed async document handler"""
    index_namespace =  "central-gpt:notes"
    embedding_dim = 768  # adjust if your embedding vector length differs

    vector_schema = IndexSchema.from_dict({
        "index": {
            "name": "centralGPT",
            "prefix": "notes",
            "key_separator": ":",
            "storage_type": "json",
        },
        "fields": [
            {"name": "user", "type": "tag"},
            {"name": "credit_score", "type": "tag"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "algorithm": "hnsw",  # better than flat for larger data
                    "dims": embedding_dim,  # must match your embedding vector size
                    "distance_metric": "cosine",
                    "datatype": "float32"
                }
            }
        ]
    })

    storage_context = StorageContext.from_defaults(
        docstore=get_doc_store(),
        index_store=get_index_store(namespace=index_namespace),
        vector_stores=get_vector_store(vector_schema)
    )
    documents = await get_documents(req)
    nodes = await get_nodes(documents=documents,storage_context=storage_context, is_code=False)
    embed_model = get_embed_model(is_code=False)

    index = await get_or_build_index(index_id=None, nodes=nodes, is_code=False,storage_context=storage_context)
    retriever = get_retriever(index, embed_model=embed_model)

    chat_engine.set_engine(
        session_id=session_id,
        context_type=context_type,
        context_id=context_id,
        engine=retriever
    )

    return JSONResponse(
        status_code=200,
        content="Engine has been Set For notes"
    )