import os
import ast
from dotenv import load_dotenv
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import TextNode
from llama_index.llms.groq import Groq
from llama_index.readers.github import GithubClient, GithubRepositoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import Field
PY_LANGUAGE = Language(tspython.language())
tree_parser = Parser(PY_LANGUAGE)




class SemanticCodeParser(NodeParser):
    chunk_size: int = Field(default=1024, description="Maximum chunk size for text splitting")

    def __init__(self, chunk_size: int = 1024, **kwargs):
        super().__init__(chunk_size=chunk_size, **kwargs)

    def _parse_nodes(self, nodes, show_progress=False, **kwargs):
        all_nodes = []
        for node in nodes:
            try:
                all_nodes.extend(self._parse_single_node(node))
            except Exception as e:
                print(f"[WARN] Parse error: {e}")
                all_nodes.append(node)
        return all_nodes

    def _parse_single_node(self, node):
        file_path = node.metadata.get("file_path", "")
        if file_path.endswith(".py"):
            return self._parse_python(node.text, node.metadata)
        return self._chunk_text(node.text, node.metadata)

    def get_nodes_from_documents(self, docs, **kwargs):
        return self._parse_nodes(docs, **kwargs)

    def _parse_python(self, code, metadata):
        nodes = []
        try:
            ast_tree = ast.parse(code)

            # Extract imports
            imports = self._get_imports(ast_tree)
            if imports:
                nodes.append(TextNode(
                    text=f"# Imports\n{chr(10).join(imports)}",
                    metadata={**metadata, "node_type": "imports"}
                ))

            # Extract classes and functions
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.ClassDef):
                    nodes.append(self._create_class_node(node, code, metadata))
                elif isinstance(node, ast.FunctionDef) and not self._is_method(node, ast_tree):
                    nodes.append(self._create_function_node(node, code, metadata))

            return nodes if nodes else self._chunk_text(code, metadata)
        except:
            return self._chunk_text(code, metadata)

    def _get_imports(self, ast_tree):
        imports = []
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Import):
                imports.extend([f"import {alias.name}" for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                names = ", ".join([alias.name for alias in node.names])
                imports.append(f"from {node.module or ''} import {names}")
        return imports

    def _create_class_node(self, class_node, code, metadata):
        source = ast.get_source_segment(code, class_node) or ""
        methods = [n.name for n in class_node.body if isinstance(n, ast.FunctionDef)]
        docstring = ast.get_docstring(class_node) or ""

        text = f"# Class: {class_node.name}\n"
        if docstring: text += f"# Doc: {docstring}\n"
        if methods: text += f"# Methods: {', '.join(methods)}\n"
        text += f"\n{source}"

        # Add syntax tree if small enough
        if len(source) < 2000:
            try:
                syntax = tree_parser.parse(bytes(source, "utf8")).root_node.sexp()
                text += f"\n\n# Syntax: {syntax}"
            except:
                pass

        return TextNode(text=text, metadata={**metadata, "node_type": "class", "name": class_node.name})

    def _create_function_node(self, func_node, code, metadata):
        source = ast.get_source_segment(code, func_node) or ""
        args = [arg.arg for arg in func_node.args.args]
        docstring = ast.get_docstring(func_node) or ""

        text = f"# Function: {func_node.name}\n"
        if docstring: text += f"# Doc: {docstring}\n"
        if args: text += f"# Args: {', '.join(args)}\n"
        text += f"\n{source}"

        # Add syntax tree if small enough
        if len(source) < 2000:
            try:
                syntax = tree_parser.parse(bytes(source, "utf8")).root_node.sexp()
                text += f"\n\n# Syntax: {syntax}"
            except:
                pass

        return TextNode(text=text, metadata={**metadata, "node_type": "function", "name": func_node.name})

    def _is_method(self, func_node, ast_tree):
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.ClassDef) and func_node in node.body:
                return True
        return False

    def _chunk_text(self, text, metadata):
        chunks = []
        lines = text.split('\n')
        current_chunk, current_size = [], 0

        for line in lines:
            if current_size + len(line) > self.chunk_size and current_chunk:
                chunks.append(TextNode(
                    text='\n'.join(current_chunk),
                    metadata={**metadata, "content_type": "chunk"}
                ))
                current_chunk, current_size = [line], len(line)
            else:
                current_chunk.append(line)
                current_size += len(line)

        if current_chunk:
            chunks.append(TextNode(
                text='\n'.join(current_chunk),
                metadata={**metadata, "content_type": "chunk"}
            ))
        return chunks


# Setup
load_dotenv()
Settings.llm = Groq(model="llama3-70b-8192")
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
Settings.node_parser = SemanticCodeParser()


def load_repo(owner, repo, branch="main", dirs=None):
    try:
        client = GithubClient(verbose=True)
        kwargs = {"github_client": client, "owner": owner, "repo": "sturdy-octo-carnival", "verbose": True}
        if dirs:
            kwargs["filter_directories"] = (dirs, GithubRepositoryReader.FilterType.INCLUDE)
        docs = GithubRepositoryReader(**kwargs,use_parser=True).load_data(branch=branch)
        print(f"✅ Loaded {len(docs)} documents")
        return docs
    except Exception as e:
        print(f"❌ Error: {e}")
        return []


if __name__ == "__main__":
    # Load and index
    docs = load_repo("DineshThumma9", "sturdy-octo-carnival", "main", ["src"])
    if not docs:
        exit(1)

    print("🔧 Building index...")
    index = VectorStoreIndex.from_documents(docs)
    query_engine = index.as_query_engine(similarity_top_k=5)
    print("🚀 Ready!")

    # Chat loop
    while True:
        try:
            q = input("\n💬 Ask: ").strip()
            if not q or q.lower() in ['exit', 'quit']:
                break
            response = query_engine.query(q)
            print(f"🤖 {response}")
        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break
        except Exception as e:
            print(f"❌ {e}")