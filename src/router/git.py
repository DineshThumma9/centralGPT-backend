# from dotenv import load_dotenv
# from llama_index.core import Settings, VectorStoreIndex
# from llama_index.core.node_parser import NodeParser
# from llama_index.core.schema import TextNode
# from llama_index.llms.groq import Groq
# from llama_index.readers.github import GithubClient, GithubRepositoryReader
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#
# import ast
# from tree_sitter import Language, Parser
# import os
#
# # ---- Setup tree-sitter ----
# def setup_tree_sitter():
#     if not os.path.exists("build/my-languages.so"):
#         Language.build_library("build/my-languages.so", ["tree-sitter-python"])
#     lang = Language("build/my-languages.so", "python")
#     parser = Parser()
#     parser.set_language(lang)
#     return parser
#
# tree_parser = setup_tree_sitter()
#
# # ---- AST + Tree-sitter parser ----
# class SemanticCodeParser(NodeParser):
#     def get_nodes_from_documents(self, docs, **kwargs):
#         all_nodes = []
#         for doc in docs:
#             if doc.metadata.get("file_path", "").endswith(".py"):
#                 all_nodes.extend(self._parse_python(doc.text))
#             else:
#                 all_nodes.append(TextNode(text=doc.text, metadata=doc.metadata))
#         return all_nodes
#
#     def _parse_python(self, code):
#         nodes = []
#         try:
#             tree = ast.parse(code)
#             for node in ast.walk(tree):
#                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
#                     src = ast.get_source_segment(code, node)
#                     if not src:
#                         continue
#                     syntax = tree_parser.parse(bytes(src, "utf8")).root_node.sexp()
#                     text = f"# {type(node).__name__}\n{src}\n\n# Syntax\n{syntax}"
#                     nodes.append(TextNode(text=text))
#         except:
#             pass
#         return nodes
#
# # ---- Environment ----
# load_dotenv()
#
# Settings.llm = Groq(model="compound-beta")
# Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
# Settings.node_parser = SemanticCodeParser()
#
# # ---- GitHub Setup ----
# github_client = GithubClient(verbose=True)
#
# documents = GithubRepositoryReader(
#     github_client=github_client,
#     owner="DineshThumma9",
#     repo="luffy",
#     verbose=True,
#     filter_directories=(["src"], GithubRepositoryReader.FilterType.INCLUDE)
# ).load_data(branch="main")
#
# # ---- Indexing ----
# index = VectorStoreIndex.from_documents(documents)
# query_engine = index.as_query_engine()
#
# while True:
#     q = input("💬 Ask: ")
#     print(query_engine.query(q))
