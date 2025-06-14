# import ast
# from inspect import walktree
#
#
# def extract_sematics(code):
#     tree = ast.parse(code)
#     chunks = []
#     for node in walktree(tree):
#         if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
#             chunk = ast.get_source_segment(node)
#             chunks.append({"type" : type(node).__class__ , "code" : chunk})
#
#
#
# from tree_sitter import Parser,Language
#
# from tree_sitter import Language, Parser
#
# # Setup
# Language.build_library(
#     'build/my-languages.so',
#     ['tree-sitter-python']
# )
# PY_LANGUAGE = Language('build/my-languages.so', 'python')
# parser = Parser()
# parser.set_language(PY_LANGUAGE)
#
# def get_tree_sitter_chunks(code):
#     tree = parser.parse(bytes(code, "utf8"))
#     root = tree.root_node
#     return root.sexp()  # OR walk the tree for comments, docstrings
#
#
# combined_chunks = []
# for chunk in extract_ast_chunks(code):
#     combined = {
#         "type": chunk["type"],
#         "code": chunk["code"],
#         "syntax": get_tree_sitter_chunks(chunk["code"])  # optional
#     }
#     combined_chunks.append(combined)
