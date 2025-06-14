from typing import Sequence, Any, List

from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import NodeWithEmbedding


class SemanticParser(NodeParser):
    def _parse_nodes(self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any) -> List[BaseNode]:
        pass



class SemanticEmbedding():
    def _parse_nodes(self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any) -> List[BaseNode]:
        pass