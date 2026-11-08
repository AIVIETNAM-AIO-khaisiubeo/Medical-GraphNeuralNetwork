"""Dataset loaders: Cora (node-level), Karate Club, synthetic relational, ESOL (graph-level)."""

from .cora import CLASSES, load_cora
from .esol import EDGE_DIM, NODE_DIM, load_esol, smiles_to_graph, split_indices
from .karate import load_karate
from .relational import RELATIONS, make_relational_graph

__all__ = [
    "load_cora", "CLASSES",
    "load_karate",
    "make_relational_graph", "RELATIONS",
    "load_esol", "smiles_to_graph", "split_indices", "NODE_DIM", "EDGE_DIM",
]
