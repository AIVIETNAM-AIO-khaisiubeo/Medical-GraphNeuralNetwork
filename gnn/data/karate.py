"""
Zachary's Karate Club — 34 members, 78 observed friendships.

Deck 2, p.32-37. A conflict split the real club into two factions (around
"Mr. Hi", node 0, and the administrator "John A", node 33). The striking
demo on p.36 is that a GCN with **random, untrained** weights already
separates the two factions, because message passing alone propagates the
community structure present in the edges.

Node features are the identity matrix (p.34): with X = I, the first layer's
weight matrix becomes a freely learnable embedding per node, so the model has
no input information other than the graph topology itself.
"""
from __future__ import annotations

import torch

from ..graph import Graph, to_undirected

# the canonical 78 undirected edges, 0-indexed
EDGES = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10),
    (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
    (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
    (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
    (3, 7), (3, 12), (3, 13),
    (4, 6), (4, 10),
    (5, 6), (5, 10), (5, 16),
    (6, 16),
    (8, 30), (8, 32), (8, 33),
    (9, 33),
    (13, 33),
    (14, 32), (14, 33),
    (15, 32), (15, 33),
    (18, 32), (18, 33),
    (19, 33),
    (20, 32), (20, 33),
    (22, 32), (22, 33),
    (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
    (24, 25), (24, 27), (24, 31),
    (25, 31),
    (26, 29), (26, 33),
    (27, 33),
    (28, 31), (28, 33),
    (29, 32), (29, 33),
    (30, 32), (30, 33),
    (31, 32), (31, 33),
    (32, 33),
]

# faction each member joined after the split (0 = Mr. Hi, 1 = John A)
FACTION = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
]


def load_karate() -> Graph:
    """Return the karate club as a `Graph` with identity node features."""
    assert len(EDGES) == 78 and len(FACTION) == 34
    edge_index = to_undirected(torch.tensor(EDGES, dtype=torch.long).t())
    x = torch.eye(34)                      # one-hot per node => learnable embedding
    y = torch.tensor(FACTION, dtype=torch.long)
    # semi-supervised flavour of the classic demo: label only the two leaders
    train_mask = torch.zeros(34, dtype=torch.bool)
    train_mask[0] = True
    train_mask[33] = True
    return Graph(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                 meta={"name": "KarateClub", "classes": ["Mr. Hi", "John A"]})
