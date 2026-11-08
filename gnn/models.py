"""
Models assembled from the scratch layers.

Deck 1 builds exactly two models for Cora and compares them:
  * MLP  (p.66): 1433 -> 16 -> 7. Ignores the graph completely.
  * GCN  (p.68-71): two message-passing layers + one linear output layer,
    with dropout, trained with cross-entropy over 7 classes.

Deck 2 adds attention (GAT), relations (R-GCN) and graph-level prediction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import add_self_loops, global_max_pool, global_mean_pool, global_sum_pool
from .layers import EdgeMPNNLayer, GATLayer, GCNLayer, RGCNLayer


class MLP(nn.Module):
    """Graph-blind baseline (Deck 1, p.66).

    Classifies each paper from its bag-of-words alone. Any gap between this
    and the GCN is the value the *graph structure* adds.
    """

    def __init__(self, in_dim: int, hidden: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index=None, edge_weight=None, return_embedding: bool = False):
        h = F.relu(self.lin1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.lin2(h)
        return (logits, h) if return_embedding else logits


class GCN(nn.Module):
    """Graph Convolutional Network for node classification.

    Two head styles, because they score very differently on Cora and the
    difference is instructive:

    head="linear"  (what Deck 1, p.71 describes: "2 Message Passing Layers and
                   one Linear output layer")
                   1433 -> 16 -> 16 -> 7.  Squeezes the features through TWO
                   16-dim bottlenecks before classifying. Measured: ~0.68.

    head="direct"  (the canonical GCN of Kipf & Welling 2017)
                   1433 -> 16 -> 7, where the *second message-passing layer
                   itself* outputs the class scores. Measured: ~0.80, matching
                   the published number.

    `num_layers` is the knob for the over-smoothing experiment: it is the
    number of hops information travels (Deck 1, p.41). Too many and every
    node's embedding collapses onto the same vector.
    """

    def __init__(self, in_dim: int, hidden: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.5, ngnn: bool = False,
                 head: str = "direct"):
        super().__init__()
        assert num_layers >= 1
        assert head in ("direct", "linear")
        self.head, self.dropout = head, dropout
        if head == "direct":
            dims = [in_dim] + [hidden] * (num_layers - 1) + [num_classes]
        else:
            dims = [in_dim] + [hidden] * num_layers
            self.out = nn.Linear(hidden, num_classes)
        self.convs = nn.ModuleList([
            # no NGNN block on the layer that produces logits
            GCNLayer(dims[i], dims[i + 1],
                     ngnn=ngnn and not (head == "direct" and i == num_layers - 1))
            for i in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_weight, return_embedding: bool = False):
        h = F.dropout(x, p=self.dropout, training=self.training)
        emb, last = None, len(self.convs) - 1
        logits = None
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_weight)
            if self.head == "direct" and i == last:
                logits = h                     # final conv emits class scores
                break
            h = F.relu(h)
            emb = h                            # hidden embedding (for t-SNE / MAD)
            h = F.dropout(h, p=self.dropout, training=self.training)
        if logits is None:
            logits = self.out(h)
        if emb is None:
            emb = logits
        return (logits, emb) if return_embedding else logits


class GAT(nn.Module):
    """Two-layer Graph Attention Network (Deck 2, p.55-59).

    Layer 1 uses several heads and concatenates them; layer 2 uses a single
    head and outputs class logits directly.
    """

    def __init__(self, in_dim: int, hidden: int, num_classes: int,
                 heads: int = 8, dropout: float = 0.6):
        super().__init__()
        self.conv1 = GATLayer(in_dim, hidden, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATLayer(hidden * heads, num_classes, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, return_embedding: bool = False):
        ei = add_self_loops(edge_index, x.shape[0])   # a node must attend to itself
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = F.elu(self.conv1(h, ei))
        emb = h
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.conv2(h, ei)
        return (logits, emb) if return_embedding else logits


class RGCN(nn.Module):
    """Relational GCN for graphs with several edge types (Deck 2, p.38-44)."""

    def __init__(self, in_dim: int, hidden: int, num_classes: int, num_relations: int):
        super().__init__()
        self.conv1 = RGCNLayer(in_dim, hidden, num_relations)
        self.conv2 = RGCNLayer(hidden, hidden, num_relations)
        self.out = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, edge_type, return_embedding: bool = False):
        h = F.relu(self.conv1(x, edge_index, edge_type))
        h = F.relu(self.conv2(h, edge_index, edge_type))
        logits = self.out(h)
        return (logits, h) if return_embedding else logits


class GraphRegressor(nn.Module):
    """Graph-LEVEL model for ESOL solubility (Deck 2, p.61-71).

    Node-level tasks read embeddings straight off the nodes. A graph-level
    task needs one more step — a permutation-invariant **readout** that
    collapses a variable number of atom embeddings into one molecule vector:

        h_graph = POOL({ h_v : v in V })      then   y_hat = MLP(h_graph)

    Uses `EdgeMPNNLayer` so bond types (edge features) influence the messages.
    """

    POOLS = {"mean": global_mean_pool, "sum": global_sum_pool, "max": global_max_pool}

    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 64,
                 num_layers: int = 3, pool: str = "mean", dropout: float = 0.1):
        super().__init__()
        dims = [node_dim] + [hidden] * num_layers
        self.convs = nn.ModuleList(
            [EdgeMPNNLayer(dims[i], dims[i + 1], edge_dim) for i in range(num_layers)])
        self.pool = self.POOLS[pool]
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, edge_index, edge_attr, batch, num_graphs):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
        g = self.pool(h, batch, num_graphs)      # readout
        return self.head(g).squeeze(-1)
