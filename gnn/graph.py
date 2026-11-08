"""
Graph primitives, implemented from scratch with plain PyTorch.

Slide reference
---------------
Deck 1, "Sparse Graph Representation" (p.28-29): a graph is stored as an
*adjacency list* / COO `edge_index` of shape [2, E] instead of a dense
[N, N] adjacency matrix. Memory is O(E) instead of O(N^2).

Everything below operates on `edge_index`, so no dense NxN matrix is ever
materialised and no graph library is used. The only primitive we need is
"scatter-add", which plain PyTorch gives us via `Tensor.index_add_`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# scatter helpers  (the one building block all message passing needs)
# ---------------------------------------------------------------------------
def scatter_sum(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Sum rows of `src` into buckets given by `index`.

    out[i] = sum_{e : index[e] == i} src[e]

    This is THE aggregation primitive. `src` holds one message per edge,
    `index` says which node each message is destined for.
    """
    out = src.new_zeros((num_nodes,) + src.shape[1:])
    return out.index_add_(0, index, src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Mean-aggregate messages per destination node."""
    total = scatter_sum(src, index, num_nodes)
    ones = src.new_ones((src.shape[0], 1))
    count = scatter_sum(ones, index, num_nodes).clamp(min=1.0)
    return total / count


def scatter_max(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Max-aggregate messages per destination node (empty buckets -> 0)."""
    out = src.new_full((num_nodes,) + src.shape[1:], float("-inf"))
    out = out.scatter_reduce(0, index.view(-1, 1).expand_as(src), src,
                             reduce="amax", include_self=True)
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax over all edges sharing the same destination node.

    Needed by attention (Deck 2, p.59): alpha_ij = softmax_j(e_ij), where the
    softmax is taken over the *neighbourhood of i only* — not over all nodes.
    Numerically stabilised by subtracting the per-node maximum.
    """
    # per-destination max
    max_per = scores.new_full((num_nodes,) + scores.shape[1:], float("-inf"))
    max_per = max_per.scatter_reduce(0, index.view(-1, 1).expand_as(scores), scores,
                                     reduce="amax", include_self=True)
    max_per = torch.where(torch.isfinite(max_per), max_per, torch.zeros_like(max_per))
    exp = (scores - max_per[index]).exp()
    denom = scatter_sum(exp, index, num_nodes)
    return exp / (denom[index] + 1e-16)


# ---------------------------------------------------------------------------
# graph construction helpers
# ---------------------------------------------------------------------------
def to_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    """Add the reverse of every edge and drop duplicates.

    Cora citations are directed ("A cites B") but the slides treat the
    citation graph as undirected, so information flows both ways.
    """
    both = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    # de-duplicate by encoding each (src, dst) pair as a single integer
    n = int(both.max()) + 1
    key = torch.unique(both[0] * n + both[1])
    src = torch.div(key, n, rounding_mode="floor")
    dst = key % n
    return torch.stack([src, dst], dim=0)


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """A_hat = A + I  (Deck 1 p.23-24, Deck 2 p.22-23).

    Without a self-loop a node's own features are dropped from the
    aggregation — it would only ever see its neighbours.
    """
    loop = torch.arange(num_nodes, device=edge_index.device)
    loop = torch.stack([loop, loop], dim=0)
    return torch.cat([edge_index, loop], dim=1)


def degree(edge_index: torch.Tensor, num_nodes: int, which: str = "dst") -> torch.Tensor:
    """Node degrees counted from the edge list."""
    idx = edge_index[1] if which == "dst" else edge_index[0]
    ones = torch.ones(idx.shape[0], 1, device=edge_index.device)
    return scatter_sum(ones, idx, num_nodes).squeeze(1)


def gcn_normalisation(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Per-edge weights of  = D^(-1/2) (A + I) D^(-1/2).

    Deck 2, p.13-14 & p.22-24 walk through exactly this. Rather than build a
    dense matrix we return one scalar weight per edge:

        w_ij = 1 / sqrt(deg(i) * deg(j))

    so that aggregation becomes a weighted sum over the edge list. Symmetric
    normalisation keeps activations from exploding/vanishing with depth, and
    stops high-degree nodes from dominating.

    Returns
    -------
    (edge_index_with_loops, edge_weight)
    """
    ei = add_self_loops(edge_index, num_nodes)
    deg = degree(ei, num_nodes, which="dst")
    dinv = deg.pow(-0.5)
    dinv[torch.isinf(dinv)] = 0.0
    w = dinv[ei[0]] * dinv[ei[1]]
    return ei, w


# ---------------------------------------------------------------------------
# graph-level readout (Deck 2, p.61: graph-level vs node-level prediction)
# ---------------------------------------------------------------------------
def global_mean_pool(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Average all node embeddings belonging to the same graph."""
    return scatter_mean(x, batch, num_graphs)


def global_sum_pool(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    return scatter_sum(x, batch, num_graphs)


def global_max_pool(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    return scatter_max(x, batch, num_graphs)


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------
@dataclass
class Graph:
    """A single graph (used for Cora / Karate: one big graph, node-level task)."""

    x: torch.Tensor                       # [N, F] node features
    edge_index: torch.Tensor              # [2, E] COO edges
    y: torch.Tensor | None = None         # [N] node labels
    edge_type: torch.Tensor | None = None  # [E] relation id (R-GCN)
    train_mask: torch.Tensor | None = None
    val_mask: torch.Tensor | None = None
    test_mask: torch.Tensor | None = None
    meta: dict = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def __repr__(self) -> str:
        return (f"Graph(nodes={self.num_nodes}, edges={self.num_edges}, "
                f"features={self.x.shape[1]}, "
                f"classes={int(self.y.max()) + 1 if self.y is not None else None})")


@dataclass
class MolBatch:
    """A batch of small graphs stacked into one disconnected graph.

    This is the batching trick the libraries hide: concatenate node features,
    offset each graph's edge indices, and keep a `batch` vector saying which
    graph every node came from. Messages never cross graphs because no edge
    connects them.
    """

    x: torch.Tensor           # [sum_N, F]
    edge_index: torch.Tensor  # [2, sum_E]
    edge_attr: torch.Tensor   # [sum_E, F_e]
    batch: torch.Tensor       # [sum_N] graph id per node
    y: torch.Tensor           # [num_graphs]
    num_graphs: int


def collate_graphs(items) -> MolBatch:
    """Stack a list of (x, edge_index, edge_attr, y) into one MolBatch."""
    xs, eis, eas, batch, ys = [], [], [], [], []
    offset = 0
    for gi, (x, ei, ea, y) in enumerate(items):
        n = x.shape[0]
        xs.append(x)
        eis.append(ei + offset)          # shift node ids into the big graph
        eas.append(ea)
        batch.append(torch.full((n,), gi, dtype=torch.long))
        ys.append(y)
        offset += n
    return MolBatch(
        x=torch.cat(xs, 0),
        edge_index=torch.cat(eis, 1) if eis else torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.cat(eas, 0) if eas else torch.zeros((0, 1)),
        batch=torch.cat(batch, 0),
        y=torch.stack(ys),
        num_graphs=len(items),
    )
