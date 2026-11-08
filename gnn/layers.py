"""
GNN layers written from scratch.

Every layer here is an instance of the one message-passing equation from
Deck 1 (p.53):

    h_u^(k+1) = UPDATE^(k)( h_u^(k),  AGGREGATE^(k)({ h_v^(k) : v in N(u) }) )

The layers differ only in *what the message is* and *how it is weighted*:

| layer        | message from v to u                  | weighting              |
|--------------|--------------------------------------|------------------------|
| GCNLayer     | W h_v                                | fixed, 1/sqrt(d_u d_v) |
| GATLayer     | W h_v                                | learned attention a_uv |
| RGCNLayer    | W_r h_v  (one W per relation r)      | 1/|N_u^r|              |
| EdgeMPNNLayer| MLP([h_u, h_v, e_uv])                | mean over neighbours   |

Nothing imports a graph library; `scatter_*` from `graph.py` is the only
primitive used.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import scatter_mean, scatter_sum, segment_softmax


class GCNLayer(nn.Module):
    """Graph Convolutional layer:  H' = sigma( Â H W ).

    Deck 2, p.22-24. `Â = D^-1/2 (A+I) D^-1/2` is supplied as per-edge
    weights, so the dense matrix product becomes a weighted scatter-add.

    Parameters
    ----------
    ngnn : bool
        If True, apply an extra non-linear MLP *inside* the layer. This is the
        over-smoothing remedy from Deck 1 (p.50-52): deepen the network
        without adding more message-passing hops.
    """

    def __init__(self, in_dim: int, out_dim: int, ngnn: bool = False, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.ngnn = nn.Sequential(nn.Linear(out_dim, out_dim), nn.ReLU()) if ngnn else None

    def forward(self, x, edge_index, edge_weight):
        # 1. TRANSFORM: project every node once (cheaper than per-edge)
        h = self.lin(x)
        src, dst = edge_index[0], edge_index[1]
        # 2. MESSAGE: neighbour feature scaled by the normalisation weight
        msg = h[src] * edge_weight.unsqueeze(1)
        # 3. AGGREGATE: sum messages per destination node
        out = scatter_sum(msg, dst, x.shape[0])
        if self.ngnn is not None:
            out = self.ngnn(out)
        return out


class GATLayer(nn.Module):
    """Graph Attention layer (Deck 2, p.55-59).

    For a reader coming from Transformers: this *is* self-attention, with two
    changes.  (1) The attention is masked to the graph — node i only attends
    to its neighbours N(i), instead of to every token.  (2) The score is an
    additive (Bahdanau-style) MLP rather than a scaled dot product:

        e_ij     = LeakyReLU( a^T [ W h_i || W h_j ] )
        alpha_ij = softmax_{j in N(i)}( e_ij )
        h_i'     = sigma( sum_j alpha_ij W h_j )

    Multiple heads are computed independently and concatenated (or averaged
    on the final layer), exactly as in multi-head attention.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 1,
                 concat: bool = True, negative_slope: float = 0.2, dropout: float = 0.0):
        super().__init__()
        self.heads, self.out_dim, self.concat = heads, out_dim, concat
        self.negative_slope, self.dropout = negative_slope, dropout
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        # the attention vector 'a', split into its source and destination half
        self.att_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, edge_index, edge_weight=None):
        N = x.shape[0]
        H, C = self.heads, self.out_dim
        h = self.lin(x).view(N, H, C)                    # [N, heads, C]
        src, dst = edge_index[0], edge_index[1]

        # a^T [Wh_i || Wh_j] splits into two dot products (cheaper, identical)
        alpha = (h * self.att_src).sum(-1)[src] + (h * self.att_dst).sum(-1)[dst]
        alpha = F.leaky_relu(alpha, self.negative_slope)  # [E, heads]
        # softmax restricted to each destination node's neighbourhood
        alpha = segment_softmax(alpha, dst, N)
        if self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = h[src] * alpha.unsqueeze(-1)               # [E, heads, C]
        out = scatter_sum(msg, dst, N)                   # [N, heads, C]
        out = out.reshape(N, H * C) if self.concat else out.mean(dim=1)
        return out + self.bias


class RGCNLayer(nn.Module):
    """Relational GCN (Deck 2, p.38-44).

    Different edge *types* (friend / family / colleague) should not be mixed:
    each relation r gets its own weight matrix W_r, and a node sums the
    per-relation aggregations plus a self-transform W_0:

        h_i' = W_0 h_i + sum_r  (1/|N_i^r|) sum_{j in N_i^r} W_r h_j

    The 1/|N_i^r| normalisation keeps relations with many neighbours from
    swamping rarer ones.
    """

    def __init__(self, in_dim: int, out_dim: int, num_relations: int):
        super().__init__()
        self.num_relations = num_relations
        self.rel_lin = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)])
        self.self_lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_type):
        out = self.self_lin(x)                            # W_0 h_i
        src, dst = edge_index[0], edge_index[1]
        for r, lin in enumerate(self.rel_lin):
            mask = edge_type == r
            if not bool(mask.any()):
                continue
            h_r = lin(x)
            out = out + scatter_mean(h_r[src[mask]], dst[mask], x.shape[0])
        return out


class EdgeMPNNLayer(nn.Module):
    """Message passing that consumes multi-dimensional EDGE features.

    Deck 2, p.48-50 ("How to integrate multi-dimensional edge features into
    the transformation of the neighbourhood states?"). The message is an MLP
    over the concatenation of both endpoints and the edge's own feature
    vector, so bond type / relation strength actually influences the message:

        m_uv = MLP([ h_u || h_v || e_uv ])
        h_u' = ReLU( W_self h_u + mean_v m_uv )
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * in_dim + edge_dim, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim))
        self.self_lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        if src.numel() == 0:                              # single-atom molecule
            return F.relu(self.self_lin(x))
        m = self.msg(torch.cat([x[dst], x[src], edge_attr], dim=1))
        agg = scatter_mean(m, dst, x.shape[0])
        return F.relu(self.self_lin(x) + agg)
