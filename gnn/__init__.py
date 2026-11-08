"""
gnn — Graph Neural Networks implemented from scratch in plain PyTorch.

No PyTorch Geometric, no DGL, no NetworkX. The only graph primitive is
scatter-add over an edge list (see `gnn/graph.py`).

Layout
------
graph.py    sparse graph representation, scatter ops, Â normalisation, pooling
layers.py   GCN / GAT / R-GCN / edge-feature message passing layers
models.py   MLP, GCN, GAT, RGCN, GraphRegressor
metrics.py  accuracy, RMSE, MAD, MADGap (over-smoothing diagnostics)
data/       Cora, Karate Club, ESOL loaders + featurisation
"""

from .graph import (Graph, MolBatch, add_self_loops, collate_graphs, degree,
                    gcn_normalisation, global_max_pool, global_mean_pool,
                    global_sum_pool, scatter_max, scatter_mean, scatter_sum,
                    segment_softmax, to_undirected)
from .layers import EdgeMPNNLayer, GATLayer, GCNLayer, RGCNLayer
from .metrics import accuracy, mad, mad_gap, mae, rmse
from .models import GAT, GCN, MLP, RGCN, GraphRegressor

__all__ = [
    "Graph", "MolBatch", "collate_graphs", "to_undirected", "add_self_loops",
    "degree", "gcn_normalisation", "scatter_sum", "scatter_mean", "scatter_max",
    "segment_softmax", "global_mean_pool", "global_sum_pool", "global_max_pool",
    "GCNLayer", "GATLayer", "RGCNLayer", "EdgeMPNNLayer",
    "MLP", "GCN", "GAT", "RGCN", "GraphRegressor",
    "accuracy", "rmse", "mae", "mad", "mad_gap",
]
