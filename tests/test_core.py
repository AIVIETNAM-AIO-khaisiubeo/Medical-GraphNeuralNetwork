"""
Correctness tests for the from-scratch implementation.

The important one is `test_gcn_matches_dense`: it checks that our sparse,
scatter-based GCN layer computes exactly  sigma(Â H W)  with
Â = D^-1/2 (A+I) D^-1/2 built as a dense matrix. If that passes, the
message-passing machinery is right.

Run:  python tests/test_core.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn import (GATLayer, GCNLayer, RGCNLayer, EdgeMPNNLayer, add_self_loops,
                 collate_graphs, gcn_normalisation, global_mean_pool, mad,
                 scatter_sum, segment_softmax, to_undirected)
from gnn.data.karate import load_karate


def test_to_undirected():
    ei = torch.tensor([[0, 1], [1, 2]])
    und = to_undirected(ei)
    pairs = set(map(tuple, und.t().tolist()))
    assert pairs == {(0, 1), (1, 0), (1, 2), (2, 1)}, pairs
    print("  to_undirected               ok")


def test_scatter_sum():
    src = torch.tensor([[1.0], [2.0], [3.0]])
    idx = torch.tensor([0, 0, 2])
    out = scatter_sum(src, idx, 3)
    assert torch.allclose(out, torch.tensor([[3.0], [0.0], [3.0]])), out
    print("  scatter_sum                 ok")


def test_segment_softmax_sums_to_one():
    scores = torch.randn(10, 2)
    index = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
    a = segment_softmax(scores, index, 4)
    sums = scatter_sum(a, index, 4)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), sums
    print("  segment_softmax             ok")


def test_gcn_matches_dense():
    """Sparse scatter GCN  ==  dense  Â H W."""
    torch.manual_seed(0)
    g = load_karate()
    n = g.num_nodes
    x = torch.randn(n, 7)

    layer = GCNLayer(7, 5)
    ei, w = gcn_normalisation(g.edge_index, n)
    sparse_out = layer(x, ei, w)

    # dense reference
    A = torch.zeros(n, n)
    A[g.edge_index[0], g.edge_index[1]] = 1.0
    A = A + torch.eye(n)
    deg = A.sum(1)
    dinv = deg.pow(-0.5)
    A_hat = torch.diag(dinv) @ A @ torch.diag(dinv)
    dense_out = A_hat @ layer.lin(x)

    assert torch.allclose(sparse_out, dense_out, atol=1e-5), \
        (sparse_out - dense_out).abs().max()
    print("  GCNLayer == dense Â H W     ok")


def test_gcn_is_permutation_equivariant():
    """Relabelling nodes permutes the output the same way (Deck 1, p.25)."""
    torch.manual_seed(0)
    g = load_karate()
    n = g.num_nodes
    x = torch.randn(n, 6)
    layer = GCNLayer(6, 4)

    ei, w = gcn_normalisation(g.edge_index, n)
    out = layer(x, ei, w)

    perm = torch.randperm(n)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n)
    x_p = x[perm]
    ei_p = inv[g.edge_index]
    ei_p2, w_p = gcn_normalisation(ei_p, n)
    out_p = layer(x_p, ei_p2, w_p)

    assert torch.allclose(out_p, out[perm], atol=1e-5), \
        (out_p - out[perm]).abs().max()
    print("  permutation equivariance    ok")


def test_gat_shapes_and_attention():
    torch.manual_seed(0)
    g = load_karate()
    n = g.num_nodes
    x = torch.randn(n, 6)
    ei = add_self_loops(g.edge_index, n)
    layer = GATLayer(6, 4, heads=3, concat=True)
    out = layer(x, ei)
    assert out.shape == (n, 12), out.shape
    avg = GATLayer(6, 4, heads=3, concat=False)
    assert avg(x, ei).shape == (n, 4)
    print("  GATLayer                    ok")


def test_rgcn_uses_relations():
    """Different relation types must produce different outputs."""
    torch.manual_seed(0)
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]])
    x = torch.randn(3, 5)
    layer = RGCNLayer(5, 4, num_relations=2)
    out_a = layer(x, ei, torch.tensor([0, 0, 0]))
    out_b = layer(x, ei, torch.tensor([1, 1, 1]))
    assert not torch.allclose(out_a, out_b), "relations are being ignored"
    print("  RGCNLayer                   ok")


def test_edge_mpnn_uses_edge_features():
    torch.manual_seed(0)
    ei = torch.tensor([[0, 1], [1, 0]])
    x = torch.randn(2, 5)
    layer = EdgeMPNNLayer(5, 4, edge_dim=3)
    e1 = torch.zeros(2, 3)
    e2 = torch.ones(2, 3)
    assert not torch.allclose(layer(x, ei, e1), layer(x, ei, e2)), \
        "edge features are being ignored"
    # isolated node (no bonds) must still work
    out = layer(x, torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 3)))
    assert out.shape == (2, 4)
    print("  EdgeMPNNLayer               ok")


def test_pooling_and_batching():
    """Batched pooling must equal pooling each graph separately."""
    torch.manual_seed(0)
    items = []
    for n in (3, 5, 2):
        x = torch.randn(n, 4)
        ei = torch.tensor([[0], [min(1, n - 1)]])
        ea = torch.zeros(1, 2)
        items.append((x, ei, ea, torch.tensor(0.0)))
    batch = collate_graphs(items)
    pooled = global_mean_pool(batch.x, batch.batch, batch.num_graphs)
    for gi, (x, *_rest) in enumerate(items):
        assert torch.allclose(pooled[gi], x.mean(0), atol=1e-6)
    # edge indices must have been offset, never pointing outside their graph
    assert int(batch.edge_index.max()) < batch.x.shape[0]
    print("  batching + pooling          ok")


def test_mad_detects_collapse():
    """Identical embeddings => MAD 0; random embeddings => MAD > 0."""
    same = torch.ones(20, 8)
    rand = torch.randn(20, 8)
    assert mad(same) < 1e-5, mad(same)
    assert mad(rand) > 0.5, mad(rand)
    print("  MAD                         ok")


if __name__ == "__main__":
    print("running from-scratch GNN tests")
    for fn in [test_to_undirected, test_scatter_sum, test_segment_softmax_sums_to_one,
               test_gcn_matches_dense, test_gcn_is_permutation_equivariant,
               test_gat_shapes_and_attention, test_rgcn_uses_relations,
               test_edge_mpnn_uses_edge_features, test_pooling_and_batching,
               test_mad_detects_collapse]:
        fn()
    print("ALL TESTS PASSED")
