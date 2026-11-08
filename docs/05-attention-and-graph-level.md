# 05 — Attention (GAT), and graph-level prediction

*Slides: Deck 2, p.55–71.*

## Part 1 — GAT

### The equations

*Deck 2, p.58–59.*

```
e_ij     = LeakyReLU( aᵀ [ W h_i ‖ W h_j ] )          raw score
α_ij     = softmax over j ∈ N(i) of e_ij               normalised
h_i'     = σ( Σ_{j ∈ N(i)} α_ij · W h_j )              aggregate
```

GCN's weight `1/√(d_i d_j)` is fixed by the graph's shape. GAT replaces it
with `α_ij`, computed from the node features — so the model decides, per edge
and per layer, which neighbours to listen to.

### If you know Transformers

This is self-attention, with exactly two differences:

| | Transformer self-attention | GAT |
|---|---|---|
| who attends to whom | every token to every token | only to graph neighbours |
| score function | scaled dot product `qᵀk/√d` | additive MLP `aᵀ[Wh_i ‖ Wh_j]` |
| normalisation | softmax over all tokens | softmax over `N(i)` |
| multi-head | concat, then project | concat (or average on the last layer) |

**The adjacency matrix is the attention mask.** Self-attention is the special
case of GAT on a fully connected graph; GAT is self-attention with a sparse,
data-given mask. That also explains the cost difference: Transformer attention
is `O(N²)`, GAT is `O(E)`.

### The one implementation subtlety

The softmax is **per destination node**, not global — each node's incoming
attention must sum to 1. With an edge list there are no neat rows to softmax
over, so it is a *segment* softmax:

```python
# gnn/graph.py
def segment_softmax(scores, index, num_nodes):
    max_per = scatter_amax(scores, index, num_nodes)     # stability
    exp     = (scores - max_per[index]).exp()
    denom   = scatter_sum(exp, index, num_nodes)
    return exp / (denom[index] + 1e-16)
```

Subtracting the per-segment maximum before exponentiating matters: scores are
unbounded and `exp` overflows without it.

One more detail: **add self-loops before attending**, or a node cannot attend
to itself and loses its own features. `GAT.forward` does this.

Also note the cheap factorisation in `GATLayer`: rather than concatenating
`[Wh_i ‖ Wh_j]` per edge and multiplying by `a`, split `a = [a_src ‖ a_dst]`
and compute `a_srcᵀWh_i + a_dstᵀWh_j`. Identical result, but the two dot
products are computed once per *node* and then gathered, instead of once per
*edge* on a doubled-width vector.

### Measured — and a caveat

Cora (`experiments/04`):

| model | test accuracy |
|---|---|
| GCN | 0.810 |
| GAT (8 heads) | **0.818** |

A small win. But inspecting the learned coefficients, `α_ij` divided by the
uniform weight `1/degree`:

| | 5th pct | median | 95th pct |
|---|---|---|---|
| layer 1 | 0.98 | 1.00 | 1.02 |
| layer 2 | 0.79 | 1.00 | 1.22 |

The attention is **nearly uniform** — it has learned to do roughly what GCN
already did. The attention parameters do move a lot during training (their
norm changes by 60–130%), so this is not a bug or a dead unit.

The reason is the data. Cora is strongly **homophilous**: almost every
citation links two papers on the same topic, so every neighbour is about
equally worth listening to and there is nothing for attention to suppress.
Attention pays off when neighbours genuinely differ in usefulness —
heterophilous graphs, noisy edges, or very high-degree hub nodes.

And attention is not a universal substitute for structure: in
[experiment 05](04-edge-features-and-relations.md) GAT scores 0.439 where
R-GCN scores 1.000, because the distinguishing information is the edge's
*type*, which attention never sees.

## Part 2 — Graph-level prediction

*Deck 2, p.61–71.*

### Readout

Node-level tasks read a prediction off each node. For a graph-level label you
need to collapse a **variable number** of node embeddings into **one** vector,
and the function must be permutation-invariant:

```
h_graph = POOL({ h_v : v ∈ V })        ŷ = MLP(h_graph)
```

The three standard choices, all in `gnn/graph.py`:

| pooling | keeps | loses |
|---|---|---|
| `mean` | average composition | size of the graph |
| `sum` | composition **and** size | comparability across sizes |
| `max` | the single most extreme node | everything else |

This is not a throwaway choice. On ESOL (`experiments/06`), same model, same
features, only the readout changed:

| readout | test RMSE |
|---|---|
| sum | **0.621** |
| mean | 0.864 |
| max | 0.980 |

Sum wins decisively, and the reason is chemistry: log-solubility scales with
molecular size, and mean pooling divides exactly that information away. Max
pooling throws away all but one atom. **Pick the readout that matches whether
your target is extensive (scales with size → sum) or intensive (a ratio or
average → mean).**

### Batching many small graphs

A node-level task has one graph; a graph-level task has thousands of small
ones, and running them one at a time wastes the hardware. The trick (which
libraries hide) is to stack a batch into **one big disconnected graph**:

- concatenate all node-feature matrices;
- offset each graph's `edge_index` by the number of nodes already placed;
- keep a `batch` vector recording which graph each node came from.

No edge connects two different graphs, so messages never leak between them —
one forward pass processes the whole batch, and the `batch` vector drives the
pooling. See `collate_graphs()` in `gnn/graph.py` and the equality check in
`tests/test_core.py::test_pooling_and_batching`.

### Results

ESOL, 1128 molecules, predicting log solubility (mol/L):

| model | test RMSE |
|---|---|
| predict the training mean | 2.087 |
| GNN, bond features zeroed | 0.937 |
| GNN + bond features, mean pooling | 0.864 |
| **GNN + bond features, sum pooling** | **0.621** |

RDKit parses the SMILES into atoms and bonds; the featurisation, batching,
layers and training loop are all in this repo.

---

Back to the [README](../README.md).
