# 01 — Graph data and tasks

> [!NOTE]
> **Study navigation:** [← 00 — Tóm tắt tiếng Việt](00-tom-tat-tieng-viet.md) · [Repository overview](../README.md#learning-roadmap) · [02 — Message passing and GCN →](02-message-passing-and-gcn.md)

---

*Slides: Deck 1, p.5–29.*

## What a graph is

A graph $G=(V,E)$ is a set of **entities** (nodes) and the **relations**
between them (edges). Information can live in three places:

- **node features** $X\in\mathbb{R}^{N\times F}$ — one vector per node;
- **edge features** $\mathbf{E}\in\mathbb{R}^{M\times F_e}$ — one vector per edge (Deck 2's whole topic);
- **global features** — one vector for the whole graph.

The slides' point in p.8–15 is that a surprising amount of data is a graph:

| data | nodes | edges |
|---|---|---|
| citation network | papers | "A cites B" |
| molecule | atoms | chemical bonds |
| social network | people | friendships |
| image | pixels | adjacency in the 8-neighbourhood |
| text | tokens | "next token" |

Images and text are graphs too — but *regular* ones. Every interior pixel has
exactly 8 neighbours, in fixed relative positions. That regularity is what
lets a CNN slide one small kernel over everything. A social network or a
molecule has no such regularity: node degree varies, and there is no "up" or
"left".

## Three levels of task

This determines the shape of your output and the design of your model's last
layer (Deck 1, p.16–21).

| level | question | example | output shape |
|---|---|---|---|
| **node** | what is this node? | which topic is this paper? | one label per node |
| **edge** | are these two related? | will these users become friends? | one label per node pair |
| **graph** | what is this whole thing? | is this molecule soluble? | one label per graph |

This repo implements **node level** (Cora, experiments 02–05) and **graph
level** (ESOL, experiment 06). Edge level is the same machinery with a scoring
function over node-embedding pairs, e.g. $\mathrm{score}(u,v)=h_u^\top h_v$.

A detail worth internalising: node-level tasks usually have **one big graph**
and most nodes unlabelled (Deck 1 p.59). Training is *transductive* — the
whole graph, including test nodes, goes through the network every epoch; only
the loss is restricted to labelled nodes. Graph-level tasks instead have
**many small graphs**, split into train/test like ordinary supervised data.

## Why you can't just feed the adjacency matrix to an MLP

Deck 1 p.25–26 lists the obstacles. The deepest one:

**Permutation invariance.** Node numbering is arbitrary. Relabel the nodes and
you get a different adjacency matrix $A'=PAP^\top$ describing *the same
graph*. An MLP over a flattened `A` would treat those as unrelated inputs.
With `N` nodes there are $N!$ equivalent matrices; you would need the model to
learn that they all mean the same thing. Hopeless.

What we want instead:

- **invariance** for graph-level outputs: relabel the nodes → same prediction;
- **equivariance** for node-level outputs: relabel the nodes → the predictions
  permute the same way.

Message passing gives both for free, because it is defined purely in terms of
"each node and the *set* of its neighbours" — and a set has no order. The
aggregation function must be order-insensitive (sum, mean, max) for this to
hold. `tests/test_core.py::test_gcn_is_permutation_equivariant` checks it
empirically.

Two more obstacles from the same slides: graphs have **variable size** (an MLP
needs fixed input dimension), and they are **non-Euclidean** — there is no
grid, so "the pixel two steps to the right" has no meaning, only "two hops
away".

## Sparse representation

*Deck 1, p.28–29.*

A dense adjacency matrix costs $O(N^2)$ memory whatever the graph looks like.
Real graphs are sparse: Cora has 2708 nodes but only ~10,500 directed edges —
0.14% of the 7.3M entries of `A`. So we store the **edge list** (COO format):

```python
edge_index = tensor([[0, 1, 1, 2],     # source nodes
                     [1, 0, 2, 1]])    # destination nodes
```

Shape `[2, E]`, cost $O(E)$. Column `k` is one directed edge
`edge_index[0,k] → edge_index[1,k]`. Undirected edges are stored as two
directed ones. This is exactly the format PyTorch Geometric uses, so the habit
transfers.

In this repo: `gnn/graph.py` holds `Graph` (one big graph) and `MolBatch`
(many small graphs stacked), plus `to_undirected`, `add_self_loops` and
`degree`, all operating on `edge_index`.

## The one primitive you need

Everything downstream is built from **scatter-add**: given one message per
edge and the destination node of each, sum the messages per node.

```python
def scatter_sum(src, index, num_nodes):   # src: [E, F], index: [E]
    out = src.new_zeros((num_nodes,) + src.shape[1:])
    return out.index_add_(0, index, src)  # out[i] = Σ_{e: index[e]==i} src[e]
```

That is the whole reason this repo needs no graph library: `index_add_` is
plain PyTorch, it is differentiable, and `scatter_sum` / `scatter_mean` /
`scatter_max` / `segment_softmax` built on it cover every model here.

---

**Continue learning:** [02 — Message passing and GCN](02-message-passing-and-gcn.md)
