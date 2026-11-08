# Graph Neural Networks from Scratch

A complete, runnable implementation of the two GNN lecture decks
(*Introduction to Graph Neural Network* and *Advanced Graph Neural Network*
by Vinh Dinh Nguyen) — **written from scratch in plain PyTorch**.

> **No graph libraries.** No PyTorch Geometric, no DGL, no NetworkX, no
> torch-scatter. Every layer, the sparse graph representation, mini-batching
> and the training loops are implemented here. The only graph primitive used
> is `Tensor.index_add_` (scatter-add), which is plain PyTorch.
>
> RDKit appears in one experiment, purely to turn SMILES strings into atoms
> and bonds — chemistry parsing, not graph learning.

Written for someone who already knows neural networks, backprop and
Transformers. It does not re-explain those; it explains what is *different*
about graphs, and shows the difference in measured numbers.

---

## The one idea

A neural network wants a fixed-size, ordered input. A graph has neither: it
has a variable number of nodes and **no canonical ordering**. Relabelling the
nodes must not change the answer (*permutation invariance*), so you cannot
simply flatten an adjacency matrix and feed it to an MLP.

The fix is to make the network operate **locally and identically everywhere** —
each node repeatedly looks at its neighbours and updates itself:

```
h_u^(k+1) = UPDATE( h_u^(k),  AGGREGATE({ h_v^(k) : v ∈ N(u) }) )
```

That is **message passing**, and every model here is one choice of
`AGGREGATE` and `UPDATE`:

| Layer | message from v to u | how it is weighted | file |
|---|---|---|---|
| `GCNLayer` | `W h_v` | fixed: `1/√(d_u d_v)` | `gnn/layers.py` |
| `GATLayer` | `W h_v` | learned attention `α_uv` | `gnn/layers.py` |
| `RGCNLayer` | `W_r h_v` (one `W` per relation) | `1/|N_u^r|` | `gnn/layers.py` |
| `EdgeMPNNLayer` | `MLP([h_u ‖ h_v ‖ e_uv])` | mean | `gnn/layers.py` |

**If you know Transformers:** a GNN layer is attention with a *mask fixed by
the adjacency matrix*. A Transformer lets every token attend to every other
token; a GNN only lets a node attend to its neighbours. GAT makes this exact —
it is self-attention restricted to the graph, with an additive score instead
of a scaled dot product. Self-attention is the special case where the graph
is fully connected.

---

## Results

Every number below was produced by the scripts in `experiments/` on this
repo's code. Figures land in `results/`.

### Node classification — Cora (2708 papers, 1433-dim bag-of-words, 7 topics)

| model | what it sees | test accuracy |
|---|---|---|
| MLP | features only, no graph | **0.537** |
| GCN — slide version (2 MP layers + linear head) | features + citations | **0.765** |
| GCN — canonical (2 MP layers, 2nd emits logits) | features + citations | **0.810** |
| GAT (8 heads) | features + citations + attention | **0.818** |

Only 140 of 2708 nodes (5%) are labelled. **The citation graph is worth +0.27
accuracy — a 51% relative improvement** over the identical model without it.
That gap is the whole argument for GNNs.

The canonical GCN's 0.810 matches the published Cora result (~0.815), which is
the main evidence that this from-scratch implementation is correct.

![Cora](results/02_cora_mlp_vs_gcn.png)

### Over-smoothing — depth is not free

| layers | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| test accuracy | 0.652 | **0.810** | 0.743 | 0.510 | 0.138 | 0.151 | 0.082 |
| MAD | 0.682 | 0.319 | 0.335 | 0.288 | 0.046 | 0.001 | 0.010 |

Accuracy peaks at **2** layers and then collapses. MAD (how different node
embeddings are from one another) falls towards 0: every node is converging to
the same vector. See [docs/03-over-smoothing.md](docs/03-over-smoothing.md).

### Relations — when the *type* of an edge matters

A synthetic graph where one relation (`family`) carries the label and two
others are noise:

| model | test accuracy |
|---|---|
| GCN (relation-blind) | 0.322 |
| GAT (attention over features) | 0.439 |
| **R-GCN (one `W` per relation)** | **1.000** |
| chance | 0.333 |

Ablating the trained R-GCN confirms what it learned: removing `family` edges
drops it to 0.339, removing `friend` or `colleague` costs ~0.00.

### Graph-level regression — ESOL (1128 molecules, water solubility)

| model | test RMSE |
|---|---|
| predict the training mean | 2.087 |
| GNN, bond features zeroed | 0.937 |
| GNN + bond features, mean pooling | 0.864 |
| **GNN + bond features, sum pooling** | **0.621** |
| GNN + bond features, max pooling | 0.980 |

Edge (bond) features are worth 0.072 RMSE. The **readout** choice matters more:
sum pooling beats mean by a wide margin because solubility depends on molecular
*size*, which mean pooling divides away.

---

## Where the slides and the measurements disagree

Three places where running the code gave a different answer than a literal
reading of the slides. These are the most interesting parts of the repo.

**1. "GCNs discover community structure with random weights" (Deck 2, p.36).**
Measured over 12 seeds, an untrained GCN on the Karate Club scores **0.529** —
chance is 0.5. The claim as stated does not reproduce. What *is* true: the
structure lives in the propagation operator `Â`, not in the random weights.
Parameter-free propagation `h ← Â h` reaches **0.880**, and the sign of `Â`'s
second eigenvector gives **0.941** — that is just spectral clustering. A random
projection into 2 dimensions usually buries it. See `experiments/01`.

**2. The slide's GCN architecture costs ~4.5 accuracy points.** Deck 1 p.71
specifies "2 Message Passing Layers and one Linear output layer", i.e.
`1433 → 16 → 16 → 7`. That pushes the features through *two* 16-dim
bottlenecks and scores **0.765**. The canonical GCN, where the second
message-passing layer emits the 7 class scores directly (`1433 → 16 → 7`),
scores **0.810**. Both are implemented (`head="linear"` vs `head="direct"`).

**3. GAT's attention barely does anything on Cora.** GAT does edge out GCN
(0.818 vs 0.810), but the learned attention is almost uniform — `α_ij` divided
by the uniform weight `1/degree` sits between 0.98 and 1.02 in layer 1. The
attention parameters do move substantially during training, so this is not a
bug: Cora is strongly *homophilous*, nearly every citation joins two papers of
the same topic, so there is little for attention to down-weight. Experiment 05
builds a graph where neighbour quality does vary — and there attention is not
enough either; you need the edge *type*.

---

## Quickstart

```bash
pip install -r requirements.txt
python scripts/download_data.py      # Cora + ESOL, plain text/CSV
python tests/test_core.py            # verify the implementation
python experiments/02_cora_mlp_vs_gcn.py
```

`tests/test_core.py` is worth running first. Its key check asserts that the
sparse scatter-based `GCNLayer` produces *exactly* the same output as the dense
matrix product `Â H W`, and that the layer is permutation-equivariant.

```
  to_undirected               ok
  scatter_sum                 ok
  segment_softmax             ok
  GCNLayer == dense Â H W     ok
  permutation equivariance    ok
  ...
  ALL TESTS PASSED
```

### Experiments

| script | what it shows | runtime (CPU) |
|---|---|---|
| `01_karate_untrained.py` | where community structure actually comes from | ~1 min |
| `02_cora_mlp_vs_gcn.py` | **the main result**: MLP vs GCN, t-SNE before/after | ~2 min |
| `03_oversmoothing.py` | depth sweep, MAD / MADGap collapse | ~8 min |
| `04_cora_gat.py` | attention vs fixed weights | ~4 min |
| `05_rgcn_relations.py` | when edge *type* decides everything | ~1 min |
| `06_esol_graph_regression.py` | graph-level readout, edge features, pooling | ~6 min |

---

## Repo layout

```
gnn/
  graph.py      sparse COO representation, scatter ops, Â normalisation,
                batching of many graphs, global pooling
  layers.py     GCNLayer, GATLayer, RGCNLayer, EdgeMPNNLayer
  models.py     MLP, GCN, GAT, RGCN, GraphRegressor
  metrics.py    accuracy, RMSE, MAD, MADGap, k-hop masks
  train.py      node-classification and graph-regression training loops
  data/         cora.py, karate.py, relational.py, esol.py
experiments/    the six scripts above
docs/           the theory, mapped to the slides
tests/          correctness tests
scripts/        download_data.py
results/        generated figures
```

## Theory notes

Read in order; each is short and maps directly onto the slides.

| doc | covers | slides |
|---|---|---|
| [00 — Tóm tắt tiếng Việt](docs/00-tom-tat-tieng-viet.md) | Vietnamese summary of the whole theory | both |
| [01 — Graph data and tasks](docs/01-graph-data-and-tasks.md) | what a graph is, node/edge/graph tasks, why an MLP can't eat a graph, sparse representation | Deck 1 p.5-29 |
| [02 — Message passing and GCN](docs/02-message-passing-and-gcn.md) | AGGREGATE/UPDATE, computational graphs, the `Â` normalisation derived, CNN vs GNN | Deck 1 p.31-56, Deck 2 p.11-31 |
| [03 — Over-smoothing](docs/03-over-smoothing.md) | why depth hurts, MAD and MADGap, what actually helps | Deck 1 p.43-52 |
| [04 — Edge features and relations](docs/04-edge-features-and-relations.md) | edge weights, R-GCN, multi-dimensional edge features | Deck 2 p.4-53 |
| [05 — Attention and graph-level prediction](docs/05-attention-and-graph-level.md) | GAT as masked self-attention, readout/pooling | Deck 2 p.55-71 |

## Slide → code map

| slide topic | implemented in |
|---|---|
| Sparse graph representation, adjacency list (D1 p.28-29) | `gnn/graph.py` |
| Message passing: sample → aggregate → update (D1 p.31-32) | `gnn/layers.py`, `scatter_*` in `graph.py` |
| `h_u^(k+1) = UPDATE(h_u, AGGREGATE(...))` (D1 p.53) | every layer in `gnn/layers.py` |
| `Â = D^-1/2 (A+I) D^-1/2` (D1 p.23, D2 p.22-24) | `gcn_normalisation()` |
| Over-smoothing, MAD, MADGap (D1 p.43-48) | `gnn/metrics.py`, `experiments/03` |
| NGNN remedy (D1 p.50-52) | `GCNLayer(ngnn=True)` |
| Cora MLP vs GCN (D1 p.59-72) | `experiments/02` |
| Karate club (D2 p.32-37) | `experiments/01` |
| Edge weights (D2 p.11-14) | `edge_weight` argument throughout |
| Relational GNN (D2 p.38-46) | `RGCNLayer`, `experiments/05` |
| Multi-dimensional edge features (D2 p.48-50) | `EdgeMPNNLayer` |
| Attention / GAT (D2 p.55-59) | `GATLayer`, `experiments/04` |
| Graph-level prediction, ESOL (D2 p.61-71) | `GraphRegressor`, `experiments/06` |

## Not covered

Deck 2 ends with Vision GNN (p.72-81), treating an image as a graph of
patches. Not implemented here — it needs a CIFAR-10 training pipeline that
would dwarf the rest of the repo. `EdgeMPNNLayer` plus a k-NN graph over patch
embeddings is the starting point if you want to add it.

## References

- Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks*, ICLR 2017.
- Veličković et al., *Graph Attention Networks*, ICLR 2018.
- Schlichtkrull et al., *Modeling Relational Data with Graph Convolutional Networks*, ESWC 2018.
- Chen et al., *Measuring and Relieving the Over-smoothing Problem*, AAAI 2020 (MAD / MADGap).
- Gilmer et al., *Neural Message Passing for Quantum Chemistry*, ICML 2017.
- Datasets: Cora (LINQS, via the `pygcn` mirror), ESOL/Delaney (DeepChem).
