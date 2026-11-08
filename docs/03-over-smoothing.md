# 03 — Over-smoothing

> [!NOTE]
> **Study navigation:** [← 02 — Message passing and GCN](02-message-passing-and-gcn.md) · [Repository overview](../README.md#learning-roadmap) · [04 — Edge features and relations →](04-edge-features-and-relations.md)

---

*Slides: Deck 1, p.43–52.*

## The phenomenon

Add layers to a CNN and it generally improves. Add layers to a GNN and after
about 2–3 it falls apart. Measured on Cora (`experiments/03`):

| layers | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| test accuracy | 0.652 | **0.810** | 0.743 | 0.510 | 0.138 | 0.151 | 0.082 |
| MAD | 0.682 | 0.319 | 0.335 | 0.288 | 0.046 | 0.001 | 0.010 |

At 8 layers the model is below the 14% you would get by guessing the majority
class. This is not ordinary overfitting — the training loss is bad too.

## Why it happens

**Reason 1 — the receptive fields merge.** *(Deck 1, p.44.)* After `k` layers
a node aggregates everything within `k` hops. Cora's diameter is small, so by
`k = 5` most nodes' computational graphs cover nearly the same set of nodes.
Two nodes that see the same inputs and apply the same weights produce nearly
the same output.

**Reason 2 — repeated averaging is a low-pass filter.** *(Deck 1, p.46.)*
Strip away `W` and `σ`, and `k` layers apply $\widehat A^k$. $\widehat A$ is symmetric with
eigenvalues in `[-1, 1]` and a largest eigenvalue of exactly 1, whose
eigenvector is the degree-weighted constant vector. So

$$
\widehat A^kX\;\longrightarrow\;\text{projection onto the dominant eigenspace}\quad(k\to\infty)
$$

Every other eigen-component decays like $\lambda_i^k$. The limit is a graph where
every node holds the same vector, scaled by `√degree`. All information that
distinguished nodes has been averaged away. Over-smoothing is this limit
arriving sooner than you expected.

Experiment 01 shows both sides of this on the Karate Club with **no learnable
parameters at all** — just $h\leftarrow\widehat Ah$:

| propagation steps `k` | 1 | 2 | 4 | 6 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| community-clustering accuracy | 0.669 | 0.767 | 0.855 | **0.880** | 0.838 | 0.529 | 0.529 |
| MAD | 0.861 | 0.639 | 0.361 | 0.227 | 0.151 | 0.030 | 0.001 |

Smoothing is *useful* up to a point — it is what surfaces the community
structure — and then it destroys everything. There is a sweet spot, and it is
small.

## Measuring it: MAD and MADGap

*Deck 1, p.47–48.* Accuracy tells you the model is broken but not why. These
two diagnostics measure the collapse directly (Chen et al., AAAI 2020).

**MAD (Mean Average Distance)** — the average cosine distance between node
embeddings:

$$
D_{ij}=1-\cos(h_i,h_j),\qquad
\operatorname{MAD}=\frac{1}{|\mathcal P|}\sum_{(i,j)\in\mathcal P}D_{ij}
$$

High MAD: nodes are distinguishable. MAD → 0: every embedding points the same
way, and no classifier on top can recover anything. In the table above, MAD at
6 layers is 0.001 — total collapse.

**MADGap** — the same thing computed on two different pair sets and
subtracted:

$$
\operatorname{MADGap}=\operatorname{MAD}_{\mathrm{remote}}-\operatorname{MAD}_{\mathrm{near}}
$$

The hypothesis on p.48: messages between nearby nodes carry useful,
same-class signal; messages from distant nodes are mostly noise. A healthy
model therefore pulls true neighbours together (low `MAD_near`) while keeping
distant nodes apart (high `MAD_remote`), so MADGap is large and positive. On
Cora it goes `+0.611` at 1 layer → `−0.004` at 8: near and far have become
indistinguishable.

Implementation is in `gnn/metrics.py`; "near" is the 1-hop mask and "remote"
is everything beyond 3 hops, built by boolean matrix powers of the adjacency.

## What helps

The slides (p.50–52) propose **NGNN**: insert a non-linear feed-forward block
*inside* each message-passing layer, so the network gains depth in parameters
without gaining depth in hops. Available here as `GCNLayer(ngnn=True)`.

Measured honestly, it does not rescue deep GCNs:

| layers | 2 | 4 | 6 | 8 |
|---|---|---|---|---|
| GCN | 0.810 | 0.510 | 0.151 | 0.082 |
| GCN + NGNN | 0.715 | 0.237 | 0.216 | 0.191 |

It helps a little at 6–8 layers (where both are already useless) and *hurts*
at 2–3. The idea is sound — decouple parameter depth from propagation depth —
but an extra MLP alone does not stop the averaging operator from doing its
work.

What actually works in practice, in rough order of value:

1. **Don't go deep.** 2 layers is the right answer for Cora, and for most
   node-classification benchmarks. This is the honest headline.
2. **Residual / skip connections.** $h^{(k+1)}=h^{(k)}+\sigma(\widehat Ah^{(k)}W)$ keeps a
   path back to the un-smoothed features, so the constant vector cannot take
   over. This is what deep GNNs (GCNII, DeepGCN) rely on.
3. **Normalisation between layers** — PairNorm, NodeNorm — explicitly rescale
   embeddings to keep their spread constant, attacking the collapse directly.
4. **Keep the initial features around.** APPNP and GCNII mix $h^{(0)}$ back in
   at every layer, which bounds how far the representation can drift.
5. **Jumping knowledge.** Concatenate the outputs of all layers, so the
   classifier can pick the hop-count it wants per node.

The repo implements (1) and the diagnostics; (2)–(5) are a natural exercise —
adding a residual connection to `GCNLayer` is a two-line change.

---

**Continue learning:** [04 — Edge features and relations](04-edge-features-and-relations.md)
