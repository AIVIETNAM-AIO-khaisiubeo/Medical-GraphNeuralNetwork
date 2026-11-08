# 04 — Edge features, edge weights, and relations

*Slides: Deck 2, p.4–53.*

## The gap in plain GCN

In `h_u' = σ(Σ_j Â_uj W h_j)` the only thing an edge contributes is the scalar
`Â_uj`, and even that is fixed by node degrees. The edge itself — what *kind*
of relation it is, how strong, what properties it has — is invisible.

Deck 2 opens with the example on p.4–8: predicting whether someone likes
watching films, from a small social graph. Her friends' features matter, but
so does *how* she is connected to them — a family tie, a colleague, someone
she lives with. An adjacency matrix of 0s and 1s throws all of that away.

There are three increasingly general ways to put it back.

## 1. Edge weights — a scalar per edge

*Deck 2, p.11–14.*

The cheapest option: replace the binary adjacency with a weighted one, where
`w_uv` is interaction strength (how often two people talk, bond order,
similarity). Normalise it the same way and aggregate:

```
h_u' = σ( Σ_j  w_uj / √(d_u d_j) · W h_j )
```

No architectural change at all — every layer in this repo already takes an
`edge_weight` argument, and `gcn_normalisation()` just produces one particular
choice of it. Use your own weights instead and everything else carries over.

Limit: one number per edge. It can say *how much* a neighbour matters, never
*in what way*.

## 2. Relations — a weight matrix per edge type

*Deck 2, p.38–46. Implemented as `RGCNLayer`.*

When edges come in discrete types — friend / family / colleague, or in a
knowledge graph `born_in` / `works_for` / `married_to` — the right move is to
give each relation `r` its own transform:

```
h_u' = W_0 h_u  +  Σ_r  (1 / |N_u^r|)  Σ_{v ∈ N_u^r}  W_r h_v
```

- `W_0 h_u` is the self-transform (the self-loop, now explicit);
- one `W_r` per relation, so a message along a *family* edge is transformed
  differently from one along a *colleague* edge;
- `1/|N_u^r|` normalises *within* each relation, so a relation with many
  neighbours cannot swamp a rare but informative one.

The cost is parameters: `R` relations means `R` weight matrices. Knowledge
graphs with hundreds of relation types need basis decomposition (share a small
set of basis matrices and learn per-relation coefficients) — not implemented
here, but it is the standard fix from the R-GCN paper.

### Why this is not a detail

`experiments/05` builds a graph where one relation carries the answer and two
are noise: a node's label is the signal of its single `family` neighbour,
while `friend` and `colleague` add ~8 random neighbours each.

| model | test accuracy |
|---|---|
| GCN (relation-blind) | 0.322 |
| GAT (attention over node features) | 0.439 |
| **R-GCN** | **1.000** |
| chance | 0.333 |

GCN sits at chance — it averages the one useful neighbour together with eight
noisy ones. **Attention does not rescue it either**, which is the subtle
point: GAT computes `α_ij` from the *features* of `i` and `j`, and here the
noisy neighbours look exactly like the informative one. The information
needed is the edge's type, and GAT never sees it.

Ablating the trained R-GCN confirms it learned the right thing:

| edges removed at inference | test accuracy |
|---|---|
| none | 1.000 |
| `family` | 0.339 |
| `friend` | 0.994 |
| `colleague` | 1.000 |

(A caution learned while writing this: the *norm* of each `W_r` is a useless
diagnostic — all three came out within a few percent of each other. Ablation
is the measurement that actually answers the question.)

## 3. Multi-dimensional edge features — an MLP on the edge

*Deck 2, p.48–50. Implemented as `EdgeMPNNLayer`.*

When an edge carries a whole feature *vector* — bond type + conjugation + ring
membership, or "friend, since 2019, lives together" — discrete relation types
no longer suffice. Make the message a learned function of both endpoints and
the edge:

```
m_uv = MLP([ h_u ‖ h_v ‖ e_uv ])
h_u' = ReLU( W_self h_u + mean_{v ∈ N(u)} m_uv )
```

This is the Message Passing Neural Network (MPNN) form, and it subsumes the
previous two: edge weights are the case where `e_uv` is a scalar, relations
the case where `e_uv` is a one-hot type.

```python
# gnn/layers.py
class EdgeMPNNLayer(nn.Module):
    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        m   = self.msg(torch.cat([x[dst], x[src], edge_attr], dim=1))
        agg = scatter_mean(m, dst, x.shape[0])
        return F.relu(self.self_lin(x) + agg)
```

The cost: the MLP runs **per edge**, not per node, so this is the most
expensive option. On large graphs it dominates runtime.

### Measured on ESOL

`experiments/06`, identical architecture trained twice — once with real bond
features, once with them zeroed:

| | test RMSE |
|---|---|
| bond features zeroed | 0.937 |
| bond features used | **0.864** |

Worth 0.072 RMSE (7.7% relative). Real, but smaller than Deck 2's emphasis
might lead you to expect — for solubility, *which* atoms are present matters
more than exactly how they are bonded. Notably, in the same experiment the
choice of readout (sum vs mean pooling) was worth four times as much. Edge
features are one knob among several, and not automatically the biggest.

## Other variants

The slides also mention two approaches not implemented here:

- **FiLM** (p.45) — the edge type generates a scale and shift applied to the
  message, `γ_r ⊙ (W h_v) + β_r`, instead of a whole matrix per relation.
  Cheaper than R-GCN for many relation types.
- **NENN** (p.51–52) — maintain embeddings for edges *and* nodes, updating
  each from the other with dual-level attention. Useful when edges are
  first-class objects with rich features of their own.

---

Next: [05 — Attention and graph-level prediction](05-attention-and-graph-level.md)
