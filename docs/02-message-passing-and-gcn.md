# 02 — Message passing and GCN

> [!NOTE]
> **Study navigation:** [← 01 — Graph data and tasks](01-graph-data-and-tasks.md) · [Repository overview](../README.md#learning-roadmap) · [03 — Over-smoothing →](03-over-smoothing.md)

---

*Slides: Deck 1 p.31–56; Deck 2 p.11–31.*

## The general form

*Deck 1, p.53.* Every GNN layer in this repo is an instance of:

$$
h_u^{(k+1)} = \operatorname{UPDATE}^{(k)}\!\left(h_u^{(k)},
\operatorname{AGGREGATE}^{(k)}\!\left(\{h_v^{(k)}:v\in\mathcal N(u)\}\right)\right)
$$

Three steps, named on Deck 1 p.31:

1. **Sample the neighbourhood** — which nodes send to `u`.
2. **Aggregate** — combine their vectors with an order-insensitive function.
3. **Update** — combine the result with `u`'s own vector, apply weights and a
   non-linearity.

Models differ only in the choice of those functions. That is the entire design
space.

## Depth = hops, not abstraction

*Deck 1, p.41, p.44.*

After one layer, `u` knows its 1-hop neighbourhood. After two, its neighbours
have already absorbed *their* neighbours, so `u` indirectly sees 2 hops. The
**computational graph** of node `u` is a tree of depth `k` rooted at `u`.

This is the sharpest difference from a CNN. In a CNN, more depth means more
abstraction and the receptive field grows slowly and controllably. In a GNN,
each layer multiplies the receptive field by the average degree — after 5 hops
on a social graph you may already be averaging over half the network. That is
why GNNs are shallow (2–4 layers) and why [over-smoothing](03-over-smoothing.md)
is the central failure mode.

## Deriving $\widehat A$

Start with the crudest possible aggregation: sum the neighbours. In matrix
form that is $AX$. Two things are wrong with it (Deck 2, p.22–24):

**Problem 1 — the node forgets itself.** Row `i` of $AX$ contains only
neighbours of `i`; `x_i` is absent. Fix: add self-loops,

$$
\widetilde A=A+I
$$

**Problem 2 — magnitudes scale with degree.** A node with 50 neighbours
produces activations ~50× larger than a node with 1. Stack layers and this
explodes (or vanishes). Fix: normalise by degree.

The obvious normalisation is the row-average, `D̃⁻¹ Ã` — "take the mean of
your neighbours". Deck 2 p.13–14 shows the alternative: scale by rows *and*
columns, giving the symmetric normalisation GCN actually uses:

$$
\widehat A=\widetilde D^{-1/2}\widetilde A\widetilde D^{-1/2},
\qquad
\widehat A_{ij}=\frac{\widetilde A_{ij}}{\sqrt{\widetilde d_i\,\widetilde d_j}}
$$

Why symmetric rather than the row-mean? Two reasons:

- **A high-degree neighbour's message is discounted.** With the row-mean, a
  message from a hub counts as much as one from a rare, specific node. With
  the symmetric form it is divided by `√d_j` too — information from a node
  that talks to everyone is less distinctive.
- **$\widehat A$ stays symmetric**, so its eigenvalues are real and lie in `[-1, 1]`.
  Repeated application neither explodes nor vanishes. This is also what makes
  the spectral story in experiment 01 work.

The layer is then:

$$
H^{(k+1)}=\sigma\!\left(\widehat A H^{(k)}W^{(k)}\right)
$$

`W` is the only learnable part; $\widehat A$ is fixed by the graph and computed once.

## From matrix form to the code

We never build the $N\times N$ matrix. `Â_ij = 1/√(d_i d_j)` is a scalar **per
edge**, so the matrix product becomes a weighted scatter-add:

```python
# gnn/graph.py
def gcn_normalisation(edge_index, num_nodes):
    ei  = add_self_loops(edge_index, num_nodes)     # Ã = A + I
    deg = degree(ei, num_nodes)
    dinv = deg.pow(-0.5); dinv[torch.isinf(dinv)] = 0.0
    return ei, dinv[ei[0]] * dinv[ei[1]]            # one weight per edge

# gnn/layers.py
class GCNLayer(nn.Module):
    def forward(self, x, edge_index, edge_weight):
        h   = self.lin(x)                                   # H W
        msg = h[edge_index[0]] * edge_weight.unsqueeze(1)    # Â_ij · (HW)_j
        return scatter_sum(msg, edge_index[1], x.shape[0])   # Σ_j
```

Three lines. `tests/test_core.py::test_gcn_matches_dense` asserts this equals
the dense $\widehat AHW$ to within 1e-5, so the shortcut is exact, not approximate.

Note the ordering: project **first** (`self.lin(x)`, one matmul over `N`
nodes), then gather. Gathering first would copy `F`-dimensional vectors for
every edge and cost far more when `E ≫ N`.

## CNN vs GNN

*Deck 1 p.35; Deck 2 p.25–31.*

A 3×3 convolution computes a weighted sum over a pixel and its 8 neighbours —
with a **different weight per relative position**: the kernel's top-left entry
always multiplies the top-left pixel.

A GCN cannot do that, because "top-left" does not exist on a graph. All
neighbours share the *same* `W`, and the only per-neighbour modulation is the
scalar $\widehat A_{ij}$ fixed by degrees. In that sense a GCN layer is a convolution
whose kernel has been forced to be isotropic.

That is a real loss of expressive power, and it is what the rest of the
course recovers:

- [GAT](05-attention-and-graph-level.md) learns a per-edge scalar $\alpha_{ij}$
  from the features instead of fixing it from degrees;
- [R-GCN](04-edge-features-and-relations.md) gives each *relation type* its
  own $W_r$, which is the closest analogue to "a different weight per
  position";
- [`EdgeMPNNLayer`](04-edge-features-and-relations.md) conditions the whole
  message on the edge's feature vector.

## Measured

Cora, 140 labelled nodes out of 2708 (`experiments/02`):

| model | test accuracy |
|---|---|
| MLP (no graph) | 0.537 |
| GCN, 2 layers | **0.810** |

Same features, same hidden width, same budget. The only difference is $\widehat A$.

---

**Continue learning:** [03 — Over-smoothing](03-over-smoothing.md)
