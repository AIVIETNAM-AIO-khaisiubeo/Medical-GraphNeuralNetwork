"""
Experiment 1 — Zachary's Karate Club: where does the structure come from?

Slide: Deck 2, p.32-37, which claims
  "GCNs can discover community structure even with random parameter values
   before any training."

We test that claim literally, and it turns out to need a correction. What we
measure (numbers from a real run, 12 random seeds):

  A. Untrained GCN, random weights, 2-D output  -> ~53%  (chance is 50%)
  B. Parameter-FREE propagation h <- Â h, k=6   -> ~88%
  C. 2nd eigenvector of Â                       -> ~94%
  D. Trained GCN, only 2 labelled nodes         -> ~97%

The lesson is more precise than the slide's: the community structure is not
created by the random weights, it lives in the **propagation operator** Â
itself. Â is (up to a sign) the normalised adjacency, so applying it
repeatedly is spectral smoothing, and its second eigenvector is exactly what
spectral clustering uses to cut a graph in two. A random GCN *can* expose
that, but a narrow random projection into 2 dimensions usually buries it in
noise.

It also previews Deck 1's over-smoothing topic: keep propagating and accuracy
collapses back to chance while MAD (embedding diversity) goes to 0.

Run:  python experiments/01_karate_untrained.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn import GCNLayer, gcn_normalisation, mad, scatter_sum
from gnn.data import load_karate

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SEEDS = 12


class KarateGCN(nn.Module):
    """34 -> 8 -> 4 -> 2; the 2-D output is plotted directly."""

    def __init__(self):
        super().__init__()
        self.c1, self.c2, self.c3 = GCNLayer(34, 8), GCNLayer(8, 4), GCNLayer(4, 2)

    def forward(self, x, edge_index, edge_weight):
        h = torch.tanh(self.c1(x, edge_index, edge_weight))
        h = torch.tanh(self.c2(h, edge_index, edge_weight))
        return self.c3(h, edge_index, edge_weight)


def cluster_accuracy(emb: np.ndarray, y: np.ndarray) -> float:
    """Split the embedding into 2 clusters; score against the true factions.
    Cluster ids are arbitrary, so take the better of the two labellings."""
    pred = KMeans(2, n_init=10, random_state=0).fit_predict(emb)
    acc = float((pred == y).mean())
    return max(acc, 1.0 - acc)


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    g = load_karate()
    n, y = g.num_nodes, g.y.numpy()
    edge_index, edge_weight = gcn_normalisation(g.edge_index, n)

    print(f"Karate club: {n} members, {g.num_edges // 2} friendships, 2 factions")
    print(f"Labelled for training: {int(g.train_mask.sum())} nodes "
          f"(only leaders 0 and 33)\n")

    # ---- A. untrained GCN with random weights --------------------------
    accs = []
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        m = KarateGCN().eval()
        with torch.no_grad():
            accs.append(cluster_accuracy(m(g.x, edge_index, edge_weight).numpy(), y))
    print(f"A. untrained GCN (random weights, 2-D out): "
          f"{np.mean(accs):.3f} +/- {np.std(accs):.3f}   <- barely above chance (0.5)")

    # ---- B. parameter-free propagation  h <- Â h -----------------------
    def propagate(h):
        """One message-passing step with NO learnable parameters at all."""
        return scatter_sum(h[edge_index[0]] * edge_weight.unsqueeze(1), edge_index[1], n)

    ks = list(range(1, 33))
    acc_by_k = {k: [] for k in ks}
    mad_by_k = {k: [] for k in ks}
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        h = torch.randn(n, 16)
        for k in ks:
            h = propagate(h)
            acc_by_k[k].append(cluster_accuracy(h.numpy(), y))
            mad_by_k[k].append(mad(h))
    mean_acc = {k: float(np.mean(v)) for k, v in acc_by_k.items()}
    mean_mad = {k: float(np.mean(v)) for k, v in mad_by_k.items()}
    best_k = max(ks, key=lambda k: mean_acc[k])
    print(f"B. parameter-free propagation, best at k={best_k} steps:        "
          f"{mean_acc[best_k]:.3f}   <- the structure is in Â, not the weights")
    print(f"     k=1: {mean_acc[1]:.3f}   k={best_k}: {mean_acc[best_k]:.3f}   "
          f"k=16: {mean_acc[16]:.3f}   k=32: {mean_acc[32]:.3f}  <- over-smoothing")
    print(f"     MAD  {mean_mad[1]:.3f} -> {mean_mad[32]:.4f} "
          f"(embeddings collapse onto one vector)")

    # ---- C. spectral view: 2nd eigenvector of Â ------------------------
    A = torch.zeros(n, n)
    A[edge_index[0], edge_index[1]] = edge_weight
    evals, evecs = torch.linalg.eigh(A)
    order = torch.argsort(evals, descending=True)
    v2 = evecs[:, order[1]].numpy()
    acc_v2 = max(float(((v2 > 0).astype(int) == y).mean()),
                 1.0 - float(((v2 > 0).astype(int) == y).mean()))
    print(f"C. sign of Â's 2nd eigenvector:                     {acc_v2:.3f}   "
          f"<- classic spectral clustering")

    # ---- D. train the GCN on just 2 labels -----------------------------
    torch.manual_seed(1)
    model = KarateGCN()
    opt = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)
    for _ in range(200):
        model.train()
        opt.zero_grad()
        out = model(g.x, edge_index, edge_weight)
        F.cross_entropy(out[g.train_mask], g.y[g.train_mask]).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        emb_after = model(g.x, edge_index, edge_weight)
    acc_sup = float((emb_after.argmax(1) == g.y).float().mean())
    print(f"D. trained GCN, 2 labelled nodes:                   {acc_sup:.3f}   "
          f"<- labels only pick which side is which")

    # ---- figure --------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    torch.manual_seed(0)
    h = torch.randn(n, 16)
    for _ in range(best_k):
        h = propagate(h)
    hc = (h - h.mean(0)).numpy()
    proj = hc @ np.linalg.svd(hc, full_matrices=False)[2][:2].T
    for f, c, lbl in [(0, "#2a6fdb", "Mr. Hi"), (1, "#d6453d", "John A")]:
        m = y == f
        axes[0].scatter(proj[m, 0], proj[m, 1], c=c, s=85, label=lbl,
                        edgecolors="white", linewidths=1.1)
    axes[0].set_title(f"B. Parameter-free propagation (k={best_k})\n"
                      f"cluster accuracy {mean_acc[best_k]:.0%}")
    axes[0].legend(); axes[0].set_xticks([]); axes[0].set_yticks([])

    e = emb_after.detach().numpy()
    for f, c, lbl in [(0, "#2a6fdb", "Mr. Hi"), (1, "#d6453d", "John A")]:
        m = y == f
        axes[1].scatter(e[m, 0], e[m, 1], c=c, s=85, label=lbl,
                        edgecolors="white", linewidths=1.1)
    for i in (0, 33):
        axes[1].annotate(str(i), (e[i, 0], e[i, 1]), fontsize=9, fontweight="bold")
    axes[1].set_title(f"D. Trained GCN, 2 labels\naccuracy {acc_sup:.0%}")
    axes[1].legend(); axes[1].set_xticks([]); axes[1].set_yticks([])

    ax2 = axes[2]
    ax2.plot(ks, [mean_acc[k] for k in ks], "o-", color="#2a6fdb", label="cluster accuracy")
    ax2.axhline(0.5, ls="--", c="grey", lw=1, label="chance")
    ax2.set_xlabel("propagation steps k"); ax2.set_ylabel("accuracy", color="#2a6fdb")
    ax2.set_title("More hops is not better:\nstructure emerges, then over-smooths")
    twin = ax2.twinx()
    twin.plot(ks, [mean_mad[k] for k in ks], "s-", color="#d6453d", ms=4, label="MAD")
    twin.set_ylabel("MAD (embedding diversity)", color="#d6453d")
    ax2.legend(loc="upper right", fontsize=8); ax2.grid(alpha=.3)

    plt.tight_layout()
    out = os.path.join(RESULTS, "01_karate.png")
    plt.savefig(out, dpi=130)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
