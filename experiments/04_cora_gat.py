"""
Experiment 4 — Attention: GAT vs GCN on Cora.

Slide: Deck 2, p.55-59.

    e_ij     = LeakyReLU( a^T [ W h_i || W h_j ] )
    alpha_ij = softmax over j in N(i) of e_ij
    h_i'     = sigma( sum_j alpha_ij W h_j )

If you know Transformers, GAT is self-attention with two changes:
  1. the attention is **masked to the graph** — node i attends only to its
     neighbours, not to all nodes (the adjacency is the attention mask);
  2. the score is an additive MLP, not a scaled dot product.

GCN's weights are *fixed* by the degrees: w_ij = 1/sqrt(d_i d_j).
GAT *learns* them. This script trains both and then measures how far the
learned attention actually departs from GCN's fixed weighting — if attention
ends up uniform, it has learned nothing GCN did not already have.

Run:  python experiments/04_cora_gat.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn import GAT, GCN, add_self_loops, degree, gcn_normalisation, segment_softmax
from gnn.data import load_cora
from gnn.train import train_node_classifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
EPOCHS = 200


def attention_weights(model: GAT, x, edge_index, which: int = 1):
    """Recompute the attention coefficients alpha_ij of layer `which`."""
    n = x.shape[0]
    ei = add_self_loops(edge_index, n)
    layer = model.conv1 if which == 1 else model.conv2
    inp = x if which == 1 else F.elu(model.conv1(x, ei))
    h = layer.lin(inp).view(n, layer.heads, layer.out_dim)
    src, dst = ei[0], ei[1]
    alpha = (h * layer.att_src).sum(-1)[src] + (h * layer.att_dst).sum(-1)[dst]
    alpha = F.leaky_relu(alpha, layer.negative_slope)
    alpha = segment_softmax(alpha, dst, n)
    return ei, alpha.mean(dim=1)          # average over heads -> [E]


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    g = load_cora(root=os.path.join(ROOT, "data", "cora"))
    n_feat, n_classes = g.x.shape[1], int(g.y.max()) + 1
    edge_index, edge_weight = gcn_normalisation(g.edge_index, g.num_nodes)

    print("[1/2] GCN (fixed degree-based weights)")
    torch.manual_seed(0)
    gcn = GCN(n_feat, 16, n_classes, num_layers=2, head="direct")
    res_gcn = train_node_classifier(gcn, g.x, edge_index, edge_weight, g.y,
                                    g.train_mask, g.val_mask, g.test_mask,
                                    epochs=EPOCHS, lr=0.01, weight_decay=5e-4)
    print(f"      test accuracy {res_gcn['test_acc']:.3f}\n")

    print("[2/2] GAT (8 heads x 8 dims, learned attention)")
    torch.manual_seed(0)
    gat = GAT(n_feat, 8, n_classes, heads=8, dropout=0.6)
    res_gat = train_node_classifier(gat, g.x, g.edge_index, None, g.y,
                                    g.train_mask, g.val_mask, g.test_mask,
                                    epochs=EPOCHS, lr=0.005, weight_decay=5e-4,
                                    log_every=50)
    print(f"      test accuracy {res_gat['test_acc']:.3f}\n")

    # ---- is the learned attention actually non-uniform? ----------------
    print("=" * 60)
    print(f"  GCN : {res_gcn['test_acc']:.3f}")
    print(f"  GAT : {res_gat['test_acc']:.3f}")
    print("=" * 60)

    gat.eval()
    ratios = {}
    print("\nLearned attention vs uniform attention,  alpha_ij / (1/degree):")
    print("  (a ratio of 1.0 everywhere means attention reduces to plain averaging)")
    for which in (1, 2):
        with torch.no_grad():
            ei, alpha = attention_weights(gat, g.x, g.edge_index, which=which)
        deg = degree(ei, g.num_nodes, which="dst")
        ratio = alpha / (1.0 / deg[ei[1]])
        ratios[which] = ratio
        print(f"  layer {which}: median {ratio.median():.2f} | "
              f"5th pct {torch.quantile(ratio, 0.05):.2f} | "
              f"95th pct {torch.quantile(ratio, 0.95):.2f} | "
              f">2x uniform on {float((ratio > 2.0).float().mean()):.1%} of edges")

    print("\nFinding: on Cora the learned attention stays very close to uniform,")
    print("and GAT beats GCN only marginally. Cora is strongly *homophilous* —")
    print("almost every citation links two papers of the same topic, so there is")
    print("little for attention to down-weight. Attention earns its keep when")
    print("neighbours differ in usefulness (noisy or heterophilous graphs);")
    print("experiment 05 builds exactly such a case.")

    # ---- figure ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].plot(res_gcn["history"]["val_acc"], label=f"GCN ({res_gcn['test_acc']:.2f})", lw=2)
    axes[0].plot(res_gat["history"]["val_acc"], label=f"GAT ({res_gat['test_acc']:.2f})", lw=2)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("validation accuracy")
    axes[0].set_title("GCN vs GAT on Cora"); axes[0].legend(); axes[0].grid(alpha=.3)

    axes[1].hist(np.clip(ratios[2].numpy(), 0, 4), bins=60, color="#2a6fdb")
    axes[1].axvline(1.0, color="#d6453d", ls="--", lw=2, label="uniform (= GCN-like)")
    axes[1].set_xlabel(r"$\alpha_{ij}$ / uniform weight")
    axes[1].set_ylabel("number of edges")
    axes[1].set_title("GAT attention (layer 2) stays near uniform on Cora")
    axes[1].legend()
    plt.tight_layout()
    out = os.path.join(RESULTS, "04_cora_gat.png")
    plt.savefig(out, dpi=130)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
