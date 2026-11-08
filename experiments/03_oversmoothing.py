"""
Experiment 3 — Over-smoothing: why deeper GNNs get worse.

Slide: Deck 1, p.43-52.
  * Depth in a GNN is not like depth in a CNN. Each extra message-passing
    layer widens the receptive field by one hop, so nodes' computational
    graphs start to overlap (p.44) and their embeddings converge.
  * Diagnostics (p.47-48): MAD (Mean Average Distance) and MADGap.
  * Suggested remedy (p.50-52): insert a non-linear feed-forward block
    *inside* each GNN layer ("NGNN") so the network gets deeper in
    parameters without getting deeper in hops.

We sweep depth 1..8 and record test accuracy, MAD and MADGap for a plain GCN
and for the NGNN variant.

What to look for
  * accuracy peaks at 2 layers and then falls away;
  * MAD falls monotonically towards 0 — embeddings are collapsing;
  * MADGap shrinks too: near and far nodes become indistinguishable.

Run:  python experiments/03_oversmoothing.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn import GCN, gcn_normalisation, mad
from gnn.data import load_cora
from gnn.metrics import khop_neighbour_mask
from gnn.train import train_node_classifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

DEPTHS = [1, 2, 3, 4, 5, 6, 8]
HIDDEN = 16
EPOCHS = 200


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    g = load_cora(root=os.path.join(ROOT, "data", "cora"))
    edge_index, edge_weight = gcn_normalisation(g.edge_index, g.num_nodes)
    n_classes = int(g.y.max()) + 1
    n_feat = g.x.shape[1]

    print("building k-hop masks for MADGap (once) ...")
    near = khop_neighbour_mask(g.edge_index, g.num_nodes, hops=1)
    within3 = khop_neighbour_mask(g.edge_index, g.num_nodes, hops=3)
    remote = ~within3
    remote.fill_diagonal_(False)
    print(f"  near pairs: {int(near.sum()):,}   remote pairs: {int(remote.sum()):,}\n")

    rows = []
    for ngnn in (False, True):
        tag = "GCN+NGNN" if ngnn else "GCN"
        for depth in DEPTHS:
            torch.manual_seed(0)
            model = GCN(n_feat, HIDDEN, n_classes, num_layers=depth,
                        dropout=0.5, ngnn=ngnn, head="direct")
            res = train_node_classifier(
                model, g.x, edge_index, edge_weight, g.y,
                g.train_mask, g.val_mask, g.test_mask,
                epochs=EPOCHS, lr=0.01, weight_decay=5e-4)
            model.eval()
            with torch.no_grad():
                _, emb = model(g.x, edge_index, edge_weight, return_embedding=True)
            m_all = mad(emb)
            gap = mad(emb, remote) - mad(emb, near)
            rows.append((tag, depth, res["test_acc"], m_all, gap))
            print(f"  {tag:9s} depth {depth}: test acc {res['test_acc']:.3f} | "
                  f"MAD {m_all:.4f} | MADGap {gap:+.4f}")
        print()

    print("=" * 62)
    print(f"{'model':10s} {'depth':>5} {'test acc':>9} {'MAD':>8} {'MADGap':>9}")
    for tag, d, acc, m, gap in rows:
        print(f"{tag:10s} {d:>5} {acc:>9.3f} {m:>8.4f} {gap:>+9.4f}")
    print("=" * 62)

    plain = [r for r in rows if r[0] == "GCN"]
    best = max(plain, key=lambda r: r[2])
    deepest = plain[-1]
    print(f"\nPlain GCN peaks at depth {best[1]} ({best[2]:.3f}) and drops to "
          f"{deepest[2]:.3f} at depth {deepest[1]}.")
    print(f"MAD falls {plain[0][3]:.3f} -> {deepest[3]:.4f}: the embeddings are "
          f"collapsing onto one vector.")
    ngnn_rows = [r for r in rows if r[0] == "GCN+NGNN"]
    deep_gain = ngnn_rows[-1][2] - deepest[2]
    print(f"At depth {deepest[1]}, adding the NGNN block changes accuracy by "
          f"{deep_gain:+.3f}.")

    # ---------------- figure --------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for tag, style in [("GCN", "o-"), ("GCN+NGNN", "s--")]:
        sub = [r for r in rows if r[0] == tag]
        d = [r[1] for r in sub]
        axes[0].plot(d, [r[2] for r in sub], style, label=tag, lw=2)
        axes[1].plot(d, [r[3] for r in sub], style, label=tag, lw=2)
        axes[2].plot(d, [r[4] for r in sub], style, label=tag, lw=2)
    axes[0].set_title("Accuracy vs depth\n(deeper is NOT better)")
    axes[0].set_ylabel("test accuracy")
    axes[1].set_title("MAD vs depth\n(embedding diversity collapses)")
    axes[1].set_ylabel("MAD")
    axes[2].set_title("MADGap vs depth\n(near vs far become alike)")
    axes[2].set_ylabel("MADGap")
    for ax in axes:
        ax.set_xlabel("number of message-passing layers")
        ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout()
    out = os.path.join(RESULTS, "03_oversmoothing.png")
    plt.savefig(out, dpi=130)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
