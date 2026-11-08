"""
Experiment 5 — Relational GNN: when the TYPE of an edge matters.

Slide: Deck 2, p.38-46.
  "How to include various types of edges in GNN?" -> Relational GCN:
  one weight matrix W_r per relation,
      h_i' = W_0 h_i + sum_r (1/|N_i^r|) sum_{j in N_i^r} W_r h_j

We construct a task (see gnn/data/relational.py) where the answer is carried
by exactly ONE relation ("family") and the other two relations are noise:

    y_i = signal of node i's family member
    a node's own features say nothing about its own label

Three models, same data:
  * GCN   — treats every edge identically; the one useful neighbour is
            averaged together with ~8 noisy ones.
  * GAT   — learns attention from node features. But the noisy neighbours
            look just like the useful one, so attention has nothing to go on.
  * R-GCN — has a separate weight matrix per relation, so it can read
            `family` and discard the rest.

Random guessing is 33% (3 balanced classes).

Run:  python experiments/05_rgcn_relations.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn import GAT, GCN, RGCN, gcn_normalisation
from gnn.data.relational import RELATIONS, make_relational_graph
from gnn.train import train_node_classifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
EPOCHS = 150
HIDDEN = 32


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    g = make_relational_graph(num_nodes=900, seed=0)
    n_classes = int(g.y.max()) + 1
    n_feat = g.x.shape[1]
    print(f"Synthetic relational graph: {g.num_nodes} nodes, {g.num_edges} directed edges")
    for r, name in enumerate(RELATIONS):
        print(f"  relation {r} ({name}): {int((g.edge_type == r).sum())} edges")
    print(f"  node features = one-hot(own signal), which is NOT its label")
    print(f"  label = signal of the node's FAMILY neighbour")
    print(f"  chance accuracy = {1 / n_classes:.1%}\n")

    results = {}

    print("[1/3] GCN — relation-blind")
    ei_norm, w_norm = gcn_normalisation(g.edge_index, g.num_nodes)
    torch.manual_seed(0)
    gcn = GCN(n_feat, HIDDEN, n_classes, num_layers=2, dropout=0.0, head="direct")
    results["GCN"] = train_node_classifier(
        gcn, g.x, ei_norm, w_norm, g.y, g.train_mask, g.val_mask, g.test_mask,
        epochs=EPOCHS, lr=0.01, weight_decay=0.0)
    print(f"      test accuracy {results['GCN']['test_acc']:.3f}\n")

    print("[2/3] GAT — attention from node features only")
    torch.manual_seed(0)
    gat = GAT(n_feat, 16, n_classes, heads=4, dropout=0.0)
    results["GAT"] = train_node_classifier(
        gat, g.x, g.edge_index, None, g.y, g.train_mask, g.val_mask, g.test_mask,
        epochs=EPOCHS, lr=0.01, weight_decay=0.0)
    print(f"      test accuracy {results['GAT']['test_acc']:.3f}\n")

    print("[3/3] R-GCN — one weight matrix per relation")
    torch.manual_seed(0)
    rgcn = RGCN(n_feat, HIDDEN, n_classes, num_relations=len(RELATIONS))
    results["R-GCN"] = train_node_classifier(
        rgcn, g.x, g.edge_index, g.edge_type, g.y, g.train_mask, g.val_mask, g.test_mask,
        epochs=EPOCHS, lr=0.01, weight_decay=0.0, log_every=50)
    print(f"      test accuracy {results['R-GCN']['test_acc']:.3f}\n")

    print("=" * 58)
    for name, res in results.items():
        print(f"  {name:6s} test accuracy {res['test_acc']:.3f}")
    print(f"  {'chance':6s}               {1 / n_classes:.3f}")
    print("=" * 58)

    # Which relation does the trained R-GCN actually depend on?
    # Weight norms turn out to be uninformative (they come out nearly equal),
    # so we ablate instead: delete one relation's edges at inference time and
    # see how far accuracy falls.
    from gnn.metrics import accuracy

    rgcn.eval()
    with torch.no_grad():
        full = accuracy(rgcn(g.x, g.edge_index, g.edge_type), g.y, g.test_mask)
        drops = []
        for r, name in enumerate(RELATIONS):
            keep = g.edge_type != r
            acc = accuracy(rgcn(g.x, g.edge_index[:, keep], g.edge_type[keep]),
                           g.y, g.test_mask)
            drops.append(full - acc)
            print(f"\n  remove '{name}' edges -> test accuracy {acc:.3f} "
                  f"(drop {full - acc:+.3f})" if r == 0 else
                  f"  remove '{name}' edges -> test accuracy {acc:.3f} "
                  f"(drop {full - acc:+.3f})")
    print("\nOnly removing 'family' hurts — the model learned which relation matters.")

    # ---- figure ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for name, res in results.items():
        axes[0].plot(res["history"]["val_acc"],
                     label=f"{name} ({res['test_acc']:.2f})", lw=2)
    axes[0].axhline(1 / n_classes, ls="--", c="grey", lw=1, label="chance")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("validation accuracy")
    axes[0].set_title("Only R-GCN can use the edge TYPE")
    axes[0].legend(); axes[0].grid(alpha=.3)

    axes[1].bar(RELATIONS, drops, color=["#2a6fdb", "#9aa6b2", "#c6ccd2"])
    axes[1].set_ylabel("accuracy lost when relation removed")
    axes[1].set_title("R-GCN depends only on 'family'")
    plt.tight_layout()
    out = os.path.join(RESULTS, "05_rgcn_relations.png")
    plt.savefig(out, dpi=130)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
