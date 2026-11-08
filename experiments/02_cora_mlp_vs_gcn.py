"""
Experiment 2 — Cora node classification: MLP vs GCN.  (The main result.)

Slide: Deck 1, p.59-72.

Three models, identical features, identical hidden size (16), identical
training budget:

  1. MLP            1433 -> 16 -> 7.  Ignores the graph entirely.
  2. GCN (slide)    1433 -> 16 -> 16 -> 7, i.e. "2 message-passing layers and
                    one linear output layer" exactly as p.71 describes.
  3. GCN (canonical) 1433 -> 16 -> 7, where the second message-passing layer
                    outputs the class scores directly (Kipf & Welling 2017).

Comparing 1 against 2/3 measures what the citation graph is worth.
Comparing 2 against 3 shows how much a seemingly harmless architectural
choice costs: pushing 1433 features through two 16-dim bottlenecks before
classifying throws away a lot.

Also reproduces p.63 ("visualise the node embeddings of the untrained GCN"):
t-SNE of the hidden layer before vs after training.

Run:  python experiments/02_cora_mlp_vs_gcn.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn import GCN, MLP, gcn_normalisation
from gnn.data import load_cora
from gnn.train import train_node_classifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

HIDDEN = 16
EPOCHS = 200


def tsne(h: torch.Tensor):
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, init="pca", perplexity=30,
                random_state=0).fit_transform(h.detach().numpy())


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)

    g = load_cora(root=os.path.join(ROOT, "data", "cora"))
    print(g)
    print(f"  train/val/test nodes: {int(g.train_mask.sum())} / "
          f"{int(g.val_mask.sum())} / {int(g.test_mask.sum())}")
    print(f"  only {int(g.train_mask.sum()) / g.num_nodes:.1%} of nodes are labelled\n")

    # Â = D^-1/2 (A+I) D^-1/2 as per-edge weights — computed once, reused
    edge_index, edge_weight = gcn_normalisation(g.edge_index, g.num_nodes)
    n_classes = int(g.y.max()) + 1
    n_feat = g.x.shape[1]
    common = dict(epochs=EPOCHS, lr=0.01, weight_decay=5e-4)

    # ---------------- 1. MLP: graph-blind baseline ----------------------
    print("[1/3] MLP — ignores the graph")
    torch.manual_seed(0)
    mlp = MLP(n_feat, HIDDEN, n_classes)
    res_mlp = train_node_classifier(mlp, g.x, None, None, g.y, g.train_mask,
                                    g.val_mask, g.test_mask, **common)
    print(f"      test accuracy {res_mlp['test_acc']:.3f}\n")

    # ---------------- 2. GCN as described in the slide ------------------
    print("[2/3] GCN — slide version (2 MP layers + linear output)")
    torch.manual_seed(0)
    gcn_slide = GCN(n_feat, HIDDEN, n_classes, num_layers=2, head="linear")
    res_slide = train_node_classifier(gcn_slide, g.x, edge_index, edge_weight, g.y,
                                      g.train_mask, g.val_mask, g.test_mask, **common)
    print(f"      test accuracy {res_slide['test_acc']:.3f}\n")

    # ---------------- 3. canonical GCN ----------------------------------
    print("[3/3] GCN — canonical (2 MP layers, 2nd emits class scores)")
    torch.manual_seed(0)
    gcn = GCN(n_feat, HIDDEN, n_classes, num_layers=2, head="direct")

    gcn.eval()
    with torch.no_grad():       # embeddings BEFORE training (Deck 1 p.63)
        _, emb_before = gcn(g.x, edge_index, edge_weight, return_embedding=True)

    res_gcn = train_node_classifier(gcn, g.x, edge_index, edge_weight, g.y,
                                    g.train_mask, g.val_mask, g.test_mask,
                                    log_every=50, **common)
    print(f"      test accuracy {res_gcn['test_acc']:.3f}\n")

    gcn.eval()
    with torch.no_grad():
        _, emb_after = gcn(g.x, edge_index, edge_weight, return_embedding=True)

    gap = res_gcn["test_acc"] - res_mlp["test_acc"]
    print("=" * 64)
    print(f"  MLP            (no graph)        : {res_mlp['test_acc']:.3f}")
    print(f"  GCN slide      (2 MP + linear)   : {res_slide['test_acc']:.3f}")
    print(f"  GCN canonical  (2 MP, direct)    : {res_gcn['test_acc']:.3f}")
    print(f"\n  value of the citation graph      : +{gap:.3f} "
          f"({gap / max(res_mlp['test_acc'], 1e-9):+.0%} relative)")
    print(f"  cost of the extra 16-dim bottleneck: "
          f"{res_slide['test_acc'] - res_gcn['test_acc']:+.3f}")
    print("=" * 64)

    # ---------------- figures -------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    axes[0].plot(res_mlp["history"]["val_acc"], label=f"MLP ({res_mlp['test_acc']:.2f})", lw=2)
    axes[0].plot(res_slide["history"]["val_acc"],
                 label=f"GCN slide ({res_slide['test_acc']:.2f})", lw=2)
    axes[0].plot(res_gcn["history"]["val_acc"],
                 label=f"GCN canonical ({res_gcn['test_acc']:.2f})", lw=2)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("validation accuracy")
    axes[0].set_title("Cora node classification"); axes[0].legend(); axes[0].grid(alpha=.3)

    print("\ncomputing t-SNE of GCN hidden embeddings ...")
    for ax, emb, title in [(axes[1], emb_before, "GCN embeddings BEFORE training"),
                           (axes[2], emb_after, "GCN embeddings AFTER training")]:
        z = tsne(emb)
        ax.scatter(z[:, 0], z[:, 1], c=g.y.numpy(), cmap="tab10", s=5, alpha=.8)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    out = os.path.join(RESULTS, "02_cora_mlp_vs_gcn.png")
    plt.savefig(out, dpi=130)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
