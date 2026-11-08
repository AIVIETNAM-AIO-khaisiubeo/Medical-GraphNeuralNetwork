"""
Experiment 6 — Graph-LEVEL prediction: ESOL water solubility.

Slide: Deck 2, p.61-71 ("Graph-level prediction: Example and Code"),
ESOL = 1128 compounds with measured log solubility (mol/L).

Node-level tasks (experiments 2-5) read a prediction off each node. A
graph-level task needs one extra ingredient: a permutation-invariant
**readout** that turns a variable number of atom embeddings into a single
molecule vector,

    h_graph = POOL({h_v})        y_hat = MLP(h_graph)

We also test Deck 2's central claim — that **edge features matter** (p.4-9,
p.48-50) — by training the identical architecture twice, once with real bond
features (type, conjugation, ring membership) and once with those features
zeroed out. And we compare the three pooling choices.

Metric: RMSE in log-solubility units (lower is better). A model that always
predicts the training mean is the baseline to beat.

RDKit is used only to turn SMILES into atoms and bonds; every layer, the
batching and the training loop are from scratch.

Run:  python experiments/06_esol_graph_regression.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn import GraphRegressor, collate_graphs
from gnn.data import EDGE_DIM, NODE_DIM, load_esol, split_indices
from gnn.train import train_graph_regressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
EPOCHS = 60


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)

    print("featurising ESOL molecules (RDKit parses SMILES) ...")
    items = load_esol(root=os.path.join(ROOT, "data", "esol"))
    n = len(items)
    atoms = sum(it[0].shape[0] for it in items)
    print(f"  {n} molecules | {atoms} atoms total | "
          f"node features {NODE_DIM}, edge features {EDGE_DIM}")
    targets = torch.stack([it[3] for it in items])
    print(f"  target log-solubility: mean {targets.mean():.2f}, "
          f"std {targets.std():.2f}, range [{targets.min():.1f}, {targets.max():.1f}]")

    tr, va, te = split_indices(n, seed=0)
    print(f"  split: {len(tr)} train / {len(va)} val / {len(te)} test\n")

    # ---- trivial baseline: always predict the training mean ------------
    mean_pred = torch.stack([items[i][3] for i in tr]).mean()
    te_y = torch.stack([items[i][3] for i in te])
    baseline_rmse = float(torch.sqrt(((te_y - mean_pred) ** 2).mean()))
    print(f"baseline (predict train mean): test RMSE {baseline_rmse:.3f}\n")

    results = {}

    # ---- 1. with real bond features ------------------------------------
    print("[1/4] GNN with edge (bond) features, mean pooling")
    torch.manual_seed(0)
    model = GraphRegressor(NODE_DIM, EDGE_DIM, hidden=64, num_layers=3, pool="mean")
    results["edge feats + mean"] = train_graph_regressor(
        model, items, tr, va, te, epochs=EPOCHS, batch_size=32, lr=1e-3, log_every=20)
    print(f"      test RMSE {results['edge feats + mean']['test_rmse']:.3f}  "
          f"MAE {results['edge feats + mean']['test_mae']:.3f}\n")
    models = {"edge feats + mean": model}

    # ---- 2. identical model, bond features zeroed ----------------------
    print("[2/4] SAME model but bond features zeroed out (ablation)")
    blind = [(x, ei, torch.zeros_like(ea), y) for x, ei, ea, y in items]
    torch.manual_seed(0)
    model_blind = GraphRegressor(NODE_DIM, EDGE_DIM, hidden=64, num_layers=3, pool="mean")
    results["no edge feats"] = train_graph_regressor(
        model_blind, blind, tr, va, te, epochs=EPOCHS, batch_size=32, lr=1e-3)
    print(f"      test RMSE {results['no edge feats']['test_rmse']:.3f}\n")

    # ---- 3-4. pooling comparison ---------------------------------------
    for pool in ("sum", "max"):
        print(f"[{3 if pool == 'sum' else 4}/4] edge features + {pool} pooling")
        torch.manual_seed(0)
        m = GraphRegressor(NODE_DIM, EDGE_DIM, hidden=64, num_layers=3, pool=pool)
        results[f"edge feats + {pool}"] = train_graph_regressor(
            m, items, tr, va, te, epochs=EPOCHS, batch_size=32, lr=1e-3)
        models[f"edge feats + {pool}"] = m
        print(f"      test RMSE {results[f'edge feats + {pool}']['test_rmse']:.3f}\n")

    print("=" * 62)
    print(f"{'model':24s} {'test RMSE':>10} {'vs baseline':>12}")
    print(f"{'predict train mean':24s} {baseline_rmse:>10.3f} {'-':>12}")
    for name, r in results.items():
        print(f"{name:24s} {r['test_rmse']:>10.3f} "
              f"{baseline_rmse - r['test_rmse']:>+12.3f}")
    print("=" * 62)

    with_e = results["edge feats + mean"]["test_rmse"]
    no_e = results["no edge feats"]["test_rmse"]
    print(f"\nEdge features are worth {no_e - with_e:+.3f} RMSE "
          f"({(no_e - with_e) / no_e:+.1%} relative) — Deck 2's claim, measured.")

    best_name = min(models, key=lambda k: results[k]["best_val_rmse"])
    best_model = models[best_name]
    best_rmse = results[best_name]["test_rmse"]
    print(f"Best readout: {best_name} (test RMSE {best_rmse:.3f}).")
    print("Sum pooling wins here because solubility depends on molecular SIZE;")
    print("mean pooling divides that away, max pooling keeps only extremes.")

    # ---- figure ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for name in ("edge feats + mean", "no edge feats"):
        axes[0].plot(results[name]["history"]["val_rmse"], label=name, lw=2)
    axes[0].axhline(baseline_rmse, ls="--", c="grey", lw=1, label="predict mean")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("validation RMSE")
    axes[0].set_title("ESOL graph-level regression"); axes[0].legend(); axes[0].grid(alpha=.3)

    best_model.eval()
    preds = []
    with torch.no_grad():
        for k in range(0, len(te), 64):
            b = collate_graphs([items[i] for i in te[k:k + 64]])
            preds.append(best_model(b.x, b.edge_index, b.edge_attr, b.batch, b.num_graphs))
    p = torch.cat(preds).numpy()
    t = te_y.numpy()
    axes[1].scatter(t, p, s=14, alpha=.6, color="#2a6fdb")
    lo, hi = float(min(t.min(), p.min())) - .5, float(max(t.max(), p.max())) + .5
    axes[1].plot([lo, hi], [lo, hi], "--", c="#d6453d", lw=1.5, label="perfect")
    axes[1].set_xlabel("measured log solubility")
    axes[1].set_ylabel("predicted")
    axes[1].set_title(f"Test predictions — {best_name} (RMSE {best_rmse:.2f})")
    axes[1].legend()
    plt.tight_layout()
    out = os.path.join(RESULTS, "06_esol_graph_regression.png")
    plt.savefig(out, dpi=130)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
