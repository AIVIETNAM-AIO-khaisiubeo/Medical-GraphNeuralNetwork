"""
Cora citation dataset — parsed by hand from the original text files.

Deck 1, p.59-62. 2708 machine-learning papers, each described by a
1433-dimensional **bag-of-words** vector (word present / absent), connected
by 5429 citation links, labelled with one of 7 research topics.

Why node classification: we have ONE big graph (not many small ones), most
nodes are unlabelled, and we want a label per node — the semi-supervised
"transductive" setting described on p.59.

Files (plain TSV, no pickle):
  data/cora/cora.content : <paper_id> <1433 binary features> <class name>
  data/cora/cora.cites   : <cited_paper_id> <citing_paper_id>
"""
from __future__ import annotations

import os

import torch

from ..graph import Graph, to_undirected

CLASSES = [
    "Case_Based", "Genetic_Algorithms", "Neural_Networks",
    "Probabilistic_Methods", "Reinforcement_Learning", "Rule_Learning", "Theory",
]


def load_cora(root: str = "data/cora", row_normalise: bool = True,
              num_train_per_class: int = 20, num_val: int = 500,
              num_test: int = 1000, seed: int = 0) -> Graph:
    """Load Cora into a `Graph`, with the standard semi-supervised split.

    The split follows the usual Planetoid recipe: `num_train_per_class`
    labelled nodes per class (140 in total — only ~5% of the graph), then 500
    validation and 1000 test nodes. Training on so few labels is exactly why
    the graph structure matters.
    """
    content = os.path.join(root, "cora.content")
    cites = os.path.join(root, "cora.cites")
    if not (os.path.exists(content) and os.path.exists(cites)):
        raise FileNotFoundError(
            f"Cora files not found in {root!r}. Run `python scripts/download_data.py` first.")

    ids, feats, labels = [], [], []
    with open(content) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            ids.append(parts[0])
            feats.append([float(v) for v in parts[1:-1]])
            labels.append(CLASSES.index(parts[-1]))

    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    x = torch.tensor(feats, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    if row_normalise:
        # each paper's BoW vector is scaled to sum to 1, so long papers don't
        # simply produce larger activations than short ones
        x = x / x.sum(dim=1, keepdim=True).clamp(min=1.0)

    src, dst = [], []
    with open(cites) as f:
        for line in f:
            a, b = line.split()
            if a in id_to_idx and b in id_to_idx:
                src.append(id_to_idx[a])
                dst.append(id_to_idx[b])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index = to_undirected(edge_index)   # citations treated as undirected

    n = x.shape[0]
    g = torch.Generator().manual_seed(seed)
    train_mask = torch.zeros(n, dtype=torch.bool)
    for c in range(len(CLASSES)):
        idx = (y == c).nonzero(as_tuple=True)[0]
        pick = idx[torch.randperm(idx.numel(), generator=g)[:num_train_per_class]]
        train_mask[pick] = True

    remaining = (~train_mask).nonzero(as_tuple=True)[0]
    remaining = remaining[torch.randperm(remaining.numel(), generator=g)]
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    val_mask[remaining[:num_val]] = True
    test_mask[remaining[num_val:num_val + num_test]] = True

    return Graph(x=x, edge_index=edge_index, y=y,
                 train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                 meta={"classes": CLASSES, "name": "Cora"})
