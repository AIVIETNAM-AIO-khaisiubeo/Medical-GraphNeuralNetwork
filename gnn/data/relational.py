"""
A small synthetic relational graph — the "friend / family / colleague" setting
from Deck 2, p.38-44.

The point of R-GCN is that **not all edges mean the same thing**. To show that
in a measurable way we build a task where exactly one relation carries the
answer and the others are noise:

  * every node i has a random signal  s_i in {0, 1, 2}; its feature vector is
    one-hot(s_i), so a node's own features say NOTHING about its own label;
  * the `family` relation is a perfect matching (one family edge per node) and
    the label is  y_i = s_{family(i)}  — the signal of your family member;
  * `friend` and `colleague` relations add random edges, i.e. pure noise.

A relation-blind model averages over all ~9 neighbours and drowns the single
informative one. An R-GCN has a separate W_r per relation and can simply
learn to read `family` and ignore the rest.

Note that attention (GAT) does not fix this either: attention is computed
from node *features*, and here the noisy neighbours look exactly like the
informative one. You need the edge *type*.
"""
from __future__ import annotations

import torch

from ..graph import Graph

RELATIONS = ["family", "friend", "colleague"]


def make_relational_graph(num_nodes: int = 900, num_classes: int = 3,
                          friends_per_node: int = 4, colleagues_per_node: int = 4,
                          seed: int = 0) -> Graph:
    """Build the synthetic graph described above."""
    g = torch.Generator().manual_seed(seed)
    signal = torch.randint(0, num_classes, (num_nodes,), generator=g)
    x = torch.zeros(num_nodes, num_classes)
    x[torch.arange(num_nodes), signal] = 1.0

    # --- family: a perfect matching (num_nodes assumed even) -------------
    perm = torch.randperm(num_nodes, generator=g)
    a, b = perm[0::2], perm[1::2]
    partner = torch.empty(num_nodes, dtype=torch.long)
    partner[a], partner[b] = b, a
    y = signal[partner]                      # the label: your family's signal

    src = [a, b]
    dst = [b, a]
    etype = [torch.zeros(a.numel(), dtype=torch.long),
             torch.zeros(b.numel(), dtype=torch.long)]

    # --- friend (type 1) and colleague (type 2): random noise edges ------
    for rel_id, k in ((1, friends_per_node), (2, colleagues_per_node)):
        s = torch.arange(num_nodes).repeat_interleave(k)
        d = torch.randint(0, num_nodes, (num_nodes * k,), generator=g)
        keep = s != d
        s, d = s[keep], d[keep]
        src += [s, d]                        # keep the graph undirected
        dst += [d, s]
        etype += [torch.full((s.numel(),), rel_id, dtype=torch.long),
                  torch.full((d.numel(),), rel_id, dtype=torch.long)]

    edge_index = torch.stack([torch.cat(src), torch.cat(dst)])
    edge_type = torch.cat(etype)

    n_tr, n_va = int(0.6 * num_nodes), int(0.2 * num_nodes)
    order = torch.randperm(num_nodes, generator=g)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[order[:n_tr]] = True
    val_mask[order[n_tr:n_tr + n_va]] = True
    test_mask[order[n_tr + n_va:]] = True

    return Graph(x=x, edge_index=edge_index, y=y, edge_type=edge_type,
                 train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                 meta={"name": "SyntheticRelational", "relations": RELATIONS})
