"""
Metrics, including the two over-smoothing diagnostics from Deck 1 (p.47-48).

MAD  (Mean Average Distance) measures how *different* node embeddings are
     from each other. If a GNN over-smooths, all embeddings converge and MAD
     falls towards 0.

MADGap = MAD(remote pairs) - MAD(neighbouring pairs).
     Intuition from the slide: messages between nearby nodes usually carry
     useful same-class signal, while messages from far-away nodes are mostly
     noise. A healthy model keeps distant nodes far apart (high MAD_remote)
     while pulling true neighbours together (low MAD_neighbour), so MADGap is
     large and positive. Over-smoothing drives it to 0.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    pred = logits.argmax(dim=1)
    if mask is not None:
        pred, y = pred[mask], y[mask]
    return float((pred == y).float().mean())


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(F.mse_loss(pred, target)))


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(F.l1_loss(pred, target))


def _cosine_distance_matrix(h: torch.Tensor) -> torch.Tensor:
    """D[i,j] = 1 - cos(h_i, h_j), in [0, 2]."""
    hn = F.normalize(h, p=2, dim=1)
    return (1.0 - hn @ hn.t()).clamp(min=0.0)


def mad(h: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Mean Average Distance over the node pairs selected by `mask`.

    Follows Chen et al. (2020): average the distances within each row over
    the *selected* entries only, then average over rows that have any.
    """
    D = _cosine_distance_matrix(h)
    n = h.shape[0]
    if mask is None:
        mask = ~torch.eye(n, dtype=torch.bool, device=h.device)
    D = D * mask.float()
    per_row_count = mask.float().sum(dim=1)
    per_row_mean = D.sum(dim=1) / per_row_count.clamp(min=1)
    keep = per_row_count > 0
    return float(per_row_mean[keep].mean()) if bool(keep.any()) else 0.0


def khop_neighbour_mask(edge_index: torch.Tensor, num_nodes: int, hops: int = 1) -> torch.Tensor:
    """Boolean [N, N] mask: True where j is reachable from i within `hops`.

    Built by repeatedly multiplying a dense boolean adjacency. Fine for Cora
    (2708 nodes); this is a diagnostic, not part of training.
    """
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=edge_index.device)
    A[edge_index[0], edge_index[1]] = True
    A = A | A.t()
    reach = A.clone()
    frontier = A.clone()
    for _ in range(hops - 1):
        frontier = (frontier.float() @ A.float()) > 0
        reach = reach | frontier
    reach.fill_diagonal_(False)
    return reach


def mad_gap(h: torch.Tensor, edge_index: torch.Tensor, num_nodes: int,
            near_hops: int = 1, remote_hops: int = 3) -> float:
    """MADGap = MAD(remote) - MAD(neighbour)  (Deck 1, p.48)."""
    near = khop_neighbour_mask(edge_index, num_nodes, hops=near_hops)
    within_remote = khop_neighbour_mask(edge_index, num_nodes, hops=remote_hops)
    remote = ~within_remote
    remote.fill_diagonal_(False)
    return mad(h, remote) - mad(h, near)
