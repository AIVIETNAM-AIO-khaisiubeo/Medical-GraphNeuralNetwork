"""
Training loops, kept in one place so the experiment scripts stay short.

Node classification (Deck 1, p.67) is *transductive*: the whole graph —
including unlabelled nodes — goes through the network every epoch, but the
loss is computed on the training nodes only. That is what `train_mask` does.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from .graph import collate_graphs
from .metrics import accuracy, mae, rmse


def train_node_classifier(model, x, edge_index, extra, y,
                          train_mask, val_mask=None, test_mask=None,
                          epochs: int = 200, lr: float = 0.01,
                          weight_decay: float = 5e-4, log_every: int = 0):
    """Full-batch training for a node-level task.

    `extra` is whatever the model's third argument needs: GCN edge weights,
    R-GCN edge types, or None for MLP / GAT.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"loss": [], "train_acc": [], "val_acc": []}
    best_val, best_state, best_epoch = -1.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(x, edge_index, extra)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index, extra)
            tr = accuracy(logits, y, train_mask)
            va = accuracy(logits, y, val_mask) if val_mask is not None else tr
        history["loss"].append(float(loss.detach()))
        history["train_acc"].append(tr)
        history["val_acc"].append(va)
        if va > best_val:
            best_val, best_epoch = va, epoch
            best_state = copy.deepcopy(model.state_dict())
        if log_every and (epoch % log_every == 0 or epoch == 1):
            print(f"    epoch {epoch:3d} | loss {float(loss.detach()):.4f} | "
                  f"train {tr:.3f} | val {va:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, extra)
        test_acc = accuracy(logits, y, test_mask) if test_mask is not None else None
    return {"history": history, "best_val": best_val, "best_epoch": best_epoch,
            "test_acc": test_acc}


def _iterate_batches(items, indices, batch_size, shuffle=False, generator=None):
    idx = list(indices)
    if shuffle:
        perm = torch.randperm(len(idx), generator=generator).tolist()
        idx = [idx[i] for i in perm]
    for k in range(0, len(idx), batch_size):
        yield collate_graphs([items[i] for i in idx[k:k + batch_size]])


def train_graph_regressor(model, items, train_idx, val_idx, test_idx,
                          epochs: int = 60, batch_size: int = 32, lr: float = 1e-3,
                          weight_decay: float = 1e-5, log_every: int = 0, seed: int = 0):
    """Mini-batch training for the graph-level ESOL regression."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    history = {"train_rmse": [], "val_rmse": []}
    best_val, best_state = float("inf"), None

    def evaluate(indices):
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for b in _iterate_batches(items, indices, batch_size):
                preds.append(model(b.x, b.edge_index, b.edge_attr, b.batch, b.num_graphs))
                ys.append(b.y)
        p, t = torch.cat(preds), torch.cat(ys)
        return rmse(p, t), mae(p, t)

    for epoch in range(1, epochs + 1):
        model.train()
        for b in _iterate_batches(items, train_idx, batch_size, shuffle=True, generator=gen):
            opt.zero_grad()
            pred = model(b.x, b.edge_index, b.edge_attr, b.batch, b.num_graphs)
            loss = F.mse_loss(pred, b.y)
            loss.backward()
            opt.step()
        tr_rmse, _ = evaluate(train_idx)
        va_rmse, _ = evaluate(val_idx)
        history["train_rmse"].append(tr_rmse)
        history["val_rmse"].append(va_rmse)
        if va_rmse < best_val:
            best_val = va_rmse
            best_state = copy.deepcopy(model.state_dict())
        if log_every and (epoch % log_every == 0 or epoch == 1):
            print(f"    epoch {epoch:3d} | train RMSE {tr_rmse:.3f} | val RMSE {va_rmse:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    te_rmse, te_mae = evaluate(test_idx)
    return {"history": history, "best_val_rmse": best_val,
            "test_rmse": te_rmse, "test_mae": te_mae}
