"""
ESOL (Delaney) water-solubility dataset — the graph-LEVEL task.

Deck 2, p.64: "ESOL is a small dataset consisting of water solubility data
for 1128 compounds." One target value per *molecule*, so this is graph-level
regression: predict log solubility (mol/L) from the molecular graph.

Featurisation produces both node AND edge features, because Deck 2's central
theme is that edge information matters (p.4-9, p.48-50): a bond's type
(single/double/aromatic) changes what the message between two atoms should be.

NOTE ON LIBRARIES
-----------------
RDKit is used *only* to read SMILES strings into atoms and bonds — that is
chemistry parsing, not graph learning. Every graph operation, layer and
training loop in this repo is written from scratch in PyTorch. No
PyTorch Geometric / DGL / NetworkX anywhere.
"""
from __future__ import annotations

import csv
import os

import torch

ATOMS = ["C", "N", "O", "S", "Cl", "F", "Br", "I", "P", "other"]
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

NODE_DIM = len(ATOMS) + 5     # symbol one-hot + degree, charge, aromatic, numH, in-ring
EDGE_DIM = len(BOND_TYPES) + 2  # bond type one-hot + conjugated, in-ring


def _onehot(value, choices):
    vec = [0.0] * len(choices)
    vec[choices.index(value) if value in choices else len(choices) - 1] = 1.0
    return vec


def _atom_features(atom) -> list[float]:
    return _onehot(atom.GetSymbol(), ATOMS) + [
        atom.GetDegree() / 4.0,
        float(atom.GetFormalCharge()),
        float(atom.GetIsAromatic()),
        atom.GetTotalNumHs() / 4.0,
        float(atom.IsInRing()),
    ]


def _bond_features(bond) -> list[float]:
    return _onehot(str(bond.GetBondType()), BOND_TYPES) + [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def smiles_to_graph(smiles: str):
    """SMILES -> (x [N,NODE_DIM], edge_index [2,E], edge_attr [E,EDGE_DIM]).

    Bonds are undirected, so each one becomes two directed edges carrying the
    same edge feature vector.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    src, dst, attr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        src += [i, j]
        dst += [j, i]
        attr += [bf, bf]
    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(attr, dtype=torch.float)
    else:                                   # e.g. a single-atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_DIM), dtype=torch.float)
    return x, edge_index, edge_attr


def load_esol(root: str = "data/esol"):
    """Return a list of (x, edge_index, edge_attr, y) tuples."""
    path = os.path.join(root, "delaney-processed.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first.")

    target_col = "measured log solubility in mols per litre"
    items, skipped = [], 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            g = smiles_to_graph(row["smiles"])
            if g is None:
                skipped += 1
                continue
            y = torch.tensor(float(row[target_col]), dtype=torch.float)
            items.append((g[0], g[1], g[2], y))
    if skipped:
        print(f"  (skipped {skipped} unparseable SMILES)")
    return items


def split_indices(n: int, seed: int = 0, fracs=(0.8, 0.1, 0.1)):
    """Random train/val/test split over molecule indices."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_tr = int(fracs[0] * n)
    n_va = int(fracs[1] * n)
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
