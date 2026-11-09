# Graph Neural Networks for Medicine — From Scratch, Then With Libraries

A hands-on course that builds a **Graph Neural Network for a real medical task twice**: once implementing every operation by hand in plain PyTorch, then rebuilding the same thing with **PyTorch Geometric**, the library used in practice.

**The task:** predict whether a drug molecule crosses the **blood–brain barrier (BBB)** — a central question in CNS drug discovery — from its chemical structure alone. Dataset: **BBBP** (MoleculeNet, ~2,000 labelled molecules). This is a *graph classification* problem: one label per molecule.

Everything here is designed for **Google Colab**. The notebooks come with their outputs already rendered, so you can read them straight through, then run and modify them yourself.

---

## The two tracks

| Track | Notebooks | What you get |
|-------|-----------|--------------|
| **From scratch** | 01 → 02 → 03 | Deep understanding. Message passing, normalization, batching, pooling and the training loop written with nothing but tensors. |
| **With libraries** | 04 → 05 | Practice. The same model in ~20 lines of PyTorch Geometric, then attention, expressiveness, honest evaluation and explainability. |

Notebook 04 maps **every** hand-written piece from Track 1 onto its one-line library equivalent, so the library never feels like magic.

---

## Learning path

| # | Notebook | Core ideas |
|---|----------|-----------|
| **01** | `01_graphs_and_molecules.ipynb` | What a graph is · molecules as graphs · atom features · `edge_index` (COO) · the BBBP task & class imbalance |
| **02** | `02_message_passing_from_scratch.ipynb` | Why raw `A·X` fails · self-loops · symmetric normalization `Â = D⁻½(A+I)D⁻½` · a GCN layer `σ(ÂHW)` by hand · depth = hops · over-smoothing |
| **03** | `03_full_gnn_from_scratch.ipynb` | Readout/pooling · block-diagonal mini-batching · the training loop · **ROC-AUC vs accuracy** · first real BBBP predictions |
| **04** | `04_pytorch_geometric.ipynb` | `MoleculeNet` · `Data` objects · `DataLoader` · `GCNConv` · `global_mean_pool` · the scratch→PyG mapping table |
| **05** | `05_advanced_gnns.ipynb` | **GAT** (attention) · **GIN** (expressiveness) · **scaffold splitting** (honest eval) · **GNNExplainer** (which atoms mattered) · where to go next |

Work through them in order — each builds on the last.

---

## How to run in Google Colab

Each notebook is self-contained (it installs what it needs and downloads the data).

**Option A — upload:**
1. Go to [colab.research.google.com](https://colab.research.google.com) → **File ▸ Upload notebook**.
2. Upload a notebook (start with `01_...`).
3. **Runtime ▸ Run all**. Notebooks 03–05 train faster on a GPU: **Runtime ▸ Change runtime type ▸ GPU** (free tier is fine).

**Option B — Google Drive:** drop the notebooks in Drive and open each with *Open with ▸ Google Colaboratory*.

The first cell of each notebook runs `pip install` for `rdkit` / `torch-geometric`; `torch` is already present in Colab.

### Running locally instead
```bash
pip install torch torch-geometric rdkit scikit-learn pandas matplotlib jupyter
jupyter notebook
```

---

## What the models score (BBBP, ROC-AUC)

Indicative numbers from the notebooks (they vary a little run-to-run; 0.5 = random):

| Setup | ROC-AUC | Notebook |
|-------|---------|----------|
| From-scratch 3-layer GCN, random split | ~0.80 | 03 |
| PyG GCN, random split | ~0.75 | 04 |
| GCN / GAT / GIN, random split | ~0.72–0.78 | 05 |
| GCN, **scaffold split** (novel chemotypes) | ~0.70 | 05 |

The drop under scaffold splitting is the point: it's the honest estimate of how well the model handles chemistry it has never seen.

---

## Concepts covered

- Graph representation: nodes, edges, adjacency, node features, COO `edge_index`
- Molecular featurization with RDKit (atoms → vectors, bonds → edges)
- Message passing; the GCN propagation rule and why normalization matters
- Receptive field vs. depth; over-smoothing
- Graph-level readout (mean/sum/max pooling), permutation invariance
- Mini-batching graphs (block-diagonal / the `batch` vector)
- Training: BCE-with-logits, Adam, early-stopping on validation
- Evaluation under class imbalance (ROC-AUC) and distribution shift (scaffold split)
- Architectures: GCN, GAT (attention), GIN (WL-expressiveness)
- Explainability with GNNExplainer
- The practical ecosystem: PyTorch Geometric, DGL, DeepChem, OGB

---

## Requirements

Python 3.9+. Key packages: `torch`, `torch-geometric`, `rdkit`, `scikit-learn`, `pandas`, `matplotlib`. All installed inside the notebooks for Colab; see the local block above otherwise.

## Data & credits

- **BBBP** dataset from **MoleculeNet** (Wu et al., 2018). Loaded via PyTorch Geometric's `MoleculeNet` in NB04–05, and from a public CSV mirror in NB01/03.
- Methods referenced: GCN (Kipf & Welling, 2017), GAT (Veličković et al., 2018), GIN (Xu et al., 2019), GNNExplainer (Ying et al., 2019).

Built as a learning resource — read, run, break, and extend it.
