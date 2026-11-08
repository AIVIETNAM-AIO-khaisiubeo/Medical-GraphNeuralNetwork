"""
Download the two datasets used by this repo.

  Cora  (node classification, Deck 1)  -> data/cora/cora.content, cora.cites
  ESOL  (graph regression,   Deck 2)  -> data/esol/delaney-processed.csv

Both are plain text/CSV, parsed by our own code in `gnn/data/`.
Run:  python scripts/download_data.py
"""
from __future__ import annotations

import os
import urllib.request

FILES = [
    ("data/cora/cora.content",
     "https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora/cora.content"),
    ("data/cora/cora.cites",
     "https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora/cora.cites"),
    ("data/esol/delaney-processed.csv",
     "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"),
]


def main() -> None:
    for path, url in FILES:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"[skip] {path} already present")
            continue
        print(f"[get ] {url}")
        urllib.request.urlretrieve(url, path)
        print(f"       -> {path} ({os.path.getsize(path):,} bytes)")
    print("\nDone. Now try:  python experiments/02_cora_mlp_vs_gcn.py")


if __name__ == "__main__":
    main()
