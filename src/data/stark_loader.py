"""Thin wrapper around stark_qa: QA loading, fixed-seed subsets, SKB loading."""
from dataclasses import dataclass, field

import numpy as np

from src.utils import resolve


@dataclass
class Query:
    query_id: int
    text: str
    answer_ids: list
    meta: dict = field(default_factory=dict)


def load_qa(dataset):
    from stark_qa import load_qa as _load_qa
    return _load_qa(dataset)


def load_skb(dataset, data_root="data/stark"):
    from stark_qa import load_skb as _load_skb
    root = resolve(data_root) / dataset
    root.mkdir(parents=True, exist_ok=True)
    return _load_skb(dataset, download_processed=True, root=str(root))


def sample_subset(qa, split="test", size=100, seed=42):
    """Deterministic subset of a split. Same (split, size, seed) -> same queries."""
    indices = qa.get_idx_split()[split].tolist()
    rng = np.random.default_rng(seed)
    if size and size < len(indices):
        indices = sorted(rng.choice(indices, size=size, replace=False).tolist())
    queries = []
    for idx in indices:
        text, qid, answer_ids, meta = qa[idx]
        queries.append(Query(int(qid), text, [int(a) for a in answer_ids], meta or {}))
    return queries
