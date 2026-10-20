"""Vector similarity search over STaRK's free precomputed ada-002 embeddings.

Unlike stark_qa.models.VSS, this never computes missing embeddings (which would
silently call the paid OpenAI API). A query without a precomputed embedding is
an error.
"""
from pathlib import Path

import torch

from src.utils import resolve


class PrecomputedVSS:
    def __init__(self, emb_dir, chunk_size=200_000):
        emb_dir = resolve(emb_dir)
        self.chunk_size = chunk_size
        self.query_embs = self._load_dict(emb_dir / "query" / "query_emb_dict.pt")
        self.candidate_ids, self.candidate_matrix = self._load_candidates(emb_dir / "doc")

    @staticmethod
    def _load_dict(path):
        return {int(k): v.view(-1).float() for k, v in torch.load(path).items()}

    def _load_candidates(self, doc_dir):
        """Stack candidate embeddings into one fp16 matrix, cached after the first load."""
        cache = Path(doc_dir) / "candidate_matrix_fp16.pt"
        if cache.exists():
            data = torch.load(cache)
            return data["ids"], data["matrix"]
        emb_dict = torch.load(Path(doc_dir) / "candidate_emb_dict.pt")
        ids = sorted(int(k) for k in emb_dict)
        matrix = torch.stack([emb_dict[i].view(-1) for i in ids]).half()
        del emb_dict
        torch.save({"ids": ids, "matrix": matrix}, cache)
        return ids, matrix

    def query_embedding(self, query_id):
        if query_id not in self.query_embs:
            raise KeyError(f"No precomputed embedding for query {query_id}; refusing to call a paid API.")
        return self.query_embs[query_id]

    def search(self, query_id, top_k=100):
        """Return (candidate_ids, scores) of the top_k candidates by dot product."""
        q = self.query_embedding(query_id)
        scores = torch.cat([
            self.candidate_matrix[s:s + self.chunk_size].float() @ q
            for s in range(0, len(self.candidate_ids), self.chunk_size)
        ])
        top = torch.topk(scores, top_k)
        return [self.candidate_ids[i] for i in top.indices.tolist()], top.values.tolist()
