"""Retrieval metrics used throughout the project (same definitions as STaRK)."""


def hit_at_k(ranked_ids, answer_ids, k=1):
    answers = set(answer_ids)
    return float(any(i in answers for i in ranked_ids[:k]))


def recall_at_k(ranked_ids, answer_ids, k=20):
    answers = set(answer_ids)
    if not answers:
        return 0.0
    return len(answers & set(ranked_ids[:k])) / len(answers)


def mrr(ranked_ids, answer_ids):
    answers = set(answer_ids)
    for rank, i in enumerate(ranked_ids, start=1):
        if i in answers:
            return 1.0 / rank
    return 0.0


def evaluate(ranked_ids, answer_ids):
    return {
        "hit@1": hit_at_k(ranked_ids, answer_ids, 1),
        "hit@5": hit_at_k(ranked_ids, answer_ids, 5),
        "recall@20": recall_at_k(ranked_ids, answer_ids, 20),
        "mrr": mrr(ranked_ids, answer_ids),
    }


def aggregate(per_query):
    """Average a list of metric dicts."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: sum(m[k] for m in per_query) / len(per_query) for k in keys}
