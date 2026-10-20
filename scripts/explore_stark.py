"""Explore a STaRK dataset: split sizes, answer-set stats, example queries and
their ground-truth documents. Also freezes the fixed-seed query subset that all
baselines are evaluated on.

Usage:
    python scripts/explore_stark.py --config configs/explore_amazon.yaml
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.stark_loader import load_qa, load_skb, sample_subset
from src.utils import load_config, save_json, set_seed


def describe_lengths(values):
    a = np.asarray(values)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "min": int(a.min()), "max": int(a.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/explore_amazon.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    print(f"== Loading QA for '{cfg['dataset']}'")
    qa = load_qa(cfg["dataset"])
    splits = {k: len(v) for k, v in qa.get_idx_split().items()}
    print("Split sizes:", splits)

    subset = sample_subset(qa, cfg["split"], cfg["subset_size"], cfg["seed"])
    n_answers = [len(q.answer_ids) for q in subset]
    q_words = [len(q.text.split()) for q in subset]
    report = {
        "dataset": cfg["dataset"],
        "splits": splits,
        "subset": {"split": cfg["split"], "size": len(subset), "seed": cfg["seed"]},
        "answers_per_query": describe_lengths(n_answers),
        "query_words": describe_lengths(q_words),
    }
    print("Answers/query:", report["answers_per_query"])
    print("Query length (words):", report["query_words"])

    skb = None
    if cfg.get("load_skb"):
        print(f"\n== Loading SKB (first run downloads to {cfg['data_root']})")
        skb = load_skb(cfg["dataset"], cfg["data_root"])
        node_types = [skb.node_type_dict[t] for t in skb.node_types.tolist()]
        report["skb"] = {
            "num_nodes": skb.num_nodes(),
            "node_types": dict(Counter(node_types)),
            "relation_types": list(skb.rel_type_lst()),
            "candidate_type": skb.candidate_types if hasattr(skb, "candidate_types") else None,
        }
        print("SKB:", {k: v for k, v in report["skb"].items() if k != "relation_types"})
        print("Relations:", report["skb"]["relation_types"])

    print(f"\n== {cfg['num_examples']} example queries")
    for q in subset[: cfg["num_examples"]]:
        print(f"\n[q{q.query_id}] {q.text}")
        print(f"  answers ({len(q.answer_ids)}): {q.answer_ids[:10]}")
        if skb is not None:
            doc = skb.get_doc_info(q.answer_ids[0], add_rel=False)
            print("  first answer doc:", doc[:400].replace("\n", " "), "...")

    save_json(report, Path(cfg["output_dir"]) / "report.json")
    save_json([q.__dict__ for q in subset], Path(cfg["output_dir"]) / "subset_queries.json")
    print(f"\nSaved report and subset to {cfg['output_dir']}/")


if __name__ == "__main__":
    main()
