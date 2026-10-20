"""Run a retrieval baseline on STaRK and log metrics to results/.

Usage:
    python scripts/run_baseline.py --config configs/baseline0_amazon.yaml
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from src.data.stark_loader import load_qa, sample_subset
from src.eval.metrics import aggregate, evaluate
from src.retrieval.vss import PrecomputedVSS
from src.utils import load_config, save_json, set_seed


def build_retriever(cfg):
    if cfg["method"] == "vss":
        return PrecomputedVSS(cfg["emb_dir"])
    raise ValueError(f"Unknown method: {cfg['method']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline0_amazon.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    qa = load_qa(cfg["dataset"])
    t0 = time.time()
    retriever = build_retriever(cfg)
    print(f"Retriever ready in {time.time() - t0:.0f}s")

    summary = {"config": cfg, "results": {}}
    for es in cfg["eval_sets"]:
        queries = sample_subset(qa, es["split"], es["size"], cfg["seed"])
        per_query = []
        for q in tqdm(queries, desc=es["name"]):
            ranked, _ = retriever.search(q.query_id, cfg["top_k"])
            per_query.append({"query_id": q.query_id, "top20": ranked[:20],
                              "answer_ids": q.answer_ids, **evaluate(ranked, q.answer_ids)})
        metrics = aggregate([{k: r[k] for k in ("hit@1", "hit@5", "recall@20", "mrr")} for r in per_query])
        summary["results"][es["name"]] = {"n": len(queries), **metrics}
        print(es["name"], {k: round(v, 4) for k, v in metrics.items()})
        save_json(per_query, Path(cfg["output_dir"]) / f"per_query_{es['name']}.json")

    save_json(summary, Path(cfg["output_dir"]) / "metrics.json")
    print(f"Saved to {cfg['output_dir']}/")


if __name__ == "__main__":
    main()
