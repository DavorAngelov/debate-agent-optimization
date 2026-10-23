import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from src.agents.actor import DEFAULT_INSTRUCTIONS, rerank
from src.data.stark_loader import load_qa, load_skb
from src.eval.metrics import aggregate, evaluate
from src.utils import load_config, resolve, save_json, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline1_amazon.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    with open(resolve(cfg["candidates_file"])) as f:
        rows = json.load(f)
    qa = load_qa(cfg["dataset"])
    queries = {int(qa[i][1]): qa[i][0] for i in qa.get_idx_split()["test"].tolist()}
    skb = load_skb(cfg["dataset"], cfg["data_root"])

    per_query = []
    for row in tqdm(rows):
        candidates = [(pid, skb.get_doc_info(pid, add_rel=False)[:cfg["doc_chars"]].replace("\n", " "))
                      for pid in row["top20"]]
        ranked, answer, tokens = rerank(queries[row["query_id"]], candidates, DEFAULT_INSTRUCTIONS, cfg["model"])
        per_query.append({"query_id": row["query_id"], "ranked": ranked, "answer_ids": row["answer_ids"],
                          "llm_answer": answer, "tokens": tokens, **evaluate(ranked, row["answer_ids"])})

    metrics = aggregate([{k: r[k] for k in ("hit@1", "hit@5", "recall@20", "mrr")} for r in per_query])
    metrics["avg_tokens"] = sum(r["tokens"] for r in per_query) / len(per_query)
    print({k: round(v, 4) for k, v in metrics.items()})
    save_json(per_query, Path(cfg["output_dir"]) / "per_query.json")
    save_json({"config": cfg, "instructions": DEFAULT_INSTRUCTIONS, "metrics": metrics},
              Path(cfg["output_dir"]) / "metrics.json")


if __name__ == "__main__":
    main()
