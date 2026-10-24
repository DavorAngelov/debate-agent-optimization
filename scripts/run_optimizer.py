import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.actor import DEFAULT_INSTRUCTIONS, run_actor
from src.data.stark_loader import load_qa, load_skb
from src.eval.metrics import aggregate
from src.optimizers.comparator import improve
from src.utils import load_config, resolve, save_json, set_seed

METRICS = ("hit@1", "hit@5", "recall@20", "mrr")


def score(results):
    return aggregate([{k: r[k] for k in METRICS} for r in results])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline2_amazon.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])

    train = json.load(open(resolve(cfg["train_file"])))
    test = json.load(open(resolve(cfg["test_file"])))
    qa = load_qa(cfg["dataset"])
    queries = {int(qa[i][1]): qa[i][0] for i in range(len(qa))}
    skb = load_skb(cfg["dataset"], cfg["data_root"])
    run = lambda rows, ins: run_actor(rows, queries, skb, ins, cfg["model"], cfg["doc_chars"])

    best = DEFAULT_INSTRUCTIONS
    best_results = run(train, best)
    best_score = score(best_results)["mrr"]
    history = [{"round": 0, "instructions": best, "train_mrr": best_score, "kept": True}]
    print(f"round 0  train mrr {best_score:.3f}")

    optimizer_tokens = 0
    for rnd in range(1, cfg["rounds"] + 1):
        good = [r for r in best_results if r["hit@1"] == 1]
        bad = [r for r in best_results if r["hit@1"] == 0]
        good = rng.sample(good, min(cfg["examples"], len(good)))
        bad = rng.sample(bad, min(cfg["examples"], len(bad)))

        new, tokens = improve(best, good, bad, queries, skb, cfg["model"], cfg["doc_chars"])
        optimizer_tokens += tokens
        new_results = run(train, new)
        new_score = score(new_results)["mrr"]
        kept = new_score > best_score
        history.append({"round": rnd, "instructions": new, "train_mrr": new_score, "kept": kept})
        print(f"round {rnd}  train mrr {new_score:.3f}  {'kept' if kept else 'rejected'}")
        if kept:
            best, best_results, best_score = new, new_results, new_score

    test_results = run(test, best)
    metrics = score(test_results)
    metrics["optimizer_tokens"] = optimizer_tokens
    print("test", {k: round(v, 4) for k, v in metrics.items()})
    print("best instructions:", best)

    save_json(history, Path(cfg["output_dir"]) / "history.json")
    save_json(test_results, Path(cfg["output_dir"]) / "per_query.json")
    save_json({"config": cfg, "instructions": best, "metrics": metrics}, Path(cfg["output_dir"]) / "metrics.json")


if __name__ == "__main__":
    main()
