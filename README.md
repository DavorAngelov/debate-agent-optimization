# Debate-Based Optimization of LLM Agents

LLM-based agentic systems

## Idea

[AvaTaR](https://arxiv.org/abs/2406.11200) improves an LLM agent (the **actor**) by having a second agent (the **comparator**) look at its good and bad answers and rewrite its instructions.

This project replaces the single comparator with a **debate**:
- a **critic** argues why the actor failed,
- a **defender** argues back,
- a **judge** turns the debate into new instructions for the actor.

**Research question:** does debate give better feedback for improving an agent than a single agent reflecting alone?

## Setup

- **Data:** [STaRK](https://stark.stanford.edu/) Amazon product search (about 1M products), evaluated on a fixed set of 100 test questions.
- **Models:** local Qwen2.5-7B through [Ollama](https://ollama.com/). Everything runs locally and for free.
- **Metrics:** Hit@1, Hit@5, Recall@20, MRR.

## Experiments

| # | Method | Status |
|---|---|---|
| 0 | Vector search | done |
| 1 | Actor (LLM re-ranks the top 20) | done |
| 2 | Actor + comparator (AvaTaR-style) | done |
| 3 | Actor + debate (critic, defender, judge) | planned |
| 4 | Deception: one debater gives misleading arguments, to test whether the judge can be fooled | planned |

Later: a second dataset, different numbers of debate rounds and agents, and a check of the judge for bias.

## Results so far (100 test questions)

| Method | Hit@1 | Hit@5 | Recall@20 | MRR |
|---|---|---|---|---|
| Vector search | 0.40 | 0.66 | 0.58 | 0.52 |
| Actor | 0.45 | 0.71 | 0.58 | 0.56 |
| Actor + comparator | 0.45 | 0.70 | 0.58 | 0.57 |

The comparator ran 5 rounds on 30 training questions and kept 2 of its 5 rewritten instructions. Training MRR went from 0.622 to 0.630, but on the test questions the result is about the same as the unoptimized actor.

On the full test set (1642 questions), vector search scores Hit@1 0.398, which matches the 0.392 reported in the STaRK paper.

## Run

```bash
pip install -r requirements.txt
python scripts/explore_stark.py --config configs/explore_amazon.yaml
python scripts/run_baseline.py --config configs/baseline0_amazon.yaml
python scripts/run_actor.py --config configs/baseline1_amazon.yaml
python scripts/run_optimizer.py --config configs/baseline2_amazon.yaml
```

Baseline 0 uses STaRK's precomputed embeddings: put them in `emb/amazon/text-embedding-ada-002/`. Baseline 1 needs Ollama running with `ollama pull qwen2.5:7b`.
