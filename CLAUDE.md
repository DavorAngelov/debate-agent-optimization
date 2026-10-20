# Project context: Debate-based optimization of LLM agents

## Course / assignment
- Student: Davor Angelov. University course project (Macedonian), **Topic 8: LLM-based agentic systems**
  (prompt-based learning in agentic systems; coordination, negotiation and debate in multi-agent systems).
- The professor wants work at the intersection of this topic and **RL in deep research / agentic systems**,
  ideally **reproducing a paper and extending / re-examining it on another dataset**.
- Papers the professor suggested:
  - CollabLLM: multiturn-aware rewards (arXiv 2502.00640, https://wuyxin.github.io/collabllm/)
  - AvaTaR: optimizing LLM agents for tool use via contrastive reasoning (arXiv 2406.11200)
  - STaRK: semi-structured retrieval benchmark (https://stark.stanford.edu/)
  - OpenScholar (Nature, https://www.nature.com/articles/s41586-025-10072-4)
  - RewardBench: evaluating reward models (arXiv 2403.13787)
  - Multi-Agent Debate (MAD), "Encouraging divergent thinking..." (arXiv 2305.19118)
  - The professor also attached a "representative RL-in-NLP paper", which is not yet identified here.

## Chosen project
**Replace AvaTaR's single "comparator" agent with multi-agent debate.**
- AvaTaR: an actor agent does retrieval on STaRK. A comparator agent contrasts positive and negative
  examples and writes improved instructions (prompt optimization).
- Our variant: **critic** and **defender** agents debate *why* the actor failed, and a **judge** agent
  turns the debate into updated instructions for the actor.
- **Research question:** Does debate give better optimization feedback for an agent than single-agent
  reflection, and under which conditions?

### Experimental design
- Baselines:
  0. Plain vector retrieval
  1. Unoptimized actor
  2. AvaTaR-style comparator
  3. Self-Refine (optional)
  4. **Ours: debate optimizer**
- Data: STaRK (start with **Amazon**, a 100-query subset; later Prime/MAG), plus a second dataset
  (HotpotQA or a scientific QA set, OpenScholar-style).
- Metrics: Hit@1, Recall@20, MRR. Also track token cost and argument diversity
  (MAD's "Degeneration-of-Thought").
- Ablations: number of debate rounds, number of agents.
- Extension (RewardBench-style): measure judge reliability (length, position and sycophancy bias),
  and try a reward model as the judge.
- Compute: API models (OpenAI/Anthropic) or a local Ollama model (e.g. Qwen2.5-7B). No GPU training.
  Keep the API budget small.

## Repo setup (planned)
```
debate-agent-optimization/
├── configs/        # YAML per experiment, fixed seeds
├── src/{data,agents,optimizers,judges,eval}/
├── scripts/        # run_baseline.py, run_debate.py, explore_stark.py
├── results/  notebooks/  paper/
├── external/       # gitignored: clones of zou-group/avatar and snap-stanford/stark
├── .env.example    # OPENAI_API_KEY, ANTHROPIC_API_KEY (.env is gitignored)
└── requirements.txt  # stark-qa, torch, openai, anthropic, python-dotenv, pyyaml, pandas, tqdm, matplotlib
```

## Roadmap (~8 weeks)
1. Weeks 1–2: skeleton and env; load STaRK (`stark_qa.load_qa("amazon")`, `load_skb(...)`);
   Baseline 0; reproduce AvaTaR on the subset.
2. Weeks 3–4: implement `src/optimizers/debate.py` (critic, defender, judge).
3. Weeks 5–6: second dataset and ablations.
4. Week 7: judge analysis (RewardBench-style).
5. Week 8: write the paper.

## Current status
- Skeleton done. Env: `.venv` (Python 3.11). `numpy<2` is pinned because ogb pulls in rdkit, which breaks on NumPy 2.
- `scripts/explore_stark.py` runs. STaRK Amazon: train/val/test = 5910/1548/1642 queries.
  SKB has 1.04M nodes (957k product candidates). Answers per query: mean 5.45, median 3.
  The fixed 100-query test subset (seed 42) is in `results/explore_amazon/subset_queries.json`.
  The SKB lives in `data/stark/amazon/` (gitignored).
- Next: Baseline 0 (vector retrieval). Precomputing embeddings for 957k products is costly, so either use
  STaRK's precomputed ada-002 embeddings or rerank a candidate pool. After that, clone AvaTaR into `external/`.

## Working conventions
- One GitHub Issue per task; feature branches (`feature/<name>`).
- Every experiment is driven by a config, reproducible, and logs results to `results/`.
- Never commit API keys.
