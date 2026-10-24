import re

from src.agents.llm import generate
from src.eval.metrics import evaluate

DEFAULT_INSTRUCTIONS = "Rank the products by how well they match the customer's request."

PROMPT = """{instructions}

Customer request: {query}

Products:
{products}

Answer with the numbers of the products that match the request, best match first, separated by commas.
Answer only with numbers."""


def rerank(query, candidates, instructions, model):
    products = "\n".join(f"[{i + 1}] {text}" for i, (_, text) in enumerate(candidates))
    prompt = PROMPT.format(instructions=instructions, query=query, products=products)
    answer, tokens = generate(prompt, model=model)

    order = []
    for n in re.findall(r"\d+", answer):
        i = int(n) - 1
        if 0 <= i < len(candidates) and i not in order:
            order.append(i)
    order += [i for i in range(len(candidates)) if i not in order]
    return [candidates[i][0] for i in order], answer, tokens


def doc_text(skb, pid, chars):
    return skb.get_doc_info(pid, add_rel=False)[:chars].replace("\n", " ")


def run_actor(rows, queries, skb, instructions, model, doc_chars):
    results = []
    for row in rows:
        candidates = [(pid, doc_text(skb, pid, doc_chars)) for pid in row["top20"]]
        ranked, answer, tokens = rerank(queries[row["query_id"]], candidates, instructions, model)
        results.append({"query_id": row["query_id"], "ranked": ranked, "answer_ids": row["answer_ids"],
                        "llm_answer": answer, "tokens": tokens, **evaluate(ranked, row["answer_ids"])})
    return results
