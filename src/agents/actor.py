import re

from src.agents.llm import generate

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
