from src.agents.actor import doc_text
from src.agents.llm import generate

PROMPT = """An assistant ranks products for customer requests using these instructions:

"{instructions}"

Here are requests where the assistant ranked a correct product first:
{good}

Here are requests where the assistant ranked a wrong product first:
{bad}

Compare the good and the bad cases. Find what the assistant does wrong and write improved instructions
that fix it. The instructions must be general, not about these specific products.
Write only the new instructions, at most 120 words."""


def format_examples(results, queries, skb, chars, wrong):
    lines = []
    for r in results:
        line = f"- Request: {queries[r['query_id']]}\n  Ranked first: {doc_text(skb, r['ranked'][0], chars)}"
        if wrong:
            correct = next((p for p in r["ranked"] if p in r["answer_ids"]), r["answer_ids"][0])
            line += f"\n  A correct product: {doc_text(skb, correct, chars)}"
        lines.append(line)
    return "\n".join(lines)


def improve(instructions, good, bad, queries, skb, model, chars):
    prompt = PROMPT.format(instructions=instructions,
                           good=format_examples(good, queries, skb, chars, wrong=False),
                           bad=format_examples(bad, queries, skb, chars, wrong=True))
    answer, tokens = generate(prompt, model=model, temperature=0.7)
    return answer.strip().strip('"'), tokens
