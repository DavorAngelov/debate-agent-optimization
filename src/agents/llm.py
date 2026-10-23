import os

import requests

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def generate(prompt, model="qwen2.5:7b", temperature=0.0, seed=42):
    r =requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "seed": seed, "num_ctx": 8192},
    }, timeout=600)
    r.raise_for_status()
    out = r.json()
    return out["response"], out.get("prompt_eval_count", 0) + out.get("eval_count", 0)
