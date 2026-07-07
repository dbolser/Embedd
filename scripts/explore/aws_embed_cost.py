"""Measure real Titan token counts on a sample of our abstracts and estimate the
cost of embedding 10k (and the whole corpus) via AWS Bedrock, vs OpenAI for
reference. Also validates that Bedrock model access is actually enabled.

Throwaway exploration. Makes a few tiny paid Bedrock calls (~cents at most).
Usage: python scripts/explore/aws_embed_cost.py [sample_n]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

from embedd import embed

REGION = "eu-west-2"
MODEL = "amazon.titan-embed-text-v2:0"

# on-demand $/1M tokens (approx, region eu-west-2 / us pricing)
PRICES = {
    "Titan Text Embeddings V2 (AWS)": 0.02,
    "Cohere Embed v3/v4 (AWS)": 0.10,
    "OpenAI ada-002 (ref)": 0.10,
    "OpenAI text-embedding-3-small (ref)": 0.02,
    "OpenAI text-embedding-3-large (ref)": 0.13,
}


def titan_tokens(text: str) -> int | None:
    env = dict(os.environ, PYTHONWARNINGS="ignore")  # keep AWS creds/config
    with tempfile.TemporaryDirectory() as d:
        bpath, opath = os.path.join(d, "b.json"), os.path.join(d, "o.json")
        with open(bpath, "w") as fh:
            json.dump({"inputText": text[:40000]}, fh)
        cmd = [
            "aws", "bedrock-runtime", "invoke-model", "--region", REGION,
            "--model-id", MODEL, "--content-type", "application/json",
            "--accept", "application/json", "--body", f"fileb://{bpath}", opath,
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if p.returncode != 0:
            if not hasattr(titan_tokens, "_shown"):
                print("  (invoke error:", p.stderr.strip().splitlines()[-1:], ")")
                titan_tokens._shown = True
            return None
        return json.load(open(opath)).get("inputTextTokenCount")


def main(sample_n=20):
    records = embed.load_corpus()
    texts = [f"{r['title']} {r['abstract']}" for r in records]
    words = [len(t.split()) for t in texts]
    avg_words = sum(words) / len(words)
    print(f"corpus: {len(texts)} abstracts, avg {avg_words:.0f} words (title+abstract)")

    # sample evenly and measure real Titan token counts
    idx = list(range(0, len(texts), max(1, len(texts) // sample_n)))[:sample_n]
    toks = []
    for i in idx:
        t = titan_tokens(texts[i])
        if t:
            toks.append((t, words[i]))
    if not toks:
        print("Bedrock invoke failed on all samples -> model access likely NOT enabled.")
        return
    avg_tok = sum(t for t, _ in toks) / len(toks)
    ratio = sum(t for t, _ in toks) / sum(w for _, w in toks)
    print(f"measured {len(toks)} real abstracts: avg {avg_tok:.0f} tokens "
          f"({ratio:.2f} tokens/word)")

    est_tok_per_abs = avg_words * ratio
    for n in (10_000, len(texts)):
        total_tok = n * est_tok_per_abs
        print(f"\n--- {n} abstracts  (~{total_tok/1e6:.2f}M tokens) ---")
        for name, price in PRICES.items():
            print(f"    {name:38s} ${total_tok/1e6*price:6.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
