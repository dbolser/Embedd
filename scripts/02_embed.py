"""Embed the corpus. Usage: python scripts/02_embed.py [local|openai]"""
import sys

from embedd import embed

if __name__ == "__main__":
    backend = sys.argv[1] if len(sys.argv) > 1 else "local"
    vecs, records = embed.build(backend=backend)
    print(f"embedded {len(records)} abstracts with backend={backend}")
