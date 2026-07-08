"""Embed the canonical corpus with one backend. Resumable.
Usage: python scripts/21_embed_canonical.py [pubbert|titan|openai]"""
import sys

from embedd import embed_canonical

if __name__ == "__main__":
    backend = sys.argv[1] if len(sys.argv) > 1 else "titan"
    embed_canonical.build(backend)
