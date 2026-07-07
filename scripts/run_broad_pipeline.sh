#!/bin/bash
# Wait for the broad fetch to finish, then Titan-embed the broad corpus.
cd /home/danbolser/Build/Embedd
source .venv/bin/activate
export PYTHONPATH=/home/danbolser/Build/Embedd PYTHONWARNINGS=ignore
echo "waiting for broad fetch to finish..."
for i in $(seq 1 360); do
  grep -q "ALL DONE" results/fetch_broad.log 2>/dev/null && { echo "fetch done"; break; }
  sleep 20
done
echo "=== launching Titan embed ==="
python -u scripts/13_embed_broad_titan.py
echo "=== broad pipeline complete ==="
