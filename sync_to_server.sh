#!/usr/bin/env bash
# Sync SRP/v4 from THIS LAPTOP to the university GPU server.
# Run it on the laptop:   bash ~/Desktop/SRP/sync_to_server.sh
set -euo pipefail

SERVER="latif@147.172.179.101"
REMOTE_DIR="SRP/v4"                      # relative to your server home directory
LOCAL_DIR="$HOME/Desktop/SRP/v4/"

# Guard: this must run on the laptop, where the files live.
if [ "$(hostname)" = "gpu01" ]; then
    echo "ERROR: you are on the SERVER (gpu01)."
    echo "Type 'exit' until the prompt shows your laptop, then run this script again."
    exit 1
fi
if [ ! -d "$LOCAL_DIR" ]; then
    echo "ERROR: $LOCAL_DIR not found — are you on the laptop?"
    exit 1
fi

echo "Sending $(du -sh "$LOCAL_DIR" --exclude=data/raw --exclude=__pycache__ --exclude=runs | cut -f1) from $(hostname) to $SERVER:$REMOTE_DIR"
echo "(you will be asked for your server password)"
# --backup: any server file this would overwrite is kept under SRP/v4_overwritten_<date>/,
# so syncing can never destroy a notebook whose results you have not copied back yet.
rsync -avz --human-readable \
      --backup --backup-dir="../v4_overwritten_$(date +%Y%m%d_%H%M%S)" \
      --exclude '__pycache__' --exclude 'runs' --exclude 'data/raw' \
      "$LOCAL_DIR" "$SERVER:$REMOTE_DIR/"

echo
echo "Done. On the server, verify with:"
echo "  ls ~/SRP/v4          # must include 07_v4_vs_v1_gpu.ipynb, embeddings.py, benchmarks.py, v1/"
echo "  du -sh ~/SRP/v4/data # must be about 210 MB"
