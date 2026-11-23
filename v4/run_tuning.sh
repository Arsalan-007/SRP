#!/usr/bin/env bash
# run_tuning.sh -- per-dataset Optuna tuning (tune_all.py) sharded across two GPUs.
#
# Usage, from ~/SRP/v4:
#   ./run_tuning.sh                 # shard A on GPU 0, shard B on GPU 2
#   ./run_tuning.sh 1 3             # the same two shards on GPUs 1 and 3
#   JOBS=2 THREADS=1 ./run_tuning.sh      # back off if the box is CPU-starved
#   SETS_A="CA" SETS_B="AD" ./run_tuning.sh   # custom split
#
# Resumable by design: tune_all.py skips any (dataset, arm) whose summary.json already
# exists, and Optuna's journal file resumes a partly finished search. Re-running the same
# command after a kill or a crash continues where it stopped -- nothing is recomputed.
#
# The two shards write into the SAME results/tuning tree but touch disjoint <KEY>/ dirs,
# so there is no contention between them.

set -euo pipefail
cd "$(dirname "$0")"

GPU_A=${1:-0}
GPU_B=${2:-2}
JOBS=${JOBS:-3}
THREADS=${THREADS:-2}

# Split so the two shards cost about the same, cheapest first within each so an early kill
# still banks finished datasets. Cost = trials x trial_epochs x rows x n_features:
#   A: CA 0.5% + AD 2% + HI 6% + SA 37%  = 45%
#   B: JA 9%  + FB 11% + CO 13% + YE 21% = 55%
# Santander is alone against CO+YE because its 200 per-feature PLR embeddings make it the
# single most expensive dataset in the suite.
SETS_A=${SETS_A:-"CA AD HI SA"}
SETS_B=${SETS_B:-"JA FB CO YE"}

mkdir -p runs results/tuning
STAMP=$(date +%Y%m%d_%H%M%S)

# python3 -u: without it nohup block-buffers stdout and the log stays empty for hours.
launch() {
  local gpu=$1 tag=$2 sets=$3
  local log="runs/tune_${tag}_${STAMP}.out"
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup python3 -u tune_all.py --datasets $sets --jobs "$JOBS" --threads "$THREADS" \
    > "$log" 2>&1 &
  ln -sfn "$(basename "$log")" "runs/tune_${tag}.out"
  printf 'GPU %-2s shard %s  [%s]  pid %s  -> %s\n' "$gpu" "$tag" "$sets" "$!" "$log"
}

echo "== self-test =="
python3 tune_all.py --selftest

echo
echo "== launching =="
launch "$GPU_A" A "$SETS_A"
launch "$GPU_B" B "$SETS_B"

cat <<MSG

follow:   tail -f runs/tune_A.out runs/tune_B.out
progress: grep -h ': DONE' runs/tune_[AB].out
results:  results/tuning/<KEY>/conf_uniform/summary.json
stop:     pkill -f tune_all.py
MSG
