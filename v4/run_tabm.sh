#!/usr/bin/env bash
# TabM-style weight sharing vs. our independent experts.
#
# Launch it the way you launch the notebooks (from ~/SRP/v4 on gpu01):
#
#   gpustat                       # pick a free card, check the WhatsApp group first
#   CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#     nohup bash run_tabm.sh > runs/tabm.out 2>&1 &
#
#   tail -f runs/tabm.out         # watch it
#   nvidia-smi | grep python      # confirm it is on the card you picked
#
# One dataset only:  nohup bash run_tabm.sh CO > runs/tabm_co.out 2>&1 &
#
# No notebook here — this calls cli.py directly and writes a report.json per run,
# so there is nothing to nbconvert and nothing to merge back into a .ipynb.
#
# Four configs per dataset. "control" vs "matched" is the comparison that decides it:
# weight sharing is only interesting if it wins at EQUAL parameter count.
#
#   control   K=16 h=16  frac0.7  independent  ~103k params   today's architecture
#   matched   K=32 h=128 frac1.0  shared       ~104k params   same budget, TabM parameterisation
#   wide      K=32 h=256 frac1.0  shared       ~176k params   spend the savings on width
#   wide+bag  K=32 h=256 frac0.7  shared       ~176k params   does OUR feature bagging add anything?
set -u

DATASETS="${*:-CO CA HI}"
OUT=results/tabm
LOGDIR=runs
mkdir -p "$LOGDIR" "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "CUDA_VISIBLE_DEVICES is not set — refusing to start, this would land on a busy card."
  echo "Run gpustat, pick a free one, then prepend CUDA_VISIBLE_DEVICES=<n> to this script."
  exit 1
fi
echo "GPU $CUDA_VISIBLE_DEVICES | datasets: $DATASETS | started $STAMP"
python3 -c "import torch; print('torch sees cuda:', torch.cuda.is_available())"

# name|extra cli flags
CONFIGS=(
  "control|--num-learners 16 --hidden-dim 16  --feature-frac 0.7"
  "matched|--shared --num-learners 32 --hidden-dim 128 --feature-frac 1.0"
  "wide|--shared --num-learners 32 --hidden-dim 256 --feature-frac 1.0"
  "widebag|--shared --num-learners 32 --hidden-dim 256 --feature-frac 0.7"
)

for ds in $DATASETS; do
  for entry in "${CONFIGS[@]}"; do
    name="${entry%%|*}"; flags="${entry#*|}"
    log="$LOGDIR/tabm_${STAMP}_${ds}_${name}.log"
    echo ""
    echo "=== $ds / $name ===  $(date +%H:%M:%S)"
    # conf_uniform is predict-then-average: TabM's own aggregation, and the pooling point
    # measured as worth 0.0194 RMSE on CA. On CO this needs today's multiclass unblock.
    # shellcheck disable=SC2086
    # -u and --line-buffered: without both, python and grep block-buffer into the pipe
    # and the log looks frozen for the whole run, then dumps everything at the end.
    python3 -u cli.py --dataset "$ds" --arm conf_uniform $flags \
        --embedding periodic --seeds 3 --epochs 200 --patience 20 \
        --out "$OUT/$name" 2>&1 | tee "$log" \
        | grep --line-buffered -E "seed |rmse |acc |WARNING|Error|Traceback"
    echo "    full log: $log"
  done
done

echo ""
echo "DONE $(date +%H:%M:%S). Reports in $OUT/, logs in $LOGDIR/."
echo "From your LAPTOP:"
echo "  rsync -avz latif@147.172.179.101:SRP/v4/results/tabm/ ~/Desktop/SRP/v4_server/results_tabm/"
