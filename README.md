# SRP — Sequential Residual Predictor

A neural architecture for tabular data: an ensemble of small "weak learner" MLPs
produces a sequence of feature tokens, which a causal transformer (attention) block
then combines into a final prediction — an end-to-end, differentiable analogue of
gradient-boosted trees (weak learners + sequential refinement) built in PyTorch.

See [main.tex](main.tex) for the full presentation (slides source) and
[q-dev-chat-2026-05-10.md](q-dev-chat-2026-05-10.md) for the original design rationale
and diversity-regularization ideas behind the architecture.

## Layout

- **Root (`srp_model.py`, `srp_model_v2.py`, `weak_learners.py`, `diversity.py`,
  `weight_schemes.py`, `preprocessing.py`, `train.py`, `train_v2.py`, `rate.py`)** —
  the original SRP model, training loop, and diversity-regularization experiments.
  `SRP_Colab*.ipynb` are the corresponding exploratory notebooks.

- **`tabjoint_vishal_averaged_experiments/`** — an earlier experiment line (CKA
  similarity analysis, grid search, joint-training variants) that fed into later
  design decisions.

- **`v4/`** — the current, most rigorous iteration. Compares several ways of
  aggregating weak-learner outputs (mean pooling, bidirectional attention, causal
  attention, causal + deep supervision) under matched capacity and shared training
  code, benchmarked against the published Gorishniy et al. (2022) numbers. See
  `v4/model.py` for the arm definitions and `v4/v1/README.md` for known quirks of
  the V1 reference implementation. Numbered notebooks (`01`–`07`) are the benchmark,
  ablation, and diagnostic runs; `v4_server/07_v4_vs_v1_gpu.ipynb` is the same
  notebook with full GPU run outputs.

## Data

Dataset arrays (`v4/data/**/*.npy`, `*.npz`, `*.tar`) are **not** tracked in git —
they're large (the raw archive alone is ~3.2GB) and easy to regenerate/refetch.
Only small metadata (`info.json`, `published_gorishniy2022.json`,
`archive_listing.txt`) is kept for reference. Ask the repo owner for the data
folder if you need to run training/benchmarks locally.
