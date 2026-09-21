# SRP v4 — Weak-Learner Ensembles for Tabular Deep Learning<!-- omit in toc -->

> [!IMPORTANT]
> **Guiding rule for this whole project:** never assume a new idea helps. Build it as an
> isolated, switchable piece, run it next to the old version under an identical protocol,
> and let the numbers say yes or no. Every section below follows from this rule.

**What this project is asking:** can an ensemble of small neural networks ("weak
learners"), combined in a learned way, close the gap to Gradient Boosted Decision Trees
(GBDTs like XGBoost) on tabular data? This is v4 — a from-scratch rebuild after V1/V2/V3
were reviewed and found to have untested claims.

Table of Contents:
- [The main results](#the-main-results)
- [How to reproduce the results](#how-to-reproduce-the-results)
    - [Set up the environment](#set-up-the-environment)
        - [Software](#software)
        - [Data](#data)
        - [GPU cluster access](#gpu-cluster-access)
    - [Quick test](#quick-test)
    - [Tutorial](#tutorial)
    - [Reproducing other results](#reproducing-other-results)
- [Understanding the repository](#understanding-the-repository)
    - [The architecture, in plain terms](#the-architecture-in-plain-terms)
    - [Code overview](#code-overview)
    - [Running notebooks](#running-notebooks)
    - [Technical notes](#technical-notes)
- [Research log](#research-log)
- [Appendix A: PLR embeddings, explained fully](#appendix-a-plr-embeddings-explained-fully)
- [Open threads / what's next](#open-threads--whats-next)

# The main results

The strongest, most-recent, apples-to-apples comparison: v4's 4 arms (mean-pool / causal
attention, each with/without PLR embeddings) plus a PlainMLP reference and our own tuned
XGBoost, against **published** numbers from Gorishniy, Rubachev, Khrulkov & Babenko
(2021), *Revisiting Deep Learning Models for Tabular Data* (arXiv:2106.11959) — 5
datasets, their exact published splits, ONE fixed v4 config (not tuned per dataset, unlike
every published number):

| Model | Params | California Housing (RMSE↓) | Adult (acc↑) | Jannis (acc↑) | Covertype (acc↑) | YearPredictionMSD (RMSE↓) |
|---|---|---|---|---|---|---|
| v4 mean-pool | 37.6k | 0.5404 | 0.8589 | 0.7195 | 0.9324 | **8.8293** |
| v4 mean-pool + PLR | 50.6k | 0.5044 | 0.8594 | **0.7225** | **0.9486** | 8.8621 |
| v4 causal attn | 37.7k | 0.4985 | 0.8567 | 0.7097 | 0.9100 | 8.8472 |
| **v4 causal attn + PLR** | 50.7k | **0.4834** | **0.8608** | 0.7195 | 0.9336 | 8.8425 |
| PlainMLP | 34.3k | 0.5117 | 0.8543 | 0.7127 | 0.9203 | 8.8754 |
| Our XGBoost (tuned) | — | 0.4328 | 0.8739 | 0.7233 | 0.9689 | 8.8647 |
| *MLP (published)* | — | *0.499* | *0.852* | *0.719* | *0.962* | *8.853* |
| *ResNet (published)* | — | *0.486* | *0.854* | *0.728* | *0.964* | *8.846* |
| *NODE (published)* | — | *0.464* | *0.858* | *0.727* | *0.958* | *8.784* |
| *FT-Transformer (published)* | — | *0.459* | *0.859* | *0.732* | *0.970* | *8.855* |
| *XGBoost (published)* | — | *0.431* | *0.872* | *0.724* | *0.969* | *8.819* |

**Headline finding:** v4 beats the published, tuned FT-Transformer on 2 of 5 datasets —
Adult (0.8608 vs 0.859) and ties XGBoost on YearPredictionMSD (8.8293 vs 8.819, beating
published FT-Transformer's 8.855) — using one fixed, never-tuned-per-dataset config.
Covertype is the clear weak spot (see below).

**What happened on Covertype since that table:** per-dataset tuning closed 41% of the gap
to XGBoost (0.9324 → 0.9569 tuned) but not the rest; feature gating — a candidate
architectural fix — was then tried and did **not** move the needle beyond seed noise. Full
story, all the numbers, and what's next: [Research log](#research-log) below.

For V1-vs-v4 (your predecessors' architecture vs. this one), see
[Research log → Step 6](#research-log).

---

# How to reproduce the results

## Set up the environment

### Software

This project runs on a shared university GPU cluster. The conda environment is already
set up there as `srp` (not `base` — `base` does not have PyTorch installed).

```shell
ssh <you>@147.172.179.101
cd ~/SRP/v4
conda activate srp
python3 -c "import torch, xgboost, optuna; print(torch.__version__, torch.cuda.is_available())"
```

You want a torch version printed and `True` for CUDA availability. If you're setting this
environment up for the first time (fresh server account), install PyTorch matching the
server's driver version (`nvidia-smi` shows it; e.g. driver 535.x needs the `cu126` build,
not the newer default), plus `xgboost`, `optuna`, `pandas`, `scikit-learn`, `matplotlib`,
`seaborn`, `joblib`, `jupyter`/`nbconvert`.

### Data

Two tiers of datasets, loaded by two different modules:

- `datasets.py` — California Housing, Covertype, Poker Hand, HIGGS as small (10k-row)
  diagnostic samples with our own reproducible splits. Auto-downloaded on first use.
- `benchmarks.py` / `benchmarks_rtdl.py` — full-scale datasets with their **exact
  published splits**, so results are directly comparable to numbers in the source papers.
  These come from pre-packaged archives that need to be extracted into `data/benchmarks/`
  and `data/benchmarks_rtdl/` respectively (see each module's own `extract()` /
  `__main__` block for the exact archive URLs and layout). A `READY` marker file per
  dataset folder confirms extraction succeeded.

Datasets are not synced to git or between laptop↔server via the sync script (see below) —
they live only in `data/` on whichever machine downloaded them.

### GPU cluster access

The whole workflow is: edit code on your laptop → sync to the server → run there → copy
results back. **The person running Claude Code never SSHes into the server directly on
your behalf** — every server command below is one you run yourself.

**1. Sync your laptop's `v4/` to the server** (run this on your laptop):
```shell
bash ~/Desktop/SRP/sync_to_server.sh
```
This refuses to run if you're already on the server (checks `hostname`), and backs up
anything it would overwrite under `../v4_overwritten_<timestamp>/` — syncing can never
silently destroy a notebook whose results you haven't copied back yet.

**2. Pick a GPU.** This is a shared cluster — check load before claiming one:
```shell
gpustat
```
Prefer the GPU with the **lowest utilization %** that also has enough free memory for
your model (a few GB is plenty for anything in this project so far). Full memory but low
utilization is a better pick than empty memory but 100% utilization — you're contending
for compute, not just space.

**3. Run with that GPU explicitly selected:**
```shell
CUDA_VISIBLE_DEVICES=<gpu id> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb
```

**4. Copy results back to your laptop** (run this on your laptop):
```shell
rsync -avz <you>@147.172.179.101:SRP/v4/<notebook>.ipynb ~/Desktop/SRP/v4_server/
```

**Golden rules:**
- Always copy a finished notebook's results back from the server **before** syncing new
  code to it — `--inplace` execution overwrites the notebook file itself, and the sync
  script's backup only protects against the sync overwriting things, not a fresh run.
- **Never Ctrl-C a long run.** `jupyter nbconvert --to notebook --execute --inplace` does
  **not** checkpoint incrementally — it writes the output file exactly once, at the very
  end, after every cell finishes. Interrupting loses the *entire* run's results, not just
  whatever hadn't finished yet.
- For anything that will run longer than your SSH session might last, detach it:
  ```shell
  nohup jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb > run.log 2>&1 &
  disown
  ```
  then watch it via the notebook's own progress log (see [Running notebooks](#running-notebooks)).

## Quick test

To check the environment and code are both wired correctly, every core module has its own
self-test — run any of these directly (no GPU needed, seconds to run):

```shell
cd ~/SRP/v4
python3 weak_learners.py    # feature bagging + ensemble forward pass
python3 attention.py        # causal masking correctness
python3 embeddings.py       # PLR rotation-sensitivity checks
python3 feature_gating.py   # gate mode shape/gradient/identity-at-init checks
python3 model.py            # full arms build correctly, capacity-matched, identical init
```
Each prints `[PASS]`/`[FAIL]` for its own checks. All should read `[PASS]` before trusting
any notebook result.

For an end-to-end (GPU) smoke test, any notebook accepts environment-variable overrides
that shrink it to run in well under a minute — see [Running notebooks](#running-notebooks).

## Tutorial

Here we reproduce Step 1's result: the weak-learner ensemble (mean-pooled, no attention,
no embeddings) vs. XGBoost on California Housing.

```shell
cd ~/SRP/v4
conda activate srp
gpustat                                             # pick a GPU
CUDA_VISIBLE_DEVICES=<gpu id> \
  jupyter nbconvert --to notebook --execute --inplace 01_baseline_benchmark.ipynb
```

This takes a few minutes. When it finishes, open the notebook — the last cells contain a
findings summary computed from that run (never hand-typed), plus comparison plots. Copy it
back to your laptop with `rsync` as shown above to view it outside the SSH session.

*Reproducing any other notebook is the same pattern — swap the filename.* Notebooks 07
onward are full-scale GPU runs and take much longer (tens of minutes to ~1.5 hours); use
the environment-variable smoke test first (see [Running notebooks](#running-notebooks)).

## Reproducing other results

| Notebook | Reproduces | Runtime |
|---|---|---|
| `01_baseline_benchmark.ipynb` | Step 1: weak learners + mean-pool vs XGBoost (housing) | minutes |
| `02_attention_ablation.ipynb` | Step 2: does causal attention beat mean-pool? (housing) | minutes |
| `03_optuna_num_learners.ipynb` | Step 3: how many weak learners (K) do we need? | minutes |
| `04_grinsztajn_diagnostics.ipynb` | Step 4: why do trees still win? (3 tests, 3 datasets) | minutes |
| `05_v1_published_benchmarks.ipynb` | superseded by 07 — kept for its data-loading notes | — |
| `06_feature_embeddings.ipynb` | Step 5: does the PLR fix work? (4 datasets) | minutes |
| `07_v4_vs_v1_gpu.ipynb` | Step 6: v4 vs V1 head-to-head, published splits (GPU) | ~30-60 min |
| `08_v4_vs_literature.ipynb` | Step 7: v4 vs published MLP/ResNet/NODE/FT-Transformer/XGBoost, 5 datasets (GPU) | ~1-1.5 hr |
| `09_tune_covertype.ipynb` | Step 8: is Covertype's gap just untuned hyperparameters? (GPU) | ~45 min |
| `10_feature_gating_covertype.ipynb` | Step 9: does feature gating help on top of the tuned config? (GPU) | ~80 min |

Full narrative for every row — what was tried, why, and what it found — is in the
[Research log](#research-log).

---

# Understanding the repository

## The architecture, in plain terms

```
raw row (features)
      │
      ▼
┌─────────────────┐
│ FEATURE EMBEDDING │   optional. Encodes EACH feature on its own, before anything
│  (embeddings.py)  │   is mixed together. "none" = skip this step entirely.
└─────────────────┘
      │
      ▼
┌─────────────────┐
│  WEAK LEARNERS    │   K small MLPs ("experts"). Each one only sees a random subset
│ (weak_learners.py)│   of the input features ("feature bagging") — this is what makes
└─────────────────┘   them different from each other, without needing a penalty loss.
      │
      ▼            each expert outputs one opinion → K opinions total
┌─────────────────┐
│   AGGREGATOR       │   how the K opinions get combined into one prediction:
│                    │     "mean-pool"  = simple average (the control / simplest option)
│ (attention.py,     │     "causal attn" = a Transformer-style attention mechanism where
│  model.py)         │                     later experts can "read" earlier ones first
└─────────────────┘
      │
      ▼
┌─────────────────┐
│  PREDICTION HEAD   │   a small final network → the actual output (price / class)
└─────────────────┘
```

**Key terms, defined once:**

| Term | Meaning |
|---|---|
| **Weak learner / expert** | One small MLP in the ensemble. K of them exist (e.g. K=16). |
| **Feature bagging** | Each expert only sees a random ~70% subset of the input columns. This is the project's diversity mechanism — it stops all experts from learning the same thing, without needing an artificial "diversity penalty" (which V1 needed and which we proved is unnecessary). |
| **Mean-pool** | Aggregator that just averages all K experts' outputs. The simplest possible combination — used as the control everything else must beat. |
| **Causal attention** | Aggregator using masked self-attention: expert *i* can "see" experts *1..i* but not later ones — inspired by the idea that later experts should build on earlier ones, like stages in boosting. |
| **Embedding mode `none`** | No feature embedding stage — features go straight into the experts as raw numbers. |
| **Embedding mode `linear`** | Each feature gets its own small `Linear → ReLU` before mixing. Tests whether *separating* features (without anything fancy) helps. |
| **Embedding mode `periodic` (a.k.a. "PLR")** | Each feature gets `Periodic (sin/cos waves) → Linear → ReLU`. Lets the model represent sharp, threshold-like behaviour in a single feature — see [Appendix A](#appendix-a-plr-embeddings-explained-fully). Currently the best-performing option. |
| **Feature gating** | A learned, optionally input-dependent per-feature weight applied before each expert's first layer — 5 switchable modes, see [Research log → Step 9](#research-log). Tried, did not clearly help. |
| **Arm** | One full configuration under test (e.g. "causal attn + PLR"). Notebooks compare several arms side by side under identical training. |
| **V1** | Your predecessors' original architecture (`tabjoint_vishal_averaged_experiments/`, vendored unchanged in `v1/`): bidirectional attention + a `[CLS]` token + a CKA diversity penalty. Kept as a permanent comparison point. |
| **PlainMLP** | One ordinary MLP, no expert structure at all. The "floor" reference — if we can't beat this, the whole ensemble idea isn't earning its complexity. |

## Code overview

```
v4/
├── weak_learners.py    The K small expert MLPs + feature bagging.
├── attention.py         The causal (and bidirectional) attention aggregator.
├── embeddings.py        Per-feature embeddings: none / linear / periodic (PLR).
├── feature_gating.py     Adaptive per-learner feature relevancy: g_t(x) in
│                         F_t(x) = F_{t-1}(x) + eta*h_t(g_t(x)*x). 5 switchable modes
│                         (none / global|per_learner x static|dynamic).
├── model.py              Wires the above into full models ("arms"), incl. make_arm().
├── training.py           ONE shared training loop used by every notebook (early
│                         stopping, device handling, chunked evaluation for GPU memory).
├── datasets.py           Loaders for California Housing, Covertype, Poker Hand, HIGGS
│                         (10k-row diagnostic samples, NOT the full published splits).
├── benchmarks.py         Loader for 3 PUBLISHED datasets (num-embeddings paper): Higgs
│                         Small, Facebook Comments, Santander — exact paper splits.
├── benchmarks_rtdl.py     Loader for 5 more PUBLISHED datasets (FT-Transformer paper):
│                         California Housing, Adult, Jannis, Covertype, YearPredictionMSD
│                         — exact paper splits, all pulled from ONE archive/paper so the
│                         published comparison table is internally consistent.
├── diagnostics.py        The 3 Grinsztajn et al. perturbation tests (noise / rotation /
│                         label smoothing) used to diagnose weaknesses vs. trees.
├── tuning.py              Optuna hyperparameter search machinery, incl. embedding-mode
│                         search and an opt-in gate_mode override.
├── v1/                    Your predecessors' code, copied UNCHANGED (do not edit).
├── v1_runner.py           Wrapper to run V1 under the same protocol as v4.
└── data/                  Downloaded/cached datasets (not synced to git or between
                           laptop/server — see .gitignore and sync_to_server.sh).
```

## Running notebooks

Every notebook past 01-02 accepts environment-variable overrides so a headless run needs
no code edits — used for smoke tests before committing to a long run:

| Variable | Meaning |
|---|---|
| `SRP_SUBSAMPLE` | Cap train/val/test to N rows — makes a run finish in seconds. |
| `SRP_TRIALS` | Number of Optuna trials (tuning notebooks). |
| `SRP_TRIAL_EPOCHS`, `SRP_TRIAL_PATIENCE` | Epoch budget per Optuna trial (small, cheap). |
| `SRP_FINAL_EPOCHS`, `SRP_FINAL_PATIENCE` | Epoch budget for the final confirm/report run (full). |
| `SRP_FINAL_SEEDS` | How many seeds the final result is averaged over. |
| `SRP_JOBS` | Parallel workers sharing the GPU (default: 3 on CUDA). |
| `SRP_KEY`, `SRP_ARM` | Which dataset / which arm, for notebooks scoped to one of each. |

Example smoke test (finishes in well under a minute, numbers are meaningless but every
cell running error-free is the point):
```shell
SRP_SUBSAMPLE=1200 SRP_FINAL_SEEDS=2 SRP_FINAL_EPOCHS=2 SRP_FINAL_PATIENCE=2 SRP_JOBS=2 \
  jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb
```

**Progress logging.** Long GPU notebooks write a timestamped log under `runs/`
(`runs/<NN>_progress_<timestamp>.log`) with `runs/<NN>_progress.log` symlinked to the
latest — never deleted on restart, so a crashed run's partial progress survives even if
the notebook itself doesn't. Watch it live with:
```shell
tail -f runs/<NN>_progress.log
```

## Technical notes

- **`--inplace` writes once, at the end.** Covered above, worth repeating: never Ctrl-C a
  running `nbconvert --execute --inplace` — there is no incremental checkpointing.
- **Feature subsets are frozen at construction**, not resampled per epoch. v1 resampled
  them every epoch, which quietly changes what each expert "means" over time and makes
  per-expert diagnostics uninterpretable. v4 draws them once, from a seed, and keeps them
  for the model's lifetime.
- **Identical expert init across arms, for a fixed seed** — `make_arm()` constructs the
  weak-learner ensemble first and runs any capacity-matching math under a forked RNG, so
  every arm compared for a given seed starts from bit-identical expert weights. This is
  what makes per-seed paired comparisons (not just averages) valid.
- **Chunked GPU evaluation.** `training.py`'s `evaluate()`/`predict()` process the
  validation/test set in batches (default 4096 rows) rather than one pass, to avoid CUDA
  OOM when several parallel workers share a GPU. Verified numerically identical to a
  single-pass evaluation to 7 decimal places.
- **`gate="none"` and `embedding="none"` are true no-ops**, not just "small effect" —
  verified via bit-for-bit/parameter-count regression checks in `model.py`'s own
  self-test, so every ablation's control arm is exactly today's un-augmented v4.
- **Ephemeral run artifacts are gitignored**: `runs/*.log`, `runs/*.journal` (Optuna
  study files). They persist on whichever machine produced them but are never synced by
  `sync_to_server.sh` or committed — treat anything under `runs/` as local, disposable
  scratch, except when a notebook explicitly reads one back (e.g. notebook 10 reloading
  notebook 09's saved study to avoid re-typing its winning hyperparameters).
- **Findings are always computed from the actual run**, never hand-typed into a cell —
  every notebook's closing "Findings" section is Python reading `study.best_params`,
  a results DataFrame, etc., specifically so a stale or wrong number can't survive a
  copy-paste.

---

# Research log

This section is the full, in-order account of what was tried, why, and what it found —
the reasoning trail behind [The main results](#the-main-results) above.

### Step 1-2 — Does the basic idea work at all? (housing only)
Built the weak-learner ensemble and tried mean-pooling vs. causal attention.
**Finding:** causal attention was statistically indistinguishable from plain mean-pooling.
The apparent gain over a plain MLP baseline turned out to just be extra parameters, not
anything architectural. This was an important, humbling result — it meant the core
"boosting-like sequential attention" idea hadn't proven itself yet.

### Step 3 — How many weak learners (K) do we need?
Ran an Optuna search over K (2 to 64) jointly with other hyperparameters.
**Finding:** performance plateaus after about K=4-8. More experts mostly just cost time.

### Step 4 — Why do trees still win? (the Grinsztajn diagnostics)
Implemented the 3 tests from Grinsztajn et al. (2022), *"Why do tree-based models still
outperform deep learning on typical tabular data?"*, on 3 new datasets (Covertype, Poker
Hand, HIGGS):
1. **Add useless noise columns** — does the model ignore them (like trees do)?
2. **Randomly rotate all features** — does the model notice (like trees do)?
3. **Smooth the labels** — does the model lose its edge on sharp/irregular functions?

**Finding (the important one):** our model was **rotation-invariant** — rotating the
input features cost trees up to 17 accuracy points and cost us essentially nothing. That
sounds harmless but it's actually damning: it means the model treats every feature the
same as any scrambled mixture of features, and can never learn "if income > 50k, then...".
This pinpointed exactly what to fix next.

### Step 5 — Fix it: per-feature embeddings (`embeddings.py`)
Built a stage that encodes every feature *individually*, before any mixing happens,
with an optional periodic ("PLR") component for sharp thresholds. See
[Appendix A](#appendix-a-plr-embeddings-explained-fully) for full detail.
**Finding, tested on 4 datasets (10k-row samples):** the fix worked on both counts —
scores went up on almost every dataset/arm combination, AND (the mechanism check) the
model became measurably sensitive to rotation for the first time, proving we'd actually
fixed the diagnosed problem rather than just gotten lucky.

### Step 6 — Confirm it at full scale + compare to V1 and the literature (GPU)
Ran V1 and 4 v4 variants (mean-pool / causal-attn, each with/without PLR) on 3 datasets
with their **exact published splits** (Higgs Small 98k rows, Facebook Comments 197k rows,
Santander 200k rows), next to a tuned XGBoost and the paper's published numbers for 33
other models.

| Dataset | V1 | Best v4 | Margin |
|---|---|---|---|
| Higgs Small (acc↑) | 0.7178 | **0.7245** | +0.0067 |
| Facebook Comments (RMSE↓) | 6.0379 | **5.7932** | −0.2447 |
| Santander (acc↑) | 0.9140 | **0.9227** | +0.0087 |

v4 beat V1 on all 3 datasets using **~1/10th the parameters** (41k-85k vs. V1's 416k).
The PLR embeddings improved EVERY arm on EVERY dataset. Our own XGBoost reproduced the
published XGBoost numbers closely (within noise), confirming the whole comparison is
trustworthy. Mean-pool and causal-attn ended up close to each other — attention still
hasn't clearly proven it beats simple averaging.

**Where we stood vs. the literature:** near parity with published GBDTs on Santander,
close on Higgs, still behind on Facebook Comments — and this is all *without* per-dataset
tuning, unlike every published number.

### Step 7 — v4 vs. the wider literature (5 standard benchmarks, incl. 2 large-scale)

Ran v4's 4 arms + PlainMLP + our own tuned XGBoost on **5 more datasets with their exact
published splits** (Gorishniy, Rubachev, Khrulkov & Babenko, 2021, arXiv:2106.11959, the
FT-Transformer paper), next to that paper's own published numbers for MLP, ResNet, NODE,
and FT-Transformer. Full table: see [The main results](#the-main-results) above.

**Sanity check passed:** our own tuned XGBoost reproduces the published XGBoost within
~0.002-0.05 on every dataset, so the comparison is trustworthy.

**Findings (the strongest result in the project at the time):**
* **v4 beats the published, tuned FT-Transformer on 2 of 5 datasets** — Adult (0.8608 vs
  0.859) and effectively ties XGBoost on YearPredictionMSD (8.8293 vs 8.819, beating
  published FT-Transformer's 8.855) — using ONE fixed config, never tuned per dataset.
* **v4's best arm beats PlainMLP on all 5 datasets** — the ensemble idea is earning its
  complexity, consistently, for the first time across this many datasets at once.
* **Covertype is the clear weak spot**: v4 trails every reference, including published
  MLP. This is one of the two large-scale (500k+ row) stress tests, and the one where
  the architecture struggles most — a natural target for whatever comes next.
* **PLR embeddings helped in 9 of 10 dataset x aggregator combinations** — the one
  exception, YearPredictionMSD mean-pool, got measurably WORSE with PLR (a real
  regression, not noise). First clean counter-example in the project: PLR is not a
  universal win.
* No std is reported in this paper's tables (unlike Step 6's paper), so "published"
  numbers here are single reference points, not distributions.

### Step 8 — Is Covertype's gap just untuned hyperparameters?

Every v4 number above used ONE fixed config (designed on housing) reused unchanged
everywhere — while every published number is tuned per dataset. Before building anything
new, notebook 09 asked the cheap question first: extended `tuning.py`'s Optuna search
(already used for K in notebook 03) to also search the embedding mode/width, and pointed
it at Covertype's cheapest, already-best arm (mean-pool). Two-phase protocol: 24-trial
parallel search (small epoch budget, validation loss), then the winning config re-trained
on 3 fresh seeds at notebook 08's full epoch budget, reported on test.

| Model | Test acc |
|---|---|
| FT-Transformer (published) | 0.9700 |
| XGBoost (published) | 0.9690 |
| Our XGBoost (tuned) | 0.9689 |
| ResNet (published) | 0.9640 |
| MLP (published) | 0.9620 |
| NODE (published) | 0.9580 |
| **v4 mean-pool — TUNED (notebook 09)** | **0.9569 ± 0.0020** |
| v4 mean-pool + PLR (untuned, notebook 08) | 0.9486 |
| v4 causal attn + PLR (untuned, notebook 08) | 0.9336 |
| v4 mean-pool (untuned, no embedding) | 0.9324 |

**Findings:**
* Tuning genuinely helped (+0.0083 over the best untuned config, beyond the 0.0020 seed
  noise) but closed only **41% of the gap to XGBoost** (0.0203 → 0.0120) and still trails
  the published plain MLP, the weakest deep baseline in the paper.
* The search independently rediscovered that PLR helps — the best trial picked
  `embedding_mode=periodic` (15/24 completed trials did), cross-validating Step 5/7 via a
  completely different process. fANOVA ranked it only 5th of 11 params, though: capacity/
  regularization knobs (`num_learners`, `dropout`, `lr`, `feature_frac`) mattered more here.
* **Verdict, computed by the notebook itself:** "MOSTLY NOT hyperparameters — tuning
  recovered under half the gap. This is real evidence for trying an architectural fix
  (e.g. feature gating) here." This is what motivated Step 9.

### Step 9 — Feature gating: adaptive per-learner feature relevancy

Feature bagging (`weak_learners.py`) gives each learner a fixed, HARD column subset, frozen
at construction — good for diversity, but a learner stuck with mostly-noisy columns has no
way to lean on its one useful column more. A second predecessor research document proposed
a boosting-style fix: a learned, optionally input-dependent SOFT gate reweighting each
learner's view of the data before its first layer,

```
F_t(x) = F_{t-1}(x) + eta * h_t( g_t(x) (elementwise*) x )
```

Implemented in `feature_gating.py` as 5 switchable modes — `none` (today's v4, bit-for-bit),
`global_static` / `global_dynamic` (one gate, shared by every learner, fixed vs.
input-dependent), and `per_learner_static` / `per_learner_dynamic` (each learner's own gate
over its own columns — closest to the original idea, since it's the thing hard feature
bagging cannot express).

Design detail that matters for the ablation: every dynamic gate's final layer is
zero-initialised, so `g_t(x)` starts as a **constant** ≈0.982 (`sigmoid(4.0)`) regardless
of x — dynamic and static modes start identically, almost fully open, and training is what
makes them diverge. `mode="none"` skips the gate module entirely (verified via `model.py`'s
existing parameter-count assertions, unchanged before/after this addition).

Notebook 10 tested all 5 modes on Covertype's mean-pool arm, holding notebook 09's exact
tuned config fixed and changing only the gate — so any difference is attributable to
gating, not to a luckier hyperparameter draw.

| Model | Test acc |
|---|---|
| XGBoost (published) | 0.9690 |
| Our XGBoost (tuned) | 0.9689 |
| MLP (published) | 0.9620 |
| v4 meanpool tuned + `global_dynamic` | 0.9577 ± 0.0010 |
| v4 meanpool tuned + `global_static` | 0.9573 ± 0.0014 |
| v4 meanpool tuned + `per_learner_static` | 0.9573 ± 0.0009 |
| **v4 meanpool — TUNED, no gate (notebook 09)** | **0.9569 ± 0.0020** |
| v4 meanpool tuned + `per_learner_dynamic` | 0.9563 ± 0.0010 |
| v4 mean-pool + PLR (untuned, notebook 08) | 0.9486 |

**Findings:**
* The `gate="none"` re-run reproduced notebook 09's result exactly (0.9569 ± 0.0020), so
  the comparison above is trustworthy — nothing else drifted between the two notebooks.
* **Gating does NOT clearly help, at default gate hyperparameters.** Every mode's delta
  from the ungated baseline is smaller than its own seed noise — best case
  `global_dynamic` at +0.0008, worst case `per_learner_dynamic` at −0.0005.
  `per_learner_dynamic` (closest to the predecessor document's original idea) is actually
  the weakest performer of the four gated variants.
* The best gate mode nudges the XGBoost gap from 0.0120 to 0.0112 — real but marginal, an
  order of magnitude smaller than what plain hyperparameter tuning bought in Step 8
  (+0.0083). No gate mode closes the gap to published plain MLP (0.9620) either.
* **Verdict:** the design itself worked as intended (near-identity-at-init confirmed, all 5
  modes trained without issue), but this specific mechanism, at its default settings, is
  not the fix for Covertype's remaining gap. Given how flat the 4 gated variants are
  relative to each other, tuning the gate's own knobs (`init_bias`, `hidden_dim`) is
  unlikely to change this conclusion — there is no partial signal here to amplify. Lower
  priority than extending Step 8's tuning to the other 7 datasets, or trying a different
  lever (piecewise-linear embeddings — see [Open threads](#open-threads--whats-next)).

---

# Appendix A: PLR embeddings, explained fully

**PLR = Periodic → Linear → ReLU.** Term from Gorishniy, Rubachev & Babenko (2022),
*"On Embeddings for Numerical Features in Tabular Deep Learning"* (arXiv:2203.05556).

For each input feature, instead of using the raw number directly, we compute:

```
1. PERIODIC:  turn the value into sine/cosine waves at LEARNED frequencies
                 v(x) = [sin(2π·c₁·x), ..., cos(2π·c₁·x), ...]
2. LINEAR:    shrink that down to a small vector (per-feature weights)
3. ReLU:      standard non-linearity
```

Every feature gets its **own private** set of frequencies and weights — feature 3 can
never influence feature 7's embedding. That's what breaks rotation-invariance: after a
rotation, "feature 3" is a different mixture of the originals, so these weights no
longer line up with it, and the model's output changes.

**Why periodic waves specifically?** This targets the "irregular functions" property —
trees represent sharp jumps easily (one split = one threshold), plain smooth neural nets
struggle to. A learned high-frequency sine wave can swing sharply around one particular
value of a feature, giving the network cheap access to threshold-like behaviour it
otherwise can't easily represent.

**Verified, not assumed:** `embeddings.py`'s own tests confirm (a) only the intended
feature's embedding moves when that feature changes, (b) rotated input produces a
genuinely different embedding, and (c) the periodic version reacts ~10x more sharply to
small input changes than the plain linear version.

---

# Open threads / what's next

* **Per-dataset tuning, beyond Covertype (top priority)** — notebook 09 only tuned
  mean-pool on Covertype (Step 8); feature gating (Step 9) turned out not to close the
  rest of that gap, so this is the most direct lever left unexplored. The other 7
  datasets, and the causal-attention arm, still use ONE fixed config designed on housing
  (notebook 06), while every published number we compare against is tuned per dataset.
* **Piecewise-linear feature encoding** — the same embeddings paper (Appendix A) also
  tested a binning-based alternative to PLR: fixed quantile (or decision-tree) bin edges
  instead of learned periodic frequencies. Cheap to add as a 3rd `embeddings.py` mode
  alongside `linear`/`periodic`, and directly tests whether PLR's win is about periodicity
  specifically or just about giving each feature its own nonlinear treatment — relevant
  since PLR already regressed once (YearPredictionMSD mean-pool, Step 7). Not yet built.
* **Feature gating: concluded, not a promising lever at default settings** — all 5 modes
  tested on Covertype (Step 9); every gated variant's delta from the ungated tuned
  baseline was smaller than its own seed noise. Deprioritized unless a future result
  suggests otherwise; `feature_gating.py`/`tuning.py`'s `gate_mode` override remain
  available if that changes.
* **Attention still unproven** — causal attention has never clearly beaten mean-pooling
  head-to-head at matched size; on the 5-dataset run (Step 7) mean-pool+PLR and causal
  attn+PLR trade wins depending on the dataset, with no consistent winner. Worth deciding
  whether to keep investigating it or treat mean-pool as the default arm.
* **More datasets** — now validated on California Housing, Covertype, Poker Hand, HIGGS
  (10k diagnostic samples), Higgs Small / Facebook Comments / Santander (full published
  scale, notebook 07), and Adult / California Housing / Jannis / Covertype /
  YearPredictionMSD (full published scale, notebook 08). ALOI, Helena, Epsilon, Yahoo,
  Microsoft are sitting in the already-downloaded `benchmarks_rtdl` archive on the GPU
  server if ever needed — just not extracted into `data/benchmarks_rtdl/` yet.
