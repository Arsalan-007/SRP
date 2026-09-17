# SRP v4 — Project Guide

**What this project is asking:** can an ensemble of small neural networks ("weak learners"),
combined in a learned way, close the gap to Gradient Boosted Decision Trees (GBDTs like
XGBoost) on tabular data? This is v4 — a from-scratch rebuild after V1/V2/V3 were reviewed
and found to have untested claims. Every idea in v4 is built as an isolated, switchable
piece and measured against a baseline before being trusted.

**Guiding rule for this whole project:** never assume a new idea helps. Build it, run it
next to the old version under an identical protocol, and let the numbers say yes or no.
This file exists so you (or anyone else) can re-read the project's logic without having to
re-derive it from the conversation history.

---

## 1. The architecture, in plain terms

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

### Key terms, defined once

| Term | Meaning |
|---|---|
| **Weak learner / expert** | One small MLP in the ensemble. K of them exist (e.g. K=16). |
| **Feature bagging** | Each expert only sees a random ~70% subset of the input columns. This is the project's diversity mechanism — it stops all experts from learning the same thing, without needing an artificial "diversity penalty" (which V1 needed and which we proved is unnecessary). |
| **Mean-pool** | Aggregator that just averages all K experts' outputs. The simplest possible combination — used as the control everything else must beat. |
| **Causal attention** | Aggregator using masked self-attention: expert *i* can "see" experts *1..i* but not later ones — inspired by the idea that later experts should build on earlier ones, like stages in boosting. |
| **Embedding mode `none`** | No feature embedding stage — features go straight into the experts as raw numbers. This was v4's behaviour through notebook 05. |
| **Embedding mode `linear`** | Each feature gets its own small `Linear → ReLU` before mixing. Tests whether *separating* features (without anything fancy) helps. |
| **Embedding mode `periodic` (a.k.a. "PLR")** | Each feature gets `Periodic (sin/cos waves) → Linear → ReLU`. Lets the model represent sharp, threshold-like behaviour in a single feature — see §4 below. This is the current best-performing option. |
| **Arm** | One full configuration under test (e.g. "causal attn + PLR"). Notebooks compare several arms side by side under identical training. |
| **V1** | Your predecessors' original architecture (`tabjoint_vishal_averaged_experiments/`, vendored unchanged in `v1/`): bidirectional attention + a `[CLS]` token + a CKA diversity penalty. Kept as a permanent comparison point. |
| **PlainMLP** | One ordinary MLP, no expert structure at all. The "floor" reference — if we can't beat this, the whole ensemble idea isn't earning its complexity. |

---

## 2. File map

```
v4/
├── weak_learners.py    The K small expert MLPs + feature bagging.
├── attention.py         The causal (and bidirectional) attention aggregator.
├── embeddings.py        Per-feature embeddings: none / linear / periodic(PLR). See §4.
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
├── tuning.py              Optuna hyperparameter search machinery (not yet pointed at the
│                         benchmarks_rtdl datasets — see §6, "per-dataset tuning").
├── v1/                    Your predecessors' code, copied UNCHANGED (do not edit).
├── v1_runner.py           Wrapper to run V1 under the same protocol as v4.
└── data/                  Downloaded/cached datasets (not synced to git or the server's
                           raw archive — see .gitignore-style excludes in sync script).

Notebooks (run in order; each one is a self-contained experiment with its findings
printed at the end, computed from the actual run — never hand-typed):
  01_baseline_benchmark.ipynb     step 1: weak learners + mean-pool vs XGBoost (housing)
  02_attention_ablation.ipynb     step 2: does causal attention beat mean-pool? (housing)
  03_optuna_num_learners.ipynb    how many weak learners (K) do we actually need?
  04_grinsztajn_diagnostics.ipynb WHY do trees still win? (3 tests, 3 new datasets)
  05_v1_published_benchmarks.ipynb  (superseded by 07 — kept for its data-loading notes)
  06_feature_embeddings.ipynb     does the embeddings.py fix work? (4 datasets)
  07_v4_vs_v1_gpu.ipynb           v4 vs V1 head-to-head on published benchmarks (GPU)
  08_v4_vs_literature.ipynb       v4 vs published MLP/ResNet/NODE/FT-Transformer/XGBoost
                                  on 5 standard benchmarks, incl. 2 large-scale (GPU)
```

---

## 3. The story so far, in order

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
with an optional periodic ("PLR") component for sharp thresholds. See §4 for full detail.
**Finding, tested on 4 datasets (10k-row samples):** the fix worked on both counts —
scores went up on almost every dataset/arm combination, AND (the mechanism check) the
model became measurably sensitive to rotation for the first time, proving we'd actually
fixed the diagnosed problem rather than just gotten lucky.

### Step 6 — Confirm it at full scale + compare to V1 and the literature (GPU)
Ran V1 and 4 v4 variants (mean-pool / causal-attn, each with/without PLR) on 3 datasets
with their **exact published splits** (Higgs Small 98k rows, Facebook Comments 197k rows,
Santander 200k rows), next to a tuned XGBoost and the paper's published numbers for 33
other models.

**Results:**

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

**Where we stand vs. the literature:** near parity with published GBDTs on Santander,
close on Higgs, still behind on Facebook Comments — and this is all *without* per-dataset
tuning, unlike every published number.

### Step 7 — v4 vs. the wider literature (5 standard benchmarks, incl. 2 large-scale)

Ran v4's 4 arms (mean-pool / causal-attn, each with/without PLR) + PlainMLP + our own
tuned XGBoost on **5 more datasets with their exact published splits** — this time from
Gorishniy, Rubachev, Khrulkov & Babenko (2021), *Revisiting Deep Learning Models for
Tabular Data*, NeurIPS 2021 (arXiv:2106.11959, the FT-Transformer paper) — next to that
paper's own published numbers for MLP, ResNet, NODE, and FT-Transformer.

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

**Sanity check passed:** our own tuned XGBoost reproduces the published XGBoost within
~0.002-0.05 on every dataset, so the comparison above is trustworthy.

**Findings (this is the strongest result in the project so far):**
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
* No std is reported in this paper's tables (unlike notebook 07's paper), so "published"
  numbers here are single reference points, not distributions.

---

## 4. PLR embeddings, explained fully

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

## 5. How to run things

See `sync_to_server.sh` (parent SRP/ folder) for laptop→GPU-server syncing, and the
comments at the top of `07_v4_vs_v1_gpu.ipynb` for environment-variable overrides
(`SRP_SUBSAMPLE`, `SRP_SEEDS`, `SRP_JOBS`, etc.) that let you smoke-test before a long run.

**Golden rule:** always copy a finished notebook's results back from the server BEFORE
syncing new code to it — `--inplace` execution overwrites the notebook file itself.

---

## 6. Open threads / what's next

* **Per-dataset tuning (top priority)** — every v4 result so far uses ONE fixed
  hyperparameter config designed on housing data (notebook 06), reused unchanged on all
  8 datasets since. Every published number we compare against was tuned per dataset.
  `tuning.py`'s Optuna machinery already exists (used in notebook 03 for housing) but has
  not yet been pointed at the `benchmarks.py`/`benchmarks_rtdl.py` datasets or extended
  to search the embedding mode/width. This is the most direct lever left to improve the
  standing numbers, and the natural next thing to close Covertype's gap specifically.
* **Feature gating** (per-learner, learned relevance weighting) — proposed by a second
  research document, targets the one Grinsztajn property we haven't fixed yet:
  robustness to useless/noisy features. Not yet implemented. Covertype — the one dataset
  where v4 now clearly underperforms, including vs. published plain MLP — is exactly the
  kind of large, many-feature, multiclass setting this is meant to help with.
* **Attention still unproven** — causal attention has never clearly beaten mean-pooling
  head-to-head at matched size; on the 5-dataset run (step 7) mean-pool+PLR and causal
  attn+PLR trade wins depending on the dataset, with no consistent winner. Worth deciding
  whether to keep investigating it or treat mean-pool as the default arm.
* **More datasets** — now validated on California Housing, Covertype, Poker Hand, HIGGS
  (10k diagnostic samples), Higgs Small / Facebook Comments / Santander (full published
  scale, notebook 07), and Adult / California Housing / Jannis / Covertype /
  YearPredictionMSD (full published scale, notebook 08). ALOI, Helena, Epsilon, Yahoo,
  Microsoft are sitting in the already-downloaded `benchmarks_rtdl` archive on the GPU
  server if ever needed — just not extracted into `data/benchmarks_rtdl/` yet.
