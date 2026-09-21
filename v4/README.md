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
├── embeddings.py        Per-feature embeddings: none / linear / periodic(PLR) /
│                         piecewise_linear (quantile-binned, fixed edges). See §4.
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
│                         search (§ notebook 09) and opt-in gate_mode (§9) / ncl_lambda
│                         (§10) overrides for run_config()/build_model().
├── feature_gating.py      Adaptive per-learner feature relevancy: g_t(x) in
│                         F_t(x) = F_{t-1}(x) + eta*h_t(g_t(x)*x). 5 switchable modes
│                         (none / global|per_learner x static|dynamic). See §3 Step 9.
├── ncl.py                 Negative Correlation Learning (Liu & Yao, 1999): a per-learner
│                         penalty rewarding deviation from the ensemble mean. Requires
│                         model.py's pool_level="output" (predict-then-average). §3 Step 10.
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
  09_tune_covertype.ipynb         is Covertype's gap just untuned hyperparameters? (GPU)
  10_feature_gating_covertype.ipynb  does feature gating help on top of the tuned config?
                                  answer: no, not at default settings (GPU)
  11_ncl_meanpool.ipynb           does Negative Correlation Learning help mean-pool + PLR?
                                  answer: yes, beats our XGBoost + published FT-Transformer
                                  (California Housing) — runs entirely on CPU, ~5 min
  12_ncl_larger_benchmarks.ipynb  does NCL generalize? answer: mixed — helps on Facebook
                                  Comments only, not Higgs Small or Santander (GPU)
  13_piecewise_linear_embeddings.ipynb  is PLR's win about periodicity, or just per-feature
                                  nonlinearity? answer: neither — YE/mean-pool doesn't want
                                  ANY per-feature nonlinear embedding (GPU)
  14_causal_order_permutation.ipynb  does causal ordering mean anything? answer: no — both
                                  attention arms overfit to an arbitrary learner order
                                  (CPU, ~15 min)
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

### Step 8 — Is Covertype's gap just untuned hyperparameters?

Every v4 number above used ONE fixed config (from notebook 06, designed on housing) reused
unchanged everywhere — while every published number is tuned per dataset. Before building
anything new, notebook 09 asked the cheap question first: extended `tuning.py`'s Optuna search
(already used for K in notebook 03) to also search the embedding mode/width, and pointed it at
Covertype's cheapest, already-best arm (mean-pool). Two-phase protocol: 24-trial parallel search
(small epoch budget, validation loss), then the winning config re-trained on 3 fresh seeds at
notebook 08's full epoch budget, reported on test.

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
* Tuning genuinely helped (+0.0083 over the best untuned config, beyond the 0.0020 seed noise)
  but closed only **41% of the gap to XGBoost** (0.0203 → 0.0120) and still trails the published
  plain MLP, the weakest deep baseline in the paper.
* The search independently rediscovered that PLR helps — the best trial picked
  `embedding_mode=periodic` (15/24 completed trials did), cross-validating Step 5/7 via a
  completely different process. fANOVA ranked it only 5th of 11 params, though: capacity/
  regularization knobs (`num_learners`, `dropout`, `lr`, `feature_frac`) mattered more here.
* **Verdict, computed by the notebook itself:** "MOSTLY NOT hyperparameters — tuning recovered
  under half the gap. This is real evidence for trying an architectural fix (e.g. feature
  gating) here." This is what motivated Step 9.

### Step 9 — Feature gating: adaptive per-learner feature relevancy

Feature bagging (`weak_learners.py`) gives each learner a fixed, HARD column subset, frozen at
construction — good for diversity, but a learner stuck with mostly-noisy columns has no way to
lean on its one useful column more. A second predecessor research document proposed a
boosting-style fix: a learned, optionally input-dependent SOFT gate reweighting each learner's
view of the data before its first layer,

```
F_t(x) = F_{t-1}(x) + eta * h_t( g_t(x) (elementwise*) x )
```

Implemented in `feature_gating.py` as 5 switchable modes — `none` (today's v4, bit-for-bit),
`global_static` / `global_dynamic` (one gate, shared by every learner, fixed vs. input-dependent),
and `per_learner_static` / `per_learner_dynamic` (each learner's own gate over its own columns —
closest to the original idea, since it's the thing hard feature bagging cannot express).

Design detail that matters for the ablation: every dynamic gate's final layer is
zero-initialised, so `g_t(x)` starts as a **constant** ≈0.982 (`sigmoid(4.0)`) regardless of x —
dynamic and static modes start identically, almost fully open, and training is what makes them
diverge. `mode="none"` skips the gate module entirely (verified via `model.py`'s existing
parameter-count assertions, unchanged before/after this addition).

Notebook 10 tests all 5 modes on Covertype's mean-pool arm, holding notebook 09's exact tuned
config fixed and changing only the gate — so any difference is attributable to gating, not to a
luckier hyperparameter draw.

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
* The `gate="none"` re-run reproduced notebook 09's result exactly (0.9569 ± 0.0020), so the
  comparison above is trustworthy — nothing else drifted between the two notebooks.
* **Gating does NOT clearly help, at default gate hyperparameters.** Every mode's delta from the
  ungated baseline is smaller than its own seed noise — best case `global_dynamic` at +0.0008,
  worst case `per_learner_dynamic` at −0.0005. `per_learner_dynamic` (closest to the predecessor
  document's original idea) is actually the weakest performer of the four gated variants.
* The best gate mode nudges the XGBoost gap from 0.0120 to 0.0112 — real but marginal, an order
  of magnitude smaller than what plain hyperparameter tuning bought in Step 8 (+0.0083). No gate
  mode closes the gap to published plain MLP (0.9620) either.
* **Verdict:** the design itself worked as intended (near-identity-at-init confirmed, all 5
  modes trained without issue), but this specific mechanism, at its default settings, is not
  the fix for Covertype's remaining gap. Given how flat the 4 gated variants are relative to
  each other, tuning the gate's own knobs (`init_bias`, `hidden_dim`) is unlikely to change this
  conclusion — there is no partial signal here to amplify. Lower priority than extending Step 8's
  tuning to the other 7 datasets, or trying a different lever (e.g. piecewise-linear embeddings,
  §6).

### Step 10 — Negative Correlation Learning (Liu & Yao, 1999) on mean-pool + PLR

Every weak learner trains independently, minimizing only its own error — nothing in the LOSS
pushes learners to disagree usefully; feature bagging is the only diversity mechanism. NCL
(Liu & Yao, 1999) adds a per-learner penalty that rewards deviating from the current ensemble
mean: if learner A overpredicts, learner B is rewarded for underpredicting, pulling the
ensemble toward the target even when no individual learner is close.

Implemented in `ncl.py`. Two things had to be resolved before it could be tested honestly:
1. **Mean-pool has no per-learner predictions today** — it averages EMBEDDINGS, then runs one
   shared head once. NCL needs each learner's own scalar output. `model.py` gained
   `pool_level="output"` (apply the shared head to every learner, then average the K scalars)
   as a prerequisite — a real architecture change, independent of NCL. `ncl_lambda=0` under
   `pool_level="output"` isolates that confound: it's plain independent training under the new
   pooling point, so it can be compared against today's `pool_level="embedding"` baseline to
   see whether the pooling POINT alone matters, before crediting anything to the penalty.
2. **`detach_mean` (whether the ensemble mean is a stop-gradient target) turns out not to
   matter under this codebase's training loop** — proven both analytically and empirically in
   `ncl.py`'s own self-test: because `training.py` always does ONE joint `.backward()` over the
   summed loss of all K learners, the per-learner "detach" factor exactly cancels against
   cross-terms from the other K-1 learners' penalties. Caught before running any real
   experiment, not assumed.

`ncl.py`'s own toy self-test (K=6 linear heads, each restricted to half the input features —
mirroring real feature bagging) showed a clean result: ensemble MSE fell smoothly as λ went
0→1.0 (3.74→0.097, recovering the full-information optimum despite each head seeing half the
features), then broke down sharply past λ≈1.0 — independently reproducing the literature's
λ∈[0,1] guidance empirically rather than from memory.

Notebook 11 tested this for real: California Housing (published split), the exact fixed
architecture from notebook 06/08 (`num_learners=16, embed_dim=32`, PLR embedding), λ swept
over `{0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5}`, 3 seeds each, full epoch budget — small enough
to run entirely on CPU, no GPU cluster round trip needed (27 runs, 4.3 min).

| Model | Test RMSE |
|---|---|
| **v4 meanpool_ncl, λ=1.0** | **0.4039 ± 0.0039** |
| v4 mean-pool + PLR (embedding-level, today's baseline) | 0.4202 ± 0.0048 |
| Our XGBoost (tuned) | 0.4328 |
| v4 meanpool_ncl, λ=0 (pooling-point control) | 0.4501 ± 0.0081 |
| Published FT-Transformer | 0.459 |
| Published MLP | 0.499 |

**Findings:**
* **The best λ beats our tuned XGBoost, published FT-Transformer, and published MLP** — the
  strongest v4 regression result in the project so far, and the config wasn't even jointly
  tuned with λ (it's notebook 06's fixed architecture, only λ varied).
* **Moving the pooling point alone (λ=0) actually HURTS** — 0.4501 vs. 0.4202, a real
  regression beyond seed noise, not a neutral confound. Predict-then-average is worse than
  average-then-predict on its own; the NCL penalty has to overcome a real architectural cost
  to win, and at λ=1.0 it does so by a wide margin (0.4501 → 0.4039).
* **Instability confirmed exactly where the toy self-test predicted**: λ=1.5 collapses to RMSE
  1.03 (vs. 0.40 at λ=1.0) — the same sharp λ>1 breakdown boundary found analytically and
  empirically before touching real data.
* **Caveats:** one dataset, one fixed (not jointly-tuned) architecture, regression only so far.
  Multiclass is not yet supported (the penalty is only defined for a scalar prediction), so
  Covertype — the dataset with the actual unresolved gap (Steps 8-9) — can't be tested until
  that's built. This result is genuinely promising, not yet a confirmed general win.

**Generalization check (notebook 12): mixed — helps on 1 of 3 datasets.** Tested against
Higgs Small, Facebook Comments, and Santander (Step 6's datasets), same protocol as notebook
11. **Important correction applied after the run**, same issue as Step 11: `benchmarks.py`
also standardises regression targets, so Facebook Comments' raw `run_config` RMSE needed
rescaling by `y_std` (now fixed in the notebook's source for future runs) — the *direction*
of every "beats baseline" comparison was still correct even before the fix (scale-invariant),
but a "beats XGBoost" claim for Facebook Comments in the raw output was an artifact of the
scale mismatch and is corrected below.

| Model | Higgs Small (acc↑) | Facebook Comments (RMSE↓) | Santander (acc↑) |
|---|---|---|---|
| embedding-level baseline (today's v4) | 0.7245 ± 0.0018 | 5.826 ± 0.027 | 0.9218 ± 0.0013 |
| meanpool_ncl, best λ | 0.7208 ± 0.0039 (λ=0.0) | **5.673 ± 0.050** (λ=0.8) | 0.9203 ± 0.0010 (λ=0.0) |
| our XGBoost (tuned) | 0.7255 | 5.4016 | 0.9235 |
| published CatBoost | 0.7260 | 5.3240 | 0.9230 |
| published MLP-PLR | 0.7280 | 5.5250 | 0.9240 |

**Findings:**
* **Higgs Small and Santander: no improvement beyond seed noise** — best λ in both cases is
  0.0 (i.e. the search couldn't find a λ>0 that helps), meaning the pooling-point-only
  control already is the "best," and even that doesn't beat the embedding-level baseline.
* **Facebook Comments: genuinely helps** — λ=0.8 improves over the baseline by 0.153 RMSE,
  well beyond seed noise (±0.03-0.05), and beats V1 (6.038) and notebook 07's wide-head PLR
  number (5.793). It does **not**, however, beat our tuned XGBoost (5.402), published
  CatBoost (5.324), or published MLP-PLR (5.525) — a real but partial win, not a repeat of
  California Housing's result.
* **Verdict: 1 of 3 datasets.** Unlike Step 10's clean California Housing win, this doesn't
  generalize uniformly — consistent with everything else in this project needing per-dataset
  tuning (Step 8) rather than one fixed recipe. Facebook Comments is a regression task with a
  skewed, long-tailed target (comment counts) — worth checking whether NCL specifically suits
  that kind of target, or whether this is just favorable seed variance on one dataset.

### Step 11 — Piecewise-linear embeddings: is PLR's win about periodicity, or just nonlinearity?

Step 7 found PLR regressed on YearPredictionMSD mean-pool — the one clean counter-example to
"PLR always helps" in the project. Implemented the paper's other embedding family,
piecewise-linear encoding ("PLE"): fixed, training-data quantile bin edges (no learned
frequencies) — `embeddings.py`'s new `mode="piecewise_linear"`,
`v_t(x) = clip((x - edge[t]) / (edge[t+1] - edge[t]), 0, 1)`. Self-tested (bin-edge
correctness, feature-independence, rotation-sensitivity — same checks "periodic" already had).

Notebook 13 compared `{none, linear, periodic, piecewise_linear}` on mean-pool, notebook 08's
exact fixed config, on YearPredictionMSD (the regression case) + California Housing (sanity
check). **Important correction applied after the run**: `run_config`'s RMSE is on
`benchmarks_rtdl.py`'s *standardized* target scale; the table below rescales by each
dataset's `y_std` (matching notebook 08's own `score_from_raw` convention) so it's directly
comparable to every other number in this document — this rescaling is now built into the
notebook itself for future runs.

| Model | YearPredictionMSD RMSE | California Housing RMSE |
|---|---|---|
| v4 mean-pool, no embedding | 8.798 ± 0.022 | 0.546 ± 0.005 |
| v4 mean-pool + linear | 8.818 ± 0.034 | 0.496 ± 0.007 |
| v4 mean-pool + periodic (PLR) | 8.883 ± 0.011 | **0.483 ± 0.005** |
| v4 mean-pool + piecewise_linear | 8.864 ± 0.054 | 0.490 ± 0.003 |
| *(reference)* v4 mean-pool, no emb. (notebook 08, wide head) | 8.829 | 0.540 |
| *(reference)* v4 mean-pool + PLR (notebook 08, wide head) | 8.862 | 0.504 |

**Findings:**
* **Both periodic AND piecewise_linear regress vs. no-embedding on YearPredictionMSD**
  (+0.085 and +0.066 respectively, both beyond seed noise). This is evidence YE/mean-pool
  genuinely doesn't want ANY per-feature nonlinear embedding — **not** a periodic-specific
  failure. The original hypothesis (periodicity itself was the problem) is not supported;
  piecewise_linear regresses almost as much as periodic does here.
* On California Housing, both embeddings clearly help (periodic best at 0.483, piecewise_linear
  close behind at 0.490) — consistent with Step 5/7's general finding, just not informative
  about the YE question specifically.
* **Verdict:** the YE regression looks like a property of that dataset/arm combination, not a
  fixable artifact of periodic's specific mechanism. Doesn't (yet) motivate switching PLR's
  default to piecewise_linear, though the module is now available for future comparisons.

### Step 12 — Does causal ordering mean anything, or is it an arbitrary bottleneck?

Direct test of the open thread above: `weak_learners.py`'s K experts are trained JOINTLY (one
shared loss), not sequentially on frozen residuals like real boosting — so the order they sit
in (fixed at construction, unrelated to anything about the data) is arbitrary. Causal masking
imposes a directional "expert i sees only 0..i-1" structure on top of that arbitrary order,
plus a learned per-slot position embedding both `causal` and `bidirectional` have
(`AttentionConfig.use_step_embedding=True`).

**The test:** train `meanpool`/`bidirectional`/`causal` once each (3 seeds, California
Housing, notebook 08's fixed config), then evaluate each trained model again with the K
tokens randomly shuffled before entering the aggregator — no retraining, same weights, only
which learner sits in which slot changes. Runs entirely on CPU (~15 min).

| Arm | Baseline (trained order) | Shuffled (10 perms, mean) | Degradation |
|---|---|---|---|
| mean-pool *(reference — provably order-invariant)* | 0.4212 | 0.4212 ± 0.004 *(pure seed noise)* | — |
| bidirectional | 0.4145 | 0.4820 ± 0.038 | **+0.068** |
| causal | 0.4223 | 0.5246 ± 0.043 | **+0.102** |

**Findings:**
* **Built-in sanity check passed:** mean-pool's "shuffled" variance (0.004) exactly matches
  pure seed-to-seed noise, since averaging is mathematically order-invariant — confirms the
  measurement itself is trustworthy before trusting what it says about attention.
* **Both attention arms degrade sharply under a random eval-time shuffle** — 10-16x more
  variance than mean-pool's noise floor, with a huge range (causal swings from 0.433 to 0.613
  depending on which arbitrary permutation hits it). Real slot-specific structure was learned,
  tied to a learner-to-slot assignment that was never meaningful to begin with.
* **Causal degrades more than bidirectional** (+0.102 vs. +0.068). Both share the same
  learned step embedding, so the ~0.035 gap isolates the causal MASK's own extra contribution
  to this overfitting, on top of the positional bias both arms already carry.
* **Verdict:** causal ordering is measuring something real — but real overfitting to an
  arbitrary choice, not a meaningful boosting-like sequence. Direct support for building a
  permutation-invariant aggregator (PMA / set-attention: a learned query cross-attends over
  the K expert embeddings as an unordered set) as the next architectural step, rather than
  continuing to invest in the causal arm as currently designed.

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

* **Negative Correlation Learning: mixed, not a clean generalization** — notebook 11's
  California Housing win (beats our XGBoost + published FT-Transformer, §3 Step 10) did not
  repeat on Higgs Small or Santander (notebook 12); only Facebook Comments improved, and only
  partially (beats V1, doesn't beat XGBoost/CatBoost). Best remaining lever: a joint λ ×
  architecture search per dataset (every result so far used notebook 06's fixed config with
  only λ varied) before concluding whether this is dataset-dependent or needs per-dataset
  tuning like everything else (Step 8). Multiclass support (for Covertype) is lower priority
  now that generalization looks uncertain rather than assumed.
* **Per-dataset tuning, beyond Covertype** — notebook 09 only tuned mean-pool on Covertype
  (§3 Step 8); feature gating (§3 Step 9) turned out not to close the rest of that gap. The
  other 7 datasets, and the causal-attention arm, still use ONE fixed config designed on
  housing (notebook 06), while every published number we compare against is tuned per
  dataset.
* **Piecewise-linear feature encoding: concluded, not the YE fix** — implemented in
  `embeddings.py` (§3 Step 11); notebook 13 found piecewise_linear regresses on YE almost as
  much as periodic does, so periodic's regression there isn't a periodicity-specific failure —
  the dataset/arm combination just doesn't want per-feature nonlinear embeddings. Module
  remains available (`mode="piecewise_linear"`) for future comparisons on other datasets.
* **Feature gating: concluded, not a promising lever at default settings** — all 5 modes
  tested on Covertype (§3 Step 9); every gated variant's delta from the ungated tuned
  baseline was smaller than its own seed noise. Deprioritized unless a future result
  suggests otherwise; `feature_gating.py`/`tuning.py`'s `gate_mode` override remain
  available if that changes.
* **Build PMA / set-attention pooling** — notebook 14 (§3 Step 12) found why causal attention
  has never clearly beaten mean-pooling: both attention arms overfit to their arbitrary,
  fixed-at-construction learner order (10-16x more eval-time variance than mean-pool's pure
  seed noise under a random shuffle). A permutation-invariant aggregator — a learned query
  cross-attending over the K expert embeddings as an unordered set — keeps "attention" in the
  architecture without the unjustified causal ordering. Not yet built.
* **More datasets** — now validated on California Housing, Covertype, Poker Hand, HIGGS
  (10k diagnostic samples), Higgs Small / Facebook Comments / Santander (full published
  scale, notebook 07), and Adult / California Housing / Jannis / Covertype /
  YearPredictionMSD (full published scale, notebook 08). ALOI, Helena, Epsilon, Yahoo,
  Microsoft are sitting in the already-downloaded `benchmarks_rtdl` archive on the GPU
  server if ever needed — just not extracted into `data/benchmarks_rtdl/` yet.
