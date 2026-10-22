# Graph Report - SRP  (2026-10-19)

## Corpus Check
- 223 files · ~140,122 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 33 file(s) not represented in the graph (top: .ipynb 26, (none) 6, .tex 1)

## Summary
- 885 nodes · 1699 edges · 46 communities (35 shown, 11 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 122 edges (avg confidence: 0.88)
- Token cost: 360,975 input · 0 output

## Community Hubs (Navigation)
- Analysis Scripts & Plotting Utilities
- Diversity & CKA Measurement (V1)
- Benchmark Data Loading & Distribution Plots
- Presentation Findings & Dataset Concepts
- Raw Dataset Download & Preprocessing
- Preprocessor Comparison Pipeline
- V1 Models & Dataset Wrappers
- Optuna Hyperparameter Tuning
- Grid Search CLI & Label Mapping
- Training Loop & Evaluation Contract
- Feature-Bagged Weak Learner Ensemble
- Attention & Embedding Module Self-Tests
- Arm Assembly & Config Dataclasses
- Feature Gating Module
- Attention Ablation Charts
- Architecture Diagrams (V1 vs Current)
- Published-Split Benchmark Loaders
- Rotation Invariance Diagnostic
- SRPModel & PlainMLP Baseline
- Diversity Loss Design History
- Causal Self-Attention Stage
- NCL Lambda Sweep Results
- Weighted Multi-Step Loss Schemes
- Weak Learner MLP & Penalties
- Pipeline & PLR Flow Diagrams
- Embedding Modes & Tuning Protocol
- Sequential Residual Predictor Model
- Negative Correlation Learning Loss
- Expert Collapse Evidence
- Per-Feature Embedding Layer
- SRPModel V2 & Feature Importance
- Aggregator Design Alternatives
- Project Lineage & Baselines
- Architecture Pipeline Stages
- Experimental Protocol Rules
- Results Convention & SOTA Baselines
- GPU Cluster Reproduction Workflow
- Dataset Archive Conventions
- V2 Causal Transformer Block
- Deep Supervision Training Loop
- University Branding Assets
- CatBoost Baseline
- Server Sync Script

## God Nodes (most connected - your core abstractions)
1. `run()` - 19 edges
2. `Bagged Expert Ensembles (BEE)` - 19 edges
3. `SRPModel` - 17 edges
4. `run_grid_search()` - 16 edges
5. `WeakLearnerEnsemble` - 15 edges
6. `TabularPreprocessor` - 14 edges
7. `TabularTask` - 13 edges
8. `WeakLearnerEnsemble` - 13 edges
9. `Preprocessor` - 11 edges
10. `ArrayDataset` - 11 edges

## Surprising Connections (you probably didn't know these)
- `Negative Correlation Learning (ncl.py)` --semantically_similar_to--> `Similarity-matrix diversity loss`  [INFERRED] [semantically similar]
  v4/README.md → q-dev-chat-2026-05-10.md
- `λ>1 NCL breakdown boundary` --semantically_similar_to--> `Diversity-weight ablation (sweet spot 0.05)`  [INFERRED] [semantically similar]
  v4/README.md → q-dev-chat-2026-05-10.md
- `PMA / set-attention aggregator (proposed)` --semantically_similar_to--> `CLS-token removal alternatives (mean pool / last token / weighted sum)`  [INFERRED] [semantically similar]
  v4/README.md → q-dev-chat-2026-05-10.md
- `SRPModel` --uses--> `WeakLearnerEnsemble`  [INFERRED]
  srp_model.py → v4/weak_learners.py
- `SRPModel` --uses--> `WeakLearnerEnsemble`  [INFERRED]
  srp_model_v2.py → v4/weak_learners.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Aggregator alternatives compared under matched capacity** — v4_readme_mean_pool, v4_readme_causal_attention, v4_readme_bidirectional_attention, v4_readme_pool_level_output, v4_readme_pma_set_attention, q_dev_chat_2026_05_10_cls_token [INFERRED 0.85]
- **Competing diversity mechanisms for the weak-learner ensemble** — v4_readme_feature_bagging, v4_readme_ncl, v4_readme_feature_gating, q_dev_chat_2026_05_10_diversity_loss, v4_v1_readme_lambda_div_zero [INFERRED 0.85]
- **Per-feature embedding modes (embeddings.py switchable family)** — v4_readme_embedding_mode_none, v4_readme_embedding_mode_linear, v4_readme_plr_embeddings, v4_readme_piecewise_linear_embeddings, v4_readme_rotation_invariance_diagnosis [EXTRACTED 1.00]
- **BEE Pipeline: Features to Prediction** — main_plr_embedding, main_mlp_expert, main_feature_bagging, main_aggregation, main_prediction_head [EXTRACTED 1.00]
- **Mechanisms for Enforcing Expert Diversity** — main_feature_bagging, main_negative_correlation_learning, main_cosine_similarity_diagnostic, main_expert_diversity [INFERRED 0.85]
- **State-of-the-Art Tabular Baselines Compared Against** — main_xgboost, main_catboost, main_ft_transformer, main_tabr, main_tabm [EXTRACTED 1.00]
- **Attention Components Add No Benefit Over Capacity-Matched Averaging** — figures_attention_simple, figures_attention_waterfall, figures_attention_simple_mean_pooling, figures_attention_simple_causal_attention, figures_attention_waterfall_capacity_dominates [INFERRED 0.85]
- **Mechanisms Guarding Against Expert Collapse** — figures_feature_bagging_collapse_feature_bagging, figures_ncl_california_housing_ncl_penalty, figures_feature_bagging_collapse_expert_collapse, figures_feature_bagging_collapse_expert_similarity_metric [INFERRED 0.85]
- **Rotation Argument: Tree Axis Dependence vs MLP Invariance** — figures_rotation_explained, figures_rotation_invariance, figures_rotation_explained_axis_aligned_splits, figures_rotation_invariance_model_rotation_invariance, figures_rotation_explained_rotation_of_feature_space [INFERRED 0.90]
- **Forward pass: features to embedding to bagged experts to aggregation to prediction** — figures_architecture_diagram_input_features, figures_architecture_diagram_plr_embedding, figures_architecture_diagram_feature_bagged_experts, figures_architecture_diagram_mean_pool, figures_architecture_diagram_prediction_head, figures_architecture_diagram_final_prediction [EXTRACTED 1.00]
- **PLR embedding stack: scalar to periodic to linear to ReLU to vector** — figures_plr_flow_scalar_feature, figures_plr_flow_periodic_layer, figures_plr_flow_linear_layer, figures_plr_flow_relu, figures_plr_flow_feature_vector [EXTRACTED 1.00]
- **Evidence that expert diversity is maintained without collapse and that NCL strengthens it** — figures_feature_bagging_collapse_feature_bagging, figures_feature_bagging_collapse_expert_collapse, figures_feature_bagging_collapse_finding, figures_ncl_california_housing_ncl_penalty, figures_ncl_california_housing_best_result [INFERRED 0.85]
- **Current Pipeline: Experts to Causal Attention to Weighted Multi-Head Loss** — ours_mlp_expert_bank, ours_causal_self_attention, ours_per_expert_weighted_loss, weights_weight_scheme [INFERRED 0.85]
- **Predecessor Pipeline: Experts plus CLS and Diversity Tokens under Full Attention** — ours_mlp_expert_bank, prev_full_self_attention, prev_cls_token_readout, prev1_diversity_regularization_token [EXTRACTED 1.00]
- **Architecture Evolution Evaluated in the Five-Dataset Benchmark** — prev_architecture_diagram, prev1_architecture_diagram, ours_architecture_diagram, output_benchmark_chart [INFERRED 0.75]

## Communities (46 total, 11 thin omitted)

### Community 0 - "Analysis Scripts & Plotting Utilities"
Cohesion: 0.05
Nodes (35): extract_lambda_cka_and_test_accuracy(), load_results(), main(), plot_lambda_cka_vs_accuracy(), extract_lambda_cka_and_test_accuracy(), load_results(), main(), plot_lambda_cka_vs_accuracy() (+27 more)

### Community 1 - "Diversity & CKA Measurement (V1)"
Cohesion: 0.06
Nodes (29): cosine_similarity_matrix(), diversity_loss(), pearson_correlation_matrix(), cka_token_matrix(), collect_tokens_sampled(), _linear_cka(), plot_matrix(), ArrayDataset (+21 more)

### Community 2 - "Benchmark Data Loading & Distribution Plots"
Cohesion: 0.07
Nodes (17): plot_all_preprocessors(), BenchmarkData, extract(), higher_is_better(), load_benchmark(), _load_part(), load_task_type(), published() (+9 more)

### Community 3 - "Presentation Findings & Dataset Concepts"
Cohesion: 0.06
Nodes (37): Adult Dataset, Expert Aggregation Stage, Bagged Expert Ensembles (BEE), California Housing Dataset, Causal Self-Attention Aggregation, Chen & Guestrin 2016, XGBoost: A Scalable Tree Boosting System, Off-Diagonal Cosine Similarity Diagnostic, Covertype Dataset (+29 more)

### Community 4 - "Raw Dataset Download & Preprocessing"
Cohesion: 0.07
Nodes (13): _fit_preprocessor(), transform(), _load_higgs(), _load_poker(), load_raw(), make_task(), TabularTask, add_noise_features() (+5 more)

### Community 5 - "Preprocessor Comparison Pipeline"
Cohesion: 0.08
Nodes (7): compare_preprocessors(), compute_distribution_stats(), evaluate_preprocessing_impact(), plot_preprocessing_comparison(), Preprocessor, RankGaussTransformer, TabularPreprocessor

### Community 6 - "V1 Models & Dataset Wrappers"
Cohesion: 0.08
Nodes (16): ArrayDataset, cka_offdiag_mean(), diversity_loss(), EmbeddingsTransformer, FeatureSubspaceCfg, JointMLPTransformer, make_feature_subsets(), make_loss() (+8 more)

### Community 7 - "Optuna Hyperparameter Tuning"
Cohesion: 0.12
Nodes (14): create_study(), load_study(), make_xgb(), optuna_worker(), objective(), _pruner(), run_config(), _sampler() (+6 more)

### Community 8 - "Grid Search CLI & Label Mapping"
Cohesion: 0.13
Nodes (7): cli_main(), run(), _set_visible_devices(), apply_mapping(), build_label_mapping(), PrepConfig, TabularPreprocessor

### Community 9 - "Training Loop & Evaluation Contract"
Cohesion: 0.13
Nodes (10): classification_metrics(), _eval_loss(), evaluate(), get_device(), _pred(), predict(), regression_metrics(), to_tensors() (+2 more)

### Community 11 - "Attention & Embedding Module Self-Tests"
Cohesion: 0.14
Nodes (4): _check_causality_by_gradient(), _check_causality_by_perturbation(), _head_param_count(), head_width_for_budget()

### Community 12 - "Arm Assembly & Config Dataclasses"
Cohesion: 0.11
Nodes (8): AttentionConfig, EmbeddingConfig, make(), make_arm(), PredictionHead, SRPConfig, build_model(), _embedding_config()

### Community 13 - "Feature Gating Module"
Cohesion: 0.10
Nodes (6): FeatureGate, FeatureGateConfig, _make_gate_net(), _gate_config(), make_feature_subsets(), WeakLearnerConfig

### Community 14 - "Attention Ablation Charts"
Cohesion: 0.15
Nodes (10): Attention vs Averaging Experts (Test R2 Bar Chart), Mean-Pool, bigger head (38k params), Test R2 0.8104, Mean-Pool, small head (12k params), Test R2 0.7889, Test R2 metric (higher is better), Accuracy Attribution Waterfall (Test R2), + attention (no order) costs -0.0032 Test R2, + capacity contributes +0.0215 Test R2, Causal attention + deep supervision final Test R2 0.8096 (+2 more)

### Community 15 - "Architecture Diagrams (V1 vs Current)"
Cohesion: 0.16
Nodes (18): Current Architecture Diagram (Causal-Attention Expert Ensemble), Causal (Masked) Self-Attention over Expert Tokens, Bank of N MLP Experts Emitting Tokens z_1..z_N, Per-Expert Predictions with Weighted Total Loss sum w_i L_i, Five-Dataset Benchmark Dot Chart (v4 vs XGBoost vs Published), v4 Variants: Causal Attention vs Mean-Pool, with/without PLR, XGBoost (ours/published) and Published MLP/FT-Transformer Reference Points, Predecessor Architecture Variant with Diversity Token (+10 more)

### Community 16 - "Published-Split Benchmark Loaders"
Cohesion: 0.14
Nodes (9): extract(), higher_is_better(), _load(), load_benchmark(), load_task_type(), published(), _published(), published_table() (+1 more)

### Community 17 - "Rotation Invariance Diagnostic"
Cohesion: 0.17
Nodes (10): Why Rotation Hurts Trees (Two-Panel Explainer), 45-degree rotated feature space (u, v blends of Age and Income), Accuracy Change Under Feature Rotation (XGBoost vs Our Model), Delta test accuracy under feature rotation (0 = no effect), Covertype Dataset, Rotation benchmark datasets: Covertype, Poker Hand, HIGGS, HIGGS Dataset, Our model without embedding under rotation (+0.002, -0.003, -0.003) (+2 more)

### Community 18 - "SRPModel & PlainMLP Baseline"
Cohesion: 0.16
Nodes (3): PlainMLP, SRPModel, step_weights()

### Community 19 - "Diversity Loss Design History"
Cohesion: 0.18
Nodes (14): Cosine similarity diversity measure, Similarity-matrix diversity loss, Diversity-weight ablation (sweet spot 0.05), Diversity-weight scheduling (warm-up vs annealing), Pearson correlation diversity measure, Original SRP design brief (weak-learner MLPs → attention), SRP design chat transcript (2026-05-12), detach_mean (stop-gradient ensemble mean) (+6 more)

### Community 20 - "Causal Self-Attention Stage"
Cohesion: 0.17
Nodes (3): build_causal_mask(), CausalAttentionStage, CausalSelfAttentionBlock

### Community 21 - "NCL Lambda Sweep Results"
Cohesion: 0.18
Nodes (7): XGBoost baseline Test R2 0.8471, NCL Penalty Strength Sweep on California Housing (Test RMSE), Best result: lambda=1.0 gives RMSE 0.4039, Model Without NCL (RMSE 0.4202), Test RMSE metric (lower is better), Tuned XGBoost Baseline (RMSE 0.4328), Tuned XGBoost baseline RMSE 0.4328

### Community 22 - "Weighted Multi-Step Loss Schemes"
Cohesion: 0.21
Nodes (4): _compute_step_weights(), _step_loss(), train(), WeightScheme

### Community 24 - "Pipeline & PLR Flow Diagrams"
Cohesion: 0.18
Nodes (12): Model Architecture Diagram, Input Features (tabular columns: Age, Income, Location), PLR Embedding (per feature), Pipeline Flow Diagram, Input Features (pipeline stage), Optional Feature Embedding (None / Linear / PLR), PLR Embedding Flow Diagram, Small Per-Feature Vector (+4 more)

### Community 25 - "Embedding Modes & Tuning Protocol"
Cohesion: 0.18
Nodes (10): SRP — Sequential Residual Predictor, Embedding mode `linear`, Gorishniy, Rubachev & Babenko (2022), On Embeddings for Numerical Features in Tabular Deep Learning (arXiv:2203.05556), Grinsztajn et al. (2022), Why do tree-based models still outperform deep learning on typical tabular data?, Grinsztajn perturbation diagnostics (noise / rotation / label smoothing), Two-phase Optuna tuning protocol, Piecewise-linear embeddings (PLE), PLR embeddings (Periodic → Linear → ReLU) (+2 more)

### Community 27 - "Negative Correlation Learning Loss"
Cohesion: 0.20
Nodes (4): BaggedHeads, ncl_loss(), prediction_correlation(), train()

### Community 28 - "Expert Collapse Evidence"
Cohesion: 0.35
Nodes (4): Expert Similarity With and Without Feature Bagging, California Housing Dataset, Higgs Small Dataset, Expert Similarity Chart (Slide Variant)

### Community 31 - "Aggregator Design Alternatives"
Cohesion: 0.36
Nodes (8): CLS-token removal alternatives (mean pool / last token / weighted sum), [CLS] token aggregation, Causal attention aggregator, Causal-order permutation test (notebook 14), Mean-pool aggregator, PMA / set-attention aggregator (proposed), pool_level="output" (predict-then-average), Weak learner / expert

### Community 32 - "Project Lineage & Baselines"
Cohesion: 0.31
Nodes (8): tabjoint_vishal_averaged_experiments (earlier experiment line), Bidirectional attention aggregator, Gorishniy, Rubachev, Khrulkov & Babenko (2021), Revisiting Deep Learning Models for Tabular Data (arXiv:2106.11959, FT-Transformer), SRP v4 README (restructured, TabR-style project guide), PlainMLP floor reference, V1 — predecessors' architecture (bidirectional attention + [CLS] + CKA diversity penalty), SRP v4 — from-scratch rebuild, V1 vendored verbatim (models.py / training.py / datasets.py)

### Community 33 - "Architecture Pipeline Stages"
Cohesion: 0.36
Nodes (8): Causal Self-Attention Aggregation, Feature-Bagged MLP Experts (K experts, ~70% random feature subset each), Final Prediction Output, Mean Pool Aggregation, Prediction Head (Regression / Classification), Aggregation (Mean Pooling / Attention), K Small MLP Experts (each sees a feature subset), Prediction (pipeline output)

### Community 34 - "Experimental Protocol Rules"
Cohesion: 0.29
Nodes (5): Arm (one full configuration under test), Embedding mode `none`, Feature gating (adaptive per-learner feature relevancy), Identical expert init across arms (make_arm forked RNG), Per-module [PASS]/[FAIL] self-tests

### Community 35 - "Results Convention & SOTA Baselines"
Cohesion: 0.33
Nodes (5): backfill_results.py (documented-source backfill), Covertype gap (v4's unresolved weak spot), report.json-per-run results convention (results.py), TabM (Gorishniy et al., 2025), TabR (Gorishniy et al., 2023, arXiv:2307.14338)

### Community 36 - "GPU Cluster Reproduction Workflow"
Cohesion: 0.33
Nodes (5): Chunked GPU evaluation, SRP_* environment-variable overrides, GPU cluster workflow (edit → sync → run → copy back), Timestamped progress logs under runs/, sync_to_server.sh safeguards

### Community 37 - "Dataset Archive Conventions"
Cohesion: 0.60
Nodes (4): num-embeddings data archive listing (12 dataset folders), READY marker file convention, RTDL/FT-Transformer data archive listing (11 dataset folders), Two dataset tiers (diagnostic samples vs exact published splits)

### Community 39 - "Deep Supervision Training Loop"
Cohesion: 0.83
Nodes (3): _deep_supervision_weights(), _step_loss(), train()

### Community 41 - "University Branding Assets"
Cohesion: 0.67
Nodes (3): University of Hildesheim Logo Lockup (Card Asset), University of Hildesheim Logo Lockup (Square Placeholder), Stiftung Universitaet Hildesheim 2003 Circular Seal

## Ambiguous Edges - Review These
- `One fixed, never-per-dataset-tuned config` → `RTDL/FT-Transformer data archive listing (11 dataset folders)`  [AMBIGUOUS]
  v4/data/raw/rtdl_full_listing.txt · relation: conceptually_related_to
- `Prediction Head` → `Output-Space Pooling`  [AMBIGUOUS]
  main.pdf · relation: shares_data_with
- `CLS Token Readout Producing a Single Prediction` → `v4 Variants: Causal Attention vs Mean-Pool, with/without PLR`  [AMBIGUOUS]
  output.png · relation: conceptually_related_to

## Knowledge Gaps
- **41 isolated node(s):** `Embedding mode `none``, `Embedding mode `linear``, `Grinsztajn et al. (2022), Why do tree-based models still outperform deep learning on typical tabular data?`, `Liu & Yao (1999), Negative Correlation Learning`, `TabM (Gorishniy et al., 2025)` (+36 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 315 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `WeakLearnerEnsemble` connect `Feature-Bagged Weak Learner Ensemble` to `Attention & Embedding Module Self-Tests`, `Feature Gating Module`, `SRPModel & PlainMLP Baseline`, `Sequential Residual Predictor Model`, `SRPModel V2 & Feature Importance`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **What connects `Embedding mode `none``, `Embedding mode `linear``, `Grinsztajn et al. (2022), Why do tree-based models still outperform deep learning on typical tabular data?` to the rest of the system?**
  _41 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Analysis Scripts & Plotting Utilities` be split into smaller, more focused modules?**
  _Cohesion score 0.050616050616050616 - nodes in this community are weakly interconnected._
- **Why does `SRPModel` connect `SRPModel & PlainMLP Baseline` to `Optuna Hyperparameter Tuning`, `Feature-Bagged Weak Learner Ensemble`, `Attention & Embedding Module Self-Tests`, `Arm Assembly & Config Dataclasses`, `Feature Gating Module`, `Causal Self-Attention Stage`, `Per-Feature Embedding Layer`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **Should `Diversity & CKA Measurement (V1)` be split into smaller, more focused modules?**
  _Cohesion score 0.05860805860805861 - nodes in this community are weakly interconnected._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `One fixed, never-per-dataset-tuned config` and `RTDL/FT-Transformer data archive listing (11 dataset folders)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Prediction Head` and `Output-Space Pooling`?**
  _Edge tagged AMBIGUOUS (relation: shares_data_with) - confidence is low._