
from dataclasses import dataclass
from typing import Dict, List, Any

# ============================================================================
# EXPERIMENT SELECTOR: Change this to switch between experiments
# Options: "lambda_div", "lambda_cka", "n_mlps"
# ============================================================================
EXPERIMENT = "n_mlps"
NUM_RUNS = 5
DATASET = "Churn_Modeling"  # Used for naming results files

# Where to write results (Colab-friendly)
# RESULTS_JSON = "/media/homes/hamza/results/Lambda_div_exp/tabjoint_search_results_titanic.json"
# BEST_JSON    = "/media/homes/hamza/results/Lambda_div_exp/tabjoint_best_titanic.json"
RESULTS_JSON = f"/media/homes/hamza/results/{EXPERIMENT}_averaged_exp/tabjoint_search_results_{DATASET}.json"
BEST_JSON    = f"/media/homes/hamza/results/{EXPERIMENT}_averaged_exp/tabjoint_best_{DATASET}.json"


# Primary metric name to maximize (our training loop returns 'best_val' and 'test')
PRIMARY_METRIC = "best_val"

# Keep the grid small first; expand later to avoid combinatorial blow-up
GRID: Dict[str, List[Any]] = {
    # MLP ensemble
    "n_mlps":            [2,4,6,8,10,12,14,16] if EXPERIMENT == "n_mlps" else [8],
    "emb_dim":           [128],
    "mlp_hidden_layers": [1],
    "mlp_dropout":       [0.1],
    # training
    "epochs":    [30],           
    "batch_size":[256],
    "lr_mlp":    [0.001],
    "lr_tr":     [1e-3],
    "lambda_cka":[0,0.05, 0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6] if EXPERIMENT == "lambda_cka" else [0.4],
    "lambda_div":[0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.2] if EXPERIMENT == "lambda_div" else [0],
    # Transformer head
    "d_model":        [128],
    "nhead":          [8],
    "num_layers":     [2],
    "dim_feedforward":[512],
    "dropout":        [0.1],
}
# Optional: static settings you don't sweep (you can move some grid keys here)
STATIC = {
    "weight_decay": 1e-5,
    "patience":     10,
    "feature_frac": 0.6,      # fraction of features per MLP token
    "resample_each_epoch": True,
    "seed": 123,
}
