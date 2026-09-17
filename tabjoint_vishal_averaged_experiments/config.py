
import numpy as np
import torch

# Reproducibility & device
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data defaults (you can change in main.py)
CSV_PATH = "/media/homes/hamza/Datasets/Churn_Modelling.csv"
TARGET_COL = "Exited"

# Training / model hyperparameters (kept same values as in your script)
N_MLPS = 15
EMB_DIM = 128
MLP_HIDDEN_LAYERS = 1
MLP_DROPOUT = 0.1

# Train config overrides for the run
TRAIN_KW = dict(epochs=50, batch_size=64, lr_mlp=5e-4, lr_tr=1e-3, lambda_div=0.0, lambda_cka=0.05)
