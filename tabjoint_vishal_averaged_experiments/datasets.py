
import numpy as np, torch
from torch.utils.data import Dataset

class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        if y.dtype.kind in {"f"}: self.y = torch.from_numpy(y.astype(np.float32))
        else:                      self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]
