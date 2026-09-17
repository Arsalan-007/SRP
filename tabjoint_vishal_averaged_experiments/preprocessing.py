
import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler

# ------------------------- config -------------------------
@dataclass
class PrepConfig:
    # numeric
    impute_num: str = "mean"                 # "mean" | "median" | "most_frequent"
    scale: Optional[str] = "standard"        # None | "standard" | "minmax"

    # categorical
    impute_cat: str = "most_frequent"        # used after rare/OTHER mapping
    cat_encoding: str = "onehot"             # "onehot" | "ordinal"
    cardinality_threshold: int = 50            # drop categoricals with > this unique values
    rare_threshold_pct: float = 0.05           # map levels <5% to "__OTHER__"
    other_token: str = "__OTHER__"
    missing_token: str = "__MISSING__"

    # misc
    noise_std: float = 0.0                     # optional numeric noise after scale (for robustness)

# ------------------------- preprocessor -------------------------
class TabularPreprocessor:
    """
    - Drops high-cardinality categorical columns (> cardinality_threshold)
    - Collapses rare categories (< rare_threshold_pct of rows) to OTHER (per column)
    - Encodes categoricals via OneHot or Ordinal
    """
    def __init__(self, cfg: PrepConfig):
        self.cfg = cfg
        self.ct: Optional[ColumnTransformer] = None

        # learned attributes after fit()
        self.num_cols: List[str] = []
        self.cat_cols_all: List[str] = []
        self.cat_cols_kept: List[str] = []
        self.cat_cols_dropped: List[str] = []
        self.cat_keep_sets: Dict[str, set] = {}     # per col -> set of kept categories (incl. OTHER & MISSING)
        self.ordinal_categories_: Optional[List[List[str]]] = None
        self.feature_names_: Optional[List[str]] = None

    # ---------- internal helpers ----------
    def _scaler(self):
        if self.cfg.scale is None:       return "passthrough"
        if self.cfg.scale == "standard": return StandardScaler()
        if self.cfg.scale == "minmax":   return MinMaxScaler()
        raise ValueError("scale must be None|standard|minmax")

    def _onehot(self):
        # compat: sklearn >=1.2 uses sparse_output
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    def _ordinal(self):
        # we’ll supply categories explicitly; handle_unknown not hit because we map to OTHER
        return OrdinalEncoder(categories=self.ordinal_categories_)

    def _make_cat_small_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply rare/OTHER mapping & missing token to kept categorical columns."""
        df = pd.DataFrame(index=X.index)
        for c in self.cat_cols_kept:
            col = X[c].astype("object").where(~X[c].isna(), self.cfg.missing_token)
            keep = self.cat_keep_sets[c]
            col = col.where(col.isin(keep), self.cfg.other_token)
            df[c] = col
        return df

    # ---------- public API ----------
    def fit(self, X: pd.DataFrame):
        # split columns by dtype
        self.cat_cols_all = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        self.num_cols = X.columns.difference(self.cat_cols_all).tolist()

        # decide which categorical columns to keep/drop by cardinality
        self.cat_cols_kept, self.cat_cols_dropped = [], []
        for c in self.cat_cols_all:
            # count uniques ignoring NaN
            nuniq = int(X[c].nunique(dropna=True))
            if nuniq > self.cfg.cardinality_threshold:
                self.cat_cols_dropped.append(c)
            else:
                self.cat_cols_kept.append(c)

        # build rare/OTHER maps for kept categoricals
        self.cat_keep_sets = {}
        N = len(X)
        for c in self.cat_cols_kept:
            series = X[c].astype("object").where(~X[c].isna(), self.cfg.missing_token)
            vc = series.value_counts(normalize=True, dropna=False)
            keep_levels = set(vc[vc >= self.cfg.rare_threshold_pct].index.astype(str))
            keep_levels.add(self.cfg.missing_token)
            keep_levels.add(self.cfg.other_token)
            self.cat_keep_sets[c] = keep_levels

        # if ordinal, define category order per column (stable list)
        if self.cfg.cat_encoding == "ordinal":
            self.ordinal_categories_ = []
            for c in self.cat_cols_kept:
                series = X[c].astype("object").where(~X[c].isna(), self.cfg.missing_token)
                vc = series.value_counts(dropna=False)
                levels = [self.cfg.missing_token] + [v for v in vc.index.astype(str) if v in self.cat_keep_sets[c] and v not in (self.cfg.missing_token, self.cfg.other_token)] + [self.cfg.other_token]
                seen, ordered = set(), []
                for v in levels:
                    if v not in seen:
                        ordered.append(v); seen.add(v)
                self.ordinal_categories_.append(ordered)

        # numeric pipeline
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=self.cfg.impute_num)),
            ("scaler",  self._scaler())
        ]) if self.num_cols else "drop"

        # categorical (kept) pipeline
        cat_transformer = None
        if self.cat_cols_kept:
            if self.cfg.cat_encoding == "onehot":
                cat_transformer = Pipeline([
                    ("rare_other", "passthrough"),
                    ("imputer", SimpleImputer(strategy=self.cfg.impute_cat, fill_value=self.cfg.missing_token)),
                    ("ohe", self._onehot()),
                ])
            elif self.cfg.cat_encoding == "ordinal":
                cat_transformer = Pipeline([
                    ("rare_other", "passthrough"),
                    ("imputer", SimpleImputer(strategy=self.cfg.impute_cat, fill_value=self.cfg.missing_token)),
                    ("ord", self._ordinal()),
                ])
            else:
                raise ValueError("cat_encoding must be 'onehot' or 'ordinal'")

        transformers = []
        if self.num_cols:
            transformers.append(("num", num_pipe, self.num_cols))
        if self.cat_cols_kept:
            transformers.append(("cat", cat_transformer, self.cat_cols_kept))

        self.ct = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
        X_cat_mapped = self._make_cat_small_df(X)
        X_fit = X.copy()
        for c in self.cat_cols_kept:
            X_fit[c] = X_cat_mapped[c]

        self.ct.fit(X_fit)

        names = []
        if self.num_cols:
            names += self.num_cols
        if self.cat_cols_kept:
            if self.cfg.cat_encoding == "onehot":
                ohe = self.ct.named_transformers_["cat"]["ohe"]
                names += ohe.get_feature_names_out(self.cat_cols_kept).tolist()
            else:
                names += [f"{c}__ord" for c in self.cat_cols_kept]
        self.feature_names_ = names
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        assert self.ct is not None, "Call fit() first."
        X_cat_mapped = self._make_cat_small_df(X)
        X_use = X.copy()
        for c in self.cat_cols_kept:
            X_use[c] = X_cat_mapped[c]
        Xt = self.ct.transform(X_use)
        Xt = np.asarray(Xt, dtype=np.float32)

        if self.cfg.noise_std and self.num_cols:
            n_num = len(self.num_cols)
            Xt[:, :n_num] += np.random.normal(0, self.cfg.noise_std, size=(Xt.shape[0], n_num)).astype(np.float32)
        return Xt

    def info(self) -> dict:
        return {
            "num_cols": self.num_cols,
            "cat_cols_all": self.cat_cols_all,
            "cat_cols_kept": self.cat_cols_kept,
            "cat_cols_dropped": self.cat_cols_dropped,
            "rare_threshold_pct": self.cfg.rare_threshold_pct,
            "cat_encoding": self.cfg.cat_encoding,
        }

def build_label_mapping(y_train: np.ndarray):
    # regression?
    if (y_train.dtype.kind in {"f"} and np.unique(y_train).size > 10):
        return {"task":"regression","out_dim":1,"mapping":None}
    classes = np.unique(y_train)
    if classes.size == 2:
        mapping = {classes[0]:0, classes[1]:1}
        return {"task":"binary","out_dim":1,"mapping":mapping}
    mapping = {c:i for i,c in enumerate(classes)}
    return {"task":"multiclass","out_dim":classes.size,"mapping":mapping}

def apply_mapping(y_raw: np.ndarray, info: dict):
    if info["task"] == "regression":
        return y_raw.astype(np.float32).reshape(-1,1)
    m = info["mapping"]
    y = np.vectorize(m.get)(y_raw).astype(np.int64)
    return y.reshape(-1)
