# V1 — "MLPs with Attention (V1)", verbatim

Unmodified copies of `models.py`, `training.py` and `datasets.py` from
`../../tabjoint_vishal_averaged_experiments/`, so that V1 travels with the v4 folder
(one `rsync`) and is run exactly as it was written.

Known behaviours of this code, kept as-is on purpose (this folder is the reference, not a fix):

* `training.py` line 40: the `lambda_div` term is multiplied by `0.0`, so it never affects the loss.
* `models.py` `resample_feature_subsets()` re-draws subsets with the same fixed seed, so the
  "resampled" subsets are identical every epoch — the subsets are effectively fixed.
* `train_joint` reports validation/test metrics as the MEAN OF PER-BATCH metrics. For RMSE this is
  not the true RMSE, so the v4 benchmark notebook recomputes every metric on the full split.
