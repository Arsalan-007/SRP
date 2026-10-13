"""
backfill_results.py  —  SRP v4

One-time (re-runnable) script: transcribes every result already documented in README.md
Steps 6-12 into results/<DATASET>/<arm>/<tag>/report.json via results.py, all tagged
source="documented" or source="published" — no training happens here, nothing is
freshly computed. See results.py's docstring for why the source field matters.

Each entry is aggregated to mean +/- std across seeds (matching how the README itself
reports everything), not one file per individual seed — dozens of files, not hundreds.
Run directly: `python3 backfill_results.py` — safe to re-run, overwrites in place.
"""

from results import DEFAULT_RESULTS_DIR, save_report

# ---------------------------------------------------------------------------
# Dataset metadata: key -> (label, metric name, higher_is_better)
# ---------------------------------------------------------------------------
DATASETS = {
    "HI": ("Higgs Small", "test_acc", True),
    "FB": ("Facebook Comments", "test_rmse", False),
    "SA": ("Santander", "test_acc", True),
    "CA": ("California Housing", "test_rmse", False),
    "AD": ("Adult", "test_acc", True),
    "JA": ("Jannis", "test_acc", True),
    "CO": ("Covertype", "test_acc", True),
    "YE": ("YearPredictionMSD", "test_rmse", False),
}


def _entry(dataset, arm, tag, mean_std, source, source_ref, n_parameters=None):
    """mean_std: a single float (no std reported), or a (mean, std) pair — stored
    EXACTLY as given, never re-derived from a reconstructed array. (An earlier version
    of this function faked a 2-point seed array as [mean-std, mean+std] to carry the
    std through — that recomputes std(ddof=1) on 2 points spaced 2*std apart, which is
    std*sqrt(2), not std. Caught before this ever ran.)"""
    label, metric, hib = DATASETS[dataset]
    if isinstance(mean_std, (tuple, list)) and len(mean_std) == 2:
        mean = float(mean_std[0])
        std = None if mean_std[1] is None else float(mean_std[1])
    else:
        mean, std = float(mean_std), None
    d = DEFAULT_RESULTS_DIR / dataset / arm / tag
    save_report(
        d, dataset=dataset, label=label, arm=arm, tag=tag, seed=None,
        source=source, source_ref=source_ref,
        config={}, n_parameters=n_parameters,
        metrics={"test": {metric: mean, f"{metric}_std": std}},
        champion_metric=metric, champion_value=mean, higher_is_better=hib,
        time_seconds=None, checkpoint=None,
    )


def main():
    # =========================================================================
    # Step 6 (notebook 07) — Higgs Small / Facebook Comments / Santander,
    # exact published splits. FB values already in ORIGINAL units (RMSE).
    # =========================================================================
    step6 = {
        "V1":                    dict(HI=(0.7178, 0.0009), FB=(6.0379, 0.0756), SA=(0.9140, 0.0002), n=416_385),
        "meanpool":              dict(HI=(0.7215, 0.0006), FB=(5.9443, 0.0149), SA=(0.9147, 0.0003), n=41_229),
        "meanpool_plr":          dict(HI=(0.7245, 0.0019), FB=(5.7932, 0.0835), SA=(0.9227, 0.0010), n=84_909),
        "causal":                dict(HI=(0.7172, 0.0029), FB=(6.0738, 0.0366), SA=(0.9131, 0.0005), n=41_281),
        "causal_plr":            dict(HI=(0.7218, 0.0014), FB=(5.8929, 0.0599), SA=(0.9225, 0.0009), n=84_961),
        "plainmlp":              dict(HI=(0.7175, 0.0012), FB=(6.0114, 0.0583), SA=(0.9112, 0.0008), n=36_865),
        "xgboost_ours_tuned":    dict(HI=(0.7255, 0.0002), FB=(5.4016, 0.0109), SA=(0.9235, 0.0001)),
    }
    for arm, d in step6.items():
        n = d.pop("n", None)
        for key, mean_std in d.items():
            _entry(key, arm, "step6-fixed", mean_std, "documented", "README Step 6 / notebook 07", n)

    step6_published = {
        "xgboost_published": dict(HI=0.724, FB=5.383, SA=0.918),
        "catboost_published": dict(HI=0.726, FB=5.324, SA=0.923),
        "mlp_published": dict(HI=0.720, FB=5.686, SA=0.912),
        "mlp_plr_published": dict(HI=0.728, FB=5.525, SA=0.924),
    }
    for arm, d in step6_published.items():
        for key, val in d.items():
            _entry(key, arm, "published", val, "published", "README Step 6 / notebook 07 (paper)")

    # =========================================================================
    # Step 7 (notebook 08) — California Housing / Adult / Jannis / Covertype /
    # YearPredictionMSD, exact published splits. No std reported for this table.
    # =========================================================================
    step7 = {
        "meanpool":     dict(CA=0.5404, AD=0.8589, JA=0.7195, CO=0.9324, YE=8.8293, n=37_600),
        "meanpool_plr": dict(CA=0.5044, AD=0.8594, JA=0.7225, CO=0.9486, YE=8.8621, n=50_600),
        "causal":       dict(CA=0.4985, AD=0.8567, JA=0.7097, CO=0.9100, YE=8.8472, n=37_700),
        "causal_plr":   dict(CA=0.4834, AD=0.8608, JA=0.7195, CO=0.9336, YE=8.8425, n=50_700),
        "plainmlp":     dict(CA=0.5117, AD=0.8543, JA=0.7127, CO=0.9203, YE=8.8754, n=34_300),
        "xgboost_ours_tuned": dict(CA=0.4328, AD=0.8739, JA=0.7233, CO=0.9689, YE=8.8647),
    }
    for arm, d in step7.items():
        n = d.pop("n", None)
        for key, val in d.items():
            _entry(key, arm, "step7-fixed", val, "documented", "README Step 7 / notebook 08", n)

    step7_published = {
        "mlp_published":            dict(CA=0.499, AD=0.852, JA=0.719, CO=0.962, YE=8.853),
        "resnet_published":         dict(CA=0.486, AD=0.854, JA=0.728, CO=0.964, YE=8.846),
        "node_published":           dict(CA=0.464, AD=0.858, JA=0.727, CO=0.958, YE=8.784),
        "ft_transformer_published": dict(CA=0.459, AD=0.859, JA=0.732, CO=0.970, YE=8.855),
        "xgboost_published":        dict(CA=0.431, AD=0.872, JA=0.724, CO=0.969, YE=8.819),
    }
    for arm, d in step7_published.items():
        for key, val in d.items():
            _entry(key, arm, "published", val, "published", "README Step 7 / notebook 08 (paper)")

    # =========================================================================
    # Step 8 (notebook 09) — Covertype, per-dataset Optuna tuning.
    # Real per-seed test_acc from notebook 09's confirm phase: 0.9551, 0.9591, 0.9564
    # -> mean 0.9569, std(ddof=1) 0.0020 (matches the README's reported 0.9569 +/- 0.0020).
    # =========================================================================
    _entry("CO", "meanpool_tuned", "notebook09-optuna", (0.9569, 0.0020),
          "documented", "README Step 8 / notebook 09")

    # =========================================================================
    # Step 9 (notebook 10) — Covertype, 5 feature-gating modes (on top of the
    # notebook 09 tuned config).
    # =========================================================================
    step9 = {
        "gate_global_dynamic":      (0.9577, 0.0010),
        "gate_global_static":       (0.9573, 0.0014),
        "gate_per_learner_static":  (0.9573, 0.0009),
        "gate_per_learner_dynamic": (0.9563, 0.0010),
    }
    for arm, mean_std in step9.items():
        _entry("CO", arm, "notebook10", mean_std, "documented", "README Step 9 / notebook 10")

    # =========================================================================
    # Step 10 (notebook 11) — California Housing, full NCL lambda sweep.
    # =========================================================================
    ca_ncl = {
        "meanpool":              [0.4202, 0.0048],   # embedding-level baseline
        "meanpool_ncl_l0.0":     [0.4501, 0.0081],
        "meanpool_ncl_l0.1":     [0.4465, 0.0076],
        "meanpool_ncl_l0.2":     [0.4435, 0.0079],
        "meanpool_ncl_l0.4":     [0.4358, 0.0077],
        "meanpool_ncl_l0.6":     [0.4273, 0.0022],
        "meanpool_ncl_l0.8":     [0.4189, 0.0026],
        "meanpool_ncl_l1.0":     [0.4039, 0.0039],   # champion
        "meanpool_ncl_l1.5":     [1.0335, 0.0811],
    }
    for arm, mean_std in ca_ncl.items():
        _entry("CA", arm, "notebook11", mean_std, "documented", "README Step 10 / notebook 11")

    # =========================================================================
    # Step 10 extension (notebook 12) — does NCL generalize? Higgs / Facebook /
    # Santander. FB rescaled to original RMSE units (see README Step 10 correction).
    # =========================================================================
    ncl_generalization = {
        "HI": {"meanpool": (0.7245, 0.0018), "meanpool_ncl_best": (0.7208, 0.0039)},
        "SA": {"meanpool": (0.9218, 0.0013), "meanpool_ncl_best": (0.9203, 0.0010)},
        "FB": {"meanpool": (5.826, 0.027), "meanpool_ncl_l0.8": (5.673, 0.050)},
    }
    for ds, arms in ncl_generalization.items():
        for arm, mean_std in arms.items():
            _entry(ds, arm, "notebook12", mean_std,
                  "documented", "README Step 10 extension / notebook 12")

    # =========================================================================
    # Step 11 (notebook 13) — YearPredictionMSD / California Housing, embedding
    # mode comparison (none / linear / periodic / piecewise_linear).
    # =========================================================================
    step11 = {
        "YE": {
            "meanpool_none":             (8.798, 0.022),
            "meanpool_linear":           (8.818, 0.034),
            "meanpool_periodic":         (8.883, 0.011),
            "meanpool_piecewise_linear": (8.864, 0.054),
        },
        "CA": {
            "meanpool_none":             (0.546, 0.005),
            "meanpool_linear":           (0.496, 0.007),
            "meanpool_periodic":         (0.483, 0.005),
            "meanpool_piecewise_linear": (0.490, 0.003),
        },
    }
    for ds, arms in step11.items():
        for arm, mean_std in arms.items():
            _entry(ds, arm, "notebook13", mean_std, "documented", "README Step 11 / notebook 13")

    # =========================================================================
    # Step 12 (notebook 14) — California Housing, causal-order permutation test.
    # "trained order" values are single runs (no seed spread reported); the
    # shuffled columns are 10-permutation means +/- std, already an aggregate.
    # =========================================================================
    step12 = {
        "meanpool_baseline":       (0.4212, 0.0043),   # pure seed noise, no shuffle possible
        "bidirectional_trained":   (0.4145, None),
        "bidirectional_shuffled":  (0.4820, 0.0378),
        "causal_trained":          (0.4223, None),
        "causal_shuffled":         (0.5246, 0.0432),
    }
    for arm, mean_std in step12.items():
        _entry("CA", arm, "notebook14", mean_std, "documented", "README Step 12 / notebook 14")

    df = __import__("results").load_reports()
    print(f"backfill complete: {len(df)} report.json entries under {DEFAULT_RESULTS_DIR}")
    print(f"datasets covered: {sorted(df['dataset'].unique())}")
    print(f"sources: {df['source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
