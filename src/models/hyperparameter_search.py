import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.features.build_features import get_feature_pipelines, get_preprocessor
from src import config

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Search space for alpha (ridge), ngram_max, vocab_size
SPACE = [
    Real(1e-8, 1e+1, prior="log-uniform", name="alpha"),
    Integer(1, min(64, config.N_GRAM_MAX), name="ngram_max"),
    Integer(1000, 30000, name="vocab_size"),
]


def _load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _cv_score(df: pd.DataFrame, alpha: float, ngram_max: int, vocab_size: int, n_splits: int = 5, random_state: int = 42) -> float:
    """Compute average Spearman correlation across targets and folds.

    Returns negative average (to minimize).
    """
    vectorizer_vh, vectorizer_vl, encoder_ohe = get_feature_pipelines(
        ngram_range=(1, ngram_max), vocab_size=vocab_size
    )

    # Fit on all available rows for columns (faster than per fold and matches prod transformers)
    df_fit = df.dropna(subset=[config.VH_SEQUENCE_COL, config.VL_SEQUENCE_COL]).copy()
    encoder_ohe.fit(df_fit[[config.HC_SUBTYPE_COL]])
    vectorizer_vh.fit(df_fit[config.VH_SEQUENCE_COL])
    vectorizer_vl.fit(df_fit[config.VL_SEQUENCE_COL])

    preprocessor = get_preprocessor(vectorizer_vh, vectorizer_vl, encoder_ohe)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=alpha, **{k: v for k, v in config.RIDGE_MODEL_PARAMS.items() if k != "alpha"})),
    ])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    corrs = []
    for target in config.TARGET_PROPERTIES:
        df_t = df.dropna(subset=[target]).copy()
        if df_t.empty:
            continue
        y = df_t[target].values
        for train_idx, val_idx in kf.split(df_t):
            X_tr, X_va = df_t.iloc[train_idx], df_t.iloc[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
            # Spearman on non-NaN
            mask = ~np.isnan(y_va)
            if mask.any():
                corr = spearmanr(preds[mask], y_va[mask]).correlation
                if np.isfinite(corr):
                    corrs.append(corr)
    if not corrs:
        return 0.0
    return -float(np.mean(corrs))


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search with scikit-optimize")
    parser.add_argument("--data-path", type=str, default=str(config.DATA_FILE), help="CSV with training data")
    parser.add_argument("--n-iter", type=int, default=50, help="Number of optimization calls")
    parser.add_argument("--output-csv", type=str, default=str(config.BASE_DIR.parent / "hyperparameter_search_results.csv"), help="CSV to save trials")
    parser.add_argument("--output-best", type=str, default=str(config.ARTEFACTS_DIR / "best_params.json"), help="Where to save best params JSON")
    parser.add_argument("--skopt-dump", type=str, default=str(config.ARTEFACTS_DIR / "skopt_result.joblib"), help="Path to dump skopt result")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = _load_data(Path(args.data_path))

    @use_named_args(SPACE)
    def objective(**p):
        alpha = float(p["alpha"])  # ensure json serializable
        ngram_max = int(p["ngram_max"])
        vocab_size = int(p["vocab_size"])
        print(f"Trying alpha={alpha:.3e}, ngram_max={ngram_max}, vocab_size={vocab_size}")
        score = _cv_score(df, alpha=alpha, ngram_max=ngram_max, vocab_size=vocab_size)
        print(f" -> objective (neg mean spearman) = {score:.6f}")
        return score

    # Checkpointing: save results after each iteration
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    best_path = Path(args.output_best)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    def _save_result(res):
        try:
            trials = pd.DataFrame(res.x_iters, columns=["alpha", "ngram_max", "vocab_size"])  # type: ignore
            trials["objective_value"] = res.func_vals
            trials.to_csv(out_csv, index=False)
            if getattr(res, "x", None) is not None and getattr(res, "fun", None) is not None:
                best = {
                    "alpha": float(res.x[0]),
                    "ngram_max": int(res.x[1]),
                    "vocab_size": int(res.x[2]),
                    "best_objective": float(res.fun),
                }
                with open(best_path, "w") as f:
                    json.dump(best, f, indent=2)
        except Exception as e:
            print(f"[checkpoint] Failed to save checkpoint: {e}")

    # Mutable holder for last result for Ctrl+C handling
    last_res_holder = {"res": None}

    class CheckpointCallback:
        def __call__(self, res):
            last_res_holder["res"] = res
            _save_result(res)

    try:
        result = gp_minimize(
            objective,
            SPACE,
            n_calls=args.n_iter,
            random_state=args.random_state,
            n_initial_points=min(10, args.n_iter),
            acq_func="EI",
            callback=[CheckpointCallback()],
        )
    except KeyboardInterrupt:
        print("\n[interrupt] Caught Ctrl+C. Saving best-so-far and exiting...")
        if last_res_holder["res"] is not None:
            _save_result(last_res_holder["res"])
        # If we have something saved, exit gracefully
        return

    # Save per-iteration results
    trials = pd.DataFrame(result.x_iters, columns=["alpha", "ngram_max", "vocab_size"])  # type: ignore
    trials["objective_value"] = result.func_vals
    trials.to_csv(out_csv, index=False)

    best = {"alpha": float(result.x[0]), "ngram_max": int(result.x[1]), "vocab_size": int(result.x[2]), "best_objective": float(result.fun)}
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)

    # Optional: dump skopt result for later analysis
    try:
        import joblib as _joblib
        _joblib.dump(result, Path(args.skopt_dump))
    except Exception:
        pass

    print(f"Best parameters: alpha={best['alpha']:.3e}, ngram_max={best['ngram_max']}, vocab_size={best['vocab_size']}")
    print(f"Best mean Spearman: {-best['best_objective']:.6f}")
    print(f"Saved trials to {out_csv} and best params to {best_path}")


if __name__ == "__main__":
    main()
