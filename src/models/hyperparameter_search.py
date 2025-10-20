import argparse
import json
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
    Integer(1, config.N_GRAM_MAX, name="ngram_max"),
    Integer(1000, 30000, name="vocab_size"),
]


def _load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _fit_featureizers(df: pd.DataFrame, ngram_max: int, vocab_size: int):
    vectorizer_vh, vectorizer_vl, encoder_ohe = get_feature_pipelines(
        ngram_range=(1, ngram_max), vocab_size=vocab_size
    )
    df_fit = df.dropna(subset=[config.VH_SEQUENCE_COL, config.VL_SEQUENCE_COL]).copy()
    encoder_ohe.fit(df_fit[[config.HC_SUBTYPE_COL]])
    vectorizer_vh.fit(df_fit[config.VH_SEQUENCE_COL])
    vectorizer_vl.fit(df_fit[config.VL_SEQUENCE_COL])
    return vectorizer_vh, vectorizer_vl, encoder_ohe


def _cv_score(df: pd.DataFrame, alpha: float, ngram_max: int, vocab_size: int, n_splits: int = 5, random_state: int = 42) -> float:
    """Compute average Spearman correlation across targets and folds.

    Returns negative average (to minimize).
    """
    vectorizer_vh, vectorizer_vl, encoder_ohe = _fit_featureizers(df, ngram_max, vocab_size)
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
            mask = ~np.isnan(y_va)
            if mask.any():
                corr = spearmanr(preds[mask], y_va[mask]).correlation
                if np.isfinite(corr):
                    corrs.append(corr)
    if not corrs:
        return 0.0
    return -float(np.mean(corrs))


def _cv_score_single_target(df: pd.DataFrame, target: str, alpha: float, ngram_max: int, vocab_size: int, n_splits: int = 5, random_state: int = 42) -> float:
    """Compute average Spearman correlation for a single target across folds.

    Returns negative average (to minimize).
    """
    if target not in config.TARGET_PROPERTIES:
        raise ValueError(f"Unknown target '{target}'. Must be one of: {config.TARGET_PROPERTIES}")

    vectorizer_vh, vectorizer_vl, encoder_ohe = _fit_featureizers(df, ngram_max, vocab_size)
    preprocessor = get_preprocessor(vectorizer_vh, vectorizer_vl, encoder_ohe)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=alpha, **{k: v for k, v in config.RIDGE_MODEL_PARAMS.items() if k != "alpha"})),
    ])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    df_t = df.dropna(subset=[target]).copy()
    if df_t.empty:
        return 0.0
    y = df_t[target].values
    corrs = []
    for train_idx, val_idx in kf.split(df_t):
        X_tr, X_va = df_t.iloc[train_idx], df_t.iloc[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        mask = ~np.isnan(y_va)
        if mask.any():
            corr = spearmanr(preds[mask], y_va[mask]).correlation
            if np.isfinite(corr):
                corrs.append(corr)
    if not corrs:
        return 0.0
    return -float(np.mean(corrs))


def _ensure_best_structure(existing: dict | None) -> dict:
    if not isinstance(existing, dict):
        return {"global": {}, "per_target": {}}
    out = {"global": existing.get("global", {}), "per_target": existing.get("per_target", {})}
    # Backward-compat: if legacy top-level keys exist (alpha, etc.), keep them under 'global'
    for k in ("alpha", "ngram_max", "vocab_size", "best_objective"):
        if k in existing and k not in out["global"]:
            out["global"][k] = existing[k]
    return out


def _save_result(res, out_csv: Path, best_path: Path, best_key_path: tuple[str, ...]):
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
            try:
                with open(best_path, "r") as f:
                    existing = json.load(f)
            except Exception:
                existing = None
            store = _ensure_best_structure(existing)

            # Write into nested key path
            cursor = store
            for key in best_key_path[:-1]:
                if key not in cursor:
                    cursor[key] = {}
                cursor = cursor[key]
            cursor[best_key_path[-1]] = best

            # Also mirror 'global' to top-level for backward-compat
            if best_key_path == ("global",):
                for k, v in best.items():
                    store[k] = v

            with open(best_path, "w") as f:
                json.dump(store, f, indent=2)
    except Exception as e:
        print(f"[checkpoint] Failed to save checkpoint: {e}")


def _run_search(space, objective_fn, n_iter: int, random_state: int, out_csv: Path, best_path: Path, best_key_path: tuple[str, ...]):
    @use_named_args(space)
    def objective(**p):
        alpha = float(p["alpha"])  # ensure json serializable
        ngram_max = int(p["ngram_max"])
        vocab_size = int(p["vocab_size"])
        print(f"Trying alpha={alpha:.3e}, ngram_max={ngram_max}, vocab_size={vocab_size}")
        score = objective_fn(alpha=alpha, ngram_max=ngram_max, vocab_size=vocab_size)
        print(f" -> objective (neg mean spearman) = {score:.6f}")
        return score

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    # Mutable holder for last result for Ctrl+C handling
    last_res_holder = {"res": None}

    class CheckpointCallback:
        def __call__(self, res):
            last_res_holder["res"] = res
            _save_result(res, out_csv=out_csv, best_path=best_path, best_key_path=best_key_path)

    try:
        result = gp_minimize(
            objective,
            space,
            n_calls=n_iter,
            random_state=random_state,
            n_initial_points=min(10, n_iter),
            acq_func="EI",
            callback=[CheckpointCallback()],
        )
    except KeyboardInterrupt:
        print("\n[interrupt] Caught Ctrl+C. Saving best-so-far and exiting...")
        if last_res_holder["res"] is not None:
            _save_result(last_res_holder["res"], out_csv=out_csv, best_path=best_path, best_key_path=best_key_path)
        return None

    _save_result(result, out_csv=out_csv, best_path=best_path, best_key_path=best_key_path)
    return result


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search with scikit-optimize")
    parser.add_argument("--data-path", type=str, default=str(config.DATA_FILE), help="CSV with training data")
    parser.add_argument("--n-iter", type=int, default=50, help="Number of optimization calls")
    parser.add_argument("--target", type=str, default="avg", help="Target to optimize: 'avg' (default), a target name, or 'all' to iterate over all targets")
    parser.add_argument("--output-csv", type=str, default=str(config.BASE_DIR.parent / "hyperparameter_search_results.csv"), help="CSV to save trials")
    parser.add_argument("--output-best", type=str, default=str(config.ARTEFACTS_DIR / "best_params.json"), help="Where to save best params JSON")
    parser.add_argument("--skopt-dump", type=str, default=str(config.ARTEFACTS_DIR / "skopt_result.joblib"), help="Path to dump skopt result")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = _load_data(Path(args.data_path))

    best_path = Path(args.output_best)
    out_csv = Path(args.output_csv)

    target_arg = args.target.strip()
    if target_arg.lower() == "avg":
        # Global/average optimization across targets
        def obj(alpha: float, ngram_max: int, vocab_size: int):
            return _cv_score(df, alpha=alpha, ngram_max=ngram_max, vocab_size=vocab_size)

        res = _run_search(SPACE, obj, args.n_iter, args.random_state, out_csv, best_path, ("global",))
        if res is not None:
            print(f"Best parameters (global): alpha={res.x[0]:.3e}, ngram_max={int(res.x[1])}, vocab_size={int(res.x[2])}")
            print(f"Best mean Spearman (global): {-float(res.fun):.6f}")
            print(f"Saved trials to {out_csv} and best params to {best_path}")
        # Optional: dump skopt result for later analysis
        try:
            import joblib as _joblib
            _joblib.dump(res, Path(args.skopt_dump))
        except Exception:
            pass
        return

    # Single target or all targets
    targets = config.TARGET_PROPERTIES if target_arg.lower() == "all" else [target_arg]
    for t in targets:
        if t not in config.TARGET_PROPERTIES:
            print(f"[warn] Skipping unknown target '{t}'. Valid: {config.TARGET_PROPERTIES}")
            continue
        print(f"\n=== Optimizing target: {t} ===")

        def obj_t(alpha: float, ngram_max: int, vocab_size: int, _t=t):
            return _cv_score_single_target(df, target=_t, alpha=alpha, ngram_max=ngram_max, vocab_size=vocab_size)

        out_csv_t = out_csv
        if out_csv_t.name == "hyperparameter_search_results.csv":
            out_csv_t = out_csv_t.with_name(f"hyperparameter_search_results_{t}.csv")
        res = _run_search(SPACE, obj_t, args.n_iter, args.random_state, out_csv_t, best_path, ("per_target", t))
        if res is None:
            continue
        print(f"Best parameters for {t}: alpha={res.x[0]:.3e}, ngram_max={int(res.x[1])}, vocab_size={int(res.x[2])}")
        print(f"Best mean Spearman for {t}: {-float(res.fun):.6f}")
        try:
            import joblib as _joblib
            dump_path = Path(args.skopt_dump)
            if dump_path.name == "skopt_result.joblib":
                dump_path = dump_path.with_name(f"skopt_result_{t}.joblib")
            _joblib.dump(res, dump_path)
        except Exception:
            pass
    print(f"Saved per-target trials to {out_csv.parent} and updated {best_path}")


if __name__ == "__main__":
    main()
