# N-gram Regression Model for Antibody Properties

This repository contains the code used for the Ginkgo Antibody Developability (AbDev) Benchmark. You can find the public leaderboard and submission interface here:

- Ginkgo AbDev Benchmark (Hugging Face Space): https://huggingface.co/spaces/ginkgo-datapoints/abdev-leaderboard

The pipeline below reproduces our baseline and generates leaderboard-ready submission files.

## 1) Project goal

Predict key antibody properties from heavy/light chain AA sequences using sparse N-gram features plus subtype one-hot encoding. Models are trained with predefined 5 folds and evaluated by mean Spearman correlation.

---

## 2) Method overview

- Feature engineering
    - Separate TF‑IDF character N‑gram vectorizers for VH (`vh_protein_sequence`) and VL (`vl_protein_sequence`).
    - One‑hot encoded subtype (`hc_subtype`).
    - Final design matrix is the horizontal concatenation of VH TF‑IDF, VL TF‑IDF, and OHE.
- Targets and transforms
    - Log transforms are configurable via `src/config.py::LOG_TRANSFORM_TARGETS` (currently none enabled). If enabled, training uses `log1p` and predictions are inverse‑transformed with `expm1`.
- Models and ensembling
    - Ridge (sparse linear) or Gradient Boosting Regressor. Default: Ridge.
    - For each fold, an ensemble of K bootstrap models is trained; predictions average across bootstrap models and folds.
- Per‑target hyperparameters (new)
    - You can tune and use per‑target `alpha`, `ngram_max`, and `vocab_size`. Artefacts and best params are saved per target and picked up automatically during training/prediction.

---

## 3) Repository layout (key paths)

```
.
├── GDPa1_v1.2_20250814.csv                # training data
├── heldout-set-sequences.csv              # private test sequences
├── requirements.txt
├── artefacts/
│   ├── vectorizer_vh.joblib               # global TF‑IDF VH (fallback)
│   ├── vectorizer_vl.joblib               # global TF‑IDF VL (fallback)
│   ├── encoder_ohe.joblib                 # global OHE (fallback)
│   ├── vectorizer_vh_{Target}.joblib      # per‑target TF‑IDF VH (optional)
│   ├── vectorizer_vl_{Target}.joblib      # per‑target TF‑IDF VL (optional)
│   ├── encoder_ohe_{Target}.joblib        # per‑target OHE (optional)
│   ├── models_{Target}_ridge.joblib       # trained ensembles (ridge)
│   ├── models_{Target}_gbr.joblib         # trained ensembles (gbr)
│   ├── best_params.json                   # tuned params (global + per_target)
│   └── skopt_result[_Target].joblib       # optional skopt dumps
├── results/
│   ├── gdp_a1_cv_predictions_raw.csv      # OOF preds (raw)
│   ├── gdp_a1_cv_predictions.csv          # OOF preds (submission format)
│   ├── private_test_predictions_raw_*.csv # test preds (by model type)
│   └── private_test_predictions.csv       # test preds (submission format)
└── src/
        ├── config.py
        ├── features/build_features.py         # fit & persist vectorizers/OHE
        ├── models/hyperparameter_search.py    # skopt tuning (global/per‑target)
        ├── models/train_model.py              # train ensembles per target
        ├── models/predict_model.py            # predict on private test
        ├── evaluation/evaluate_model.py       # OOF predictions & metrics
        └── submission/create_submission.py    # format final CSVs
```

---

## 4) Quick start

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) (Optional) Tune hyperparameters

- Global/average tuning across all targets

```bash
python -m src.models.hyperparameter_search --target avg --n-iter 50
```

- Single target or all targets (per‑target tuning)

```bash
# one target (faster)
python -m src.models.hyperparameter_search --target Titer --n-iter 50

# all targets (longer)
python -m src.models.hyperparameter_search --target all --n-iter 50
```

This writes trials CSV(s) and updates `artefacts/best_params.json` with:

```json
{
    "global": {"alpha": ..., "ngram_max": ..., "vocab_size": ..., "best_objective": ...},
    "per_target": {
        "Titer": {"alpha": ..., "ngram_max": ..., "vocab_size": ..., "best_objective": ...},
        "HIC":   { ... }
    },
    "alpha": ..., "ngram_max": ..., "vocab_size": ...       // legacy mirror of global
}
```

3) Build features

- Build global transformers (fallback; not really recommended nor supported)

```bash
python -m src.features.build_features
```

- Build per‑target transformers using tuned params

```bash
python -m src.features.build_features --use-best --target all
# or a single target
python -m src.features.build_features --use-best --target Titer
```

4) Train models

```bash
# Ridge (default). Uses per‑target alpha and per‑target vectorizers if available.
python -m src.models.train_model --model-type ridge

# Gradient Boosting (not supported yet)
python -m src.models.train_model --model-type gbr
```

5) Evaluate and get OOF predictions

```bash
mkdir results
python -m src.evaluation.evaluate_model
```

6) Predict on private test

```bash
python -m src.models.predict_model --model-type ridge
```

7) Create final submission files

```bash
python -m src.submission.create_submission
```

By default this uses `MODEL_TYPE` from `src/config.py` and will auto-detect the private test raw predictions file. If you generated predictions with a specific model type, you can be explicit:

```bash
python -m src.submission.create_submission --model-type ridge
# or
python -m src.submission.create_submission --model-type gbr
```

---

## 5) Outputs

- Out-of-fold (training) predictions:
    - `results/gdp_a1_cv_predictions_raw.csv`
    - `results/gdp_a1_cv_predictions.csv`
- Private test predictions:
    - `results/private_test_predictions_raw_{model_type}.csv` (e.g., `_ridge`)
    - `results/private_test_predictions.csv` (submission format)

The submission creator looks for the private test raw predictions in this order:

- `results/private_test_predictions_raw_{model_type}.csv`
- `results/private_test_predictions_raw.csv`
- A unique match of `results/private_test_predictions_raw_*.csv`

You can override detection with `--model-type`.

---

## 6) Troubleshooting

- Error: `Raw private test prediction file not found at .../results/private_test_predictions_raw.csv`
    - Cause: By design, `src/models/predict_model.py` saves to `results/private_test_predictions_raw_{model_type}.csv` (e.g., `_ridge`).
    - Fix:
        - Generate predictions if missing:
            ```bash
            python -m src.models.predict_model --model-type ridge
            ```
        - Or pass the model type explicitly when creating the submission:
            ```bash
            python -m src.submission.create_submission --model-type ridge
            ```

---

These CSVs are formatted to be compatible with the Ginkgo AbDev leaderboard upload flow. Refer to the Space linked above for any updates to the submission format or evaluation process.
