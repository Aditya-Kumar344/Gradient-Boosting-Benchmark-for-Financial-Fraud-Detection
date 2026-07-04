import time
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

import config


def _random_search(model_class, base_params, param_dist, X, y, n_iter, cv, random_state):
    # Light random search: sample n_iter param combos, score each with
    # stratified CV (mean ROC-AUC), return the best-scoring fitted model
    # (refit on the full X/y with the winning params)
    candidates = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=random_state))
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    best_score = -np.inf
    best_params = None

    for i, params in enumerate(candidates, 1):
        full_params = {**base_params, **params}
        fold_scores = []
        for train_idx, val_idx in skf.split(X, y):
            model = model_class(**full_params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict_proba(X.iloc[val_idx])[:, 1]
            fold_scores.append(roc_auc_score(y.iloc[val_idx], preds))
        mean_score = float(np.mean(fold_scores))
        print(f"  [{i}/{len(candidates)}] mean CV ROC-AUC={mean_score:.4f}  params={params}")
        if mean_score > best_score:
            best_score = mean_score
            best_params = full_params

    final_model = model_class(**best_params)
    final_model.fit(X, y)
    return final_model, best_params


def evaluate_model(name, model_class, base_params, param_dist, X_train, y_train, X_val, y_val,
                    tune=None, n_iter=None, cv=None):
    tune = config.TUNE if tune is None else tune
    n_iter = config.N_ITER if n_iter is None else n_iter
    cv = config.CV_FOLDS if cv is None else cv

    if tune:
        start = time.time()
        model, best_params = _random_search(
            model_class, base_params, param_dist, X_train, y_train,
            n_iter=n_iter, cv=cv, random_state=config.RANDOM_STATE,
        )
        train_time = time.time() - start
        print(f"{name} best params: { {k: v for k, v in best_params.items() if k in param_dist} }")
    else:
        start = time.time()
        model = model_class(**base_params)
        model.fit(X_train, y_train)
        train_time = time.time() - start

    start = time.time()
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    inference_time = time.time() - start
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_val, y_pred_proba)
    f1 = f1_score(y_val, y_pred)

    print(f"{name} -> ROC-AUC: {auc:.4f} | F1: {f1:.4f} | "
          f"Train: {train_time:.2f}s | Infer: {inference_time:.4f}s")

    result = {
        "Model": name,
        "ROC-AUC": round(auc, 4),
        "F1": round(f1, 4),
        "Training Time (s)": round(train_time, 2),
        "Inference Time (s)": round(inference_time, 4),
    }
    return model, result


def run_all_models(model_configs, X_train, y_train, X_val, y_val):
    results = []
    trained_models = {}
    for name, (model_class, base_params, param_dist) in model_configs.items():
        print(f"\n--- Training {name} ---")
        model, result = evaluate_model(
            name, model_class, base_params, param_dist, X_train, y_train, X_val, y_val
        )
        trained_models[name] = model
        results.append(result)

    results_df = pd.DataFrame(results)[
        ["Model", "ROC-AUC", "F1", "Training Time (s)", "Inference Time (s)"]
    ]
    return results_df, trained_models
