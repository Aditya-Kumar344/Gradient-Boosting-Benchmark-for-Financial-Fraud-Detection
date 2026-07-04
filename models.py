# Model classes + base params + light random-search grids for all three models

# Note: we return (model_class, base_params, param_dist) rather than pre-built
# estimator instances. This lets evaluate.py build fresh model objects directly
# during random search instead of relying on sklearn's clone(), which has a
# known incompatibility with CatBoostClassifier

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import config


def get_model_configs(scale_pos_weight: float, n_jobs=None):
    # Returns {name: (model_class, base_params, param_distribution)}
    n_jobs = config.N_JOBS if n_jobs is None else n_jobs

    xgb_base_params = dict(
        objective="binary:logistic", eval_metric="auc", tree_method="hist",
        scale_pos_weight=scale_pos_weight, random_state=config.RANDOM_STATE,
        n_jobs=n_jobs,
    )
    xgb_param_dist = {
        "n_estimators": [200, 400, 600],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }

    lgb_base_params = dict(
        objective="binary", metric="auc", scale_pos_weight=scale_pos_weight,
        random_state=config.RANDOM_STATE, n_jobs=n_jobs, verbosity=-1,
    )
    lgb_param_dist = {
        "n_estimators": [200, 400, 600],
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }

    cat_base_params = dict(
        loss_function="Logloss", eval_metric="AUC",
        scale_pos_weight=scale_pos_weight, random_state=config.RANDOM_STATE,
        verbose=0, thread_count=n_jobs,
    )
    cat_param_dist = {
        "iterations": [200, 400, 600],
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5, 7],
    }

    return {
        "XGBoost": (xgb.XGBClassifier, xgb_base_params, xgb_param_dist),
        "LightGBM": (lgb.LGBMClassifier, lgb_base_params, lgb_param_dist),
        "CatBoost": (CatBoostClassifier, cat_base_params, cat_param_dist),
    }
