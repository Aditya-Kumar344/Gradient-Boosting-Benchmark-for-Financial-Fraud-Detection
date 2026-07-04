# Run the full benchmark end to end

import argparse
import gc
import os
import config
from data_loader import download_dataset, load_and_merge
from preprocessing import prepare_data
from models import get_model_configs
from evaluate import run_all_models
from explain import explain_best_model


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark XGBoost / LightGBM / CatBoost on IEEE-CIS Fraud Detection"
    )
    p.add_argument("--sample-frac", type=float, default=config.SAMPLE_FRAC,
                    help="Fraction of rows to use (default: 1.0 = full data)")
    p.add_argument("--no-tune", action="store_true",
                    help="Skip RandomizedSearchCV; fit each model with its default params")
    p.add_argument("--n-iter", type=int, default=config.N_ITER,
                    help="Random search trials per model (default: 10)")
    p.add_argument("--cv-folds", type=int, default=config.CV_FOLDS,
                    help="CV folds for the random search (default: 3)")
    p.add_argument("--n-jobs", type=int, default=config.N_JOBS,
                    help="CPU threads to use, -1 = all cores (default: -1)")
    p.add_argument("--data-dir", type=str, default=None,
                    help="Path to an already-downloaded competition folder; "
                         "skips the Kaggle download if given")
    return p.parse_args()


def main():
    args = parse_args()

    # Apply CLI overrides to the shared config module
    config.SAMPLE_FRAC = args.sample_frac
    config.TUNE = not args.no_tune
    config.N_ITER = args.n_iter
    config.CV_FOLDS = args.cv_folds
    config.N_JOBS = args.n_jobs

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("STEP 1/5: Download / locate data")
    data_path = args.data_dir if args.data_dir else download_dataset()

    print("STEP 2/5: Load and merge transaction + identity tables")
    df = load_and_merge(data_path)

    print("STEP 3/5: Preprocess (missing-value handling, frequency encoding)")
    X_train, X_val, y_train, y_val = prepare_data(df)
    del df
    gc.collect()

    print("STEP 4/5: Train & evaluate XGBoost / LightGBM / CatBoost")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")
    model_configs = get_model_configs(scale_pos_weight)
    results_df, trained_models = run_all_models(model_configs, X_train, y_train, X_val, y_val)

    print("COMPARISON TABLE")
    print(results_df.to_string(index=False))
    results_path = os.path.join(config.OUTPUT_DIR, "results_comparison.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved to: {results_path}")

    print("STEP 5/5: Explainability (SHAP) for the best model")
    best_model_name, feature_importance = explain_best_model(results_df, trained_models, X_val)
    print(feature_importance.to_string(index=False))

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    fastest_train = results_df.loc[results_df["Training Time (s)"].idxmin(), "Model"]
    fastest_infer = results_df.loc[results_df["Inference Time (s)"].idxmin(), "Model"]
    print(f"Best overall model (highest ROC-AUC): {best_model_name}")
    print(f"Fastest to train:                     {fastest_train}")
    print(f"Fastest at inference:                 {fastest_infer}")
    print(
        "Recommended for deployment: production fraud scoring is latency-sensitive, so prefer the fastest-inference model unless its "
        "ROC-AUC/F1 trail the leader by a meaningful margin."
    )
    print(f"\nAll outputs saved under: {os.path.abspath(config.OUTPUT_DIR)}/")

if __name__ == "__main__":
    main()
