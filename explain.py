# Global SHAP summary plot + top-10 feature table for the best model only
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import shap

import config


def explain_best_model(results_df: pd.DataFrame, trained_models: dict, X_val,
                        output_dir=None, sample_size=None):
    output_dir = config.OUTPUT_DIR if output_dir is None else output_dir
    sample_size = config.SHAP_SAMPLE_SIZE if sample_size is None else sample_size
    os.makedirs(output_dir, exist_ok=True)

    best_model_name = results_df.loc[results_df["ROC-AUC"].idxmax(), "Model"]
    best_model = trained_models[best_model_name]
    print(f"Best model by ROC-AUC: {best_model_name}")

    rng = np.random.default_rng(config.RANDOM_STATE)
    n = min(sample_size, len(X_val))
    sample_idx = rng.choice(X_val.index, size=n, replace=False)
    X_sample = X_val.loc[sample_idx]

    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive (fraud) class

    plt.figure()
    shap.summary_plot(shap_values, X_sample, max_display=10, show=False)
    plot_path = os.path.join(output_dir, "shap_summary_plot.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"SHAP summary plot saved to: {plot_path}")

    feature_importance = (
        pd.DataFrame({
            "feature": X_sample.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    fi_path = os.path.join(output_dir, "top10_features.csv")
    feature_importance.to_csv(fi_path, index=False)
    print(f"Top 10 features saved to: {fi_path}")

    return best_model_name, feature_importance
