# Basic feature engineering only:
#  > drop columns with too much missing data (fit on train split only)
#  > frequency-encode categorical columns
#  > constant-fill missing numeric values
import gc

import pandas as pd
from sklearn.model_selection import train_test_split

import config


def get_high_missing_cols(df: pd.DataFrame, thresh: float):
    missing_pct = df.isnull().mean()
    return missing_pct[missing_pct > thresh].index.tolist()


def frequency_encode(df: pd.DataFrame, cols, freq_maps=None, fit=True):
    df = df.copy()
    if fit:
        freq_maps = {c: df[c].value_counts(dropna=False) for c in cols}
    for c in cols:
        df[c] = df[c].map(freq_maps[c]).fillna(0)
    return df, freq_maps


def prepare_data(df: pd.DataFrame, sample_frac=None, missing_thresh=None,
                  test_size=None, random_state=None):
    sample_frac = config.SAMPLE_FRAC if sample_frac is None else sample_frac
    missing_thresh = config.MISSING_THRESH if missing_thresh is None else missing_thresh
    test_size = config.TEST_SIZE if test_size is None else test_size
    random_state = config.RANDOM_STATE if random_state is None else random_state

    target = df["isFraud"].copy()
    df = df.drop(columns=["isFraud"])

    if sample_frac < 1.0:
        idx = df.sample(frac=sample_frac, random_state=random_state).index
        df = df.loc[idx]
        target = target.loc[idx]
        print(f"Sampled shape: {df.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        df, target, test_size=test_size, random_state=random_state, stratify=target
    )
    del df
    gc.collect()

    X_train = X_train.drop(columns=["TransactionID"])
    X_val = X_val.drop(columns=["TransactionID"])

    high_missing_cols = get_high_missing_cols(X_train, missing_thresh)
    X_train = X_train.drop(columns=high_missing_cols)
    X_val = X_val.drop(columns=high_missing_cols)
    print(f"Dropped {len(high_missing_cols)} columns with > {missing_thresh*100:.0f}% missing")

    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    print(f"{len(cat_cols)} categorical, {len(num_cols)} numeric columns")

    X_train, freq_maps = frequency_encode(X_train, cat_cols, fit=True)
    X_val, _ = frequency_encode(X_val, cat_cols, freq_maps=freq_maps, fit=False)

    X_train[num_cols] = X_train[num_cols].fillna(-999)
    X_val[num_cols] = X_val[num_cols].fillna(-999)

    print(f"Final shapes: train={X_train.shape}, val={X_val.shape}")
    return X_train, X_val, y_train, y_val
