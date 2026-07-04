# Handles Kaggle authentication, downloading the IEEE-CIS competition files,
# and loading + merging the transaction and identity tables

# Kaggle credentials are intentionally not hardcoded here: set them up once
# per README.md (either ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY
# env vars) and every run will just work

import os

import pandas as pd

import config
from utils import downcast_dtypes


def ensure_kaggle_credentials():
    # Raise a clear error early if no Kaggle credentials are configured
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    has_file = os.path.exists(kaggle_json)
    has_env = bool(os.environ.get("KAGGLE_USERNAME")) and bool(os.environ.get("KAGGLE_KEY"))
    if not (has_file or has_env):
        raise EnvironmentError(
            "No Kaggle credentials found.\n"
            f"  Option A: place your kaggle.json at {kaggle_json} (chmod 600)\n"
            "  Option B: export KAGGLE_USERNAME=... and KAGGLE_KEY=...\n"
            "See README.md 'Kaggle credentials' section for the exact steps."
        )


def download_dataset() -> str:
    ensure_kaggle_credentials()
    import kagglehub
    path = kagglehub.competition_download(config.KAGGLE_COMPETITION)
    print(f"Dataset available at: {path}")
    return path


def load_and_merge(path: str) -> pd.DataFrame:
    # Load train_transaction.csv + train_identity.csv, left-join on
    # TransactionID, downcast dtypes to save memory
    txn_path = os.path.join(path, "train_transaction.csv")
    idn_path = os.path.join(path, "train_identity.csv")

    if not os.path.exists(txn_path):
        raise FileNotFoundError(
            f"Expected {txn_path} -- did the Kaggle download complete? "
            "Check the printed path above and confirm the files exist there."
        )

    txn = pd.read_csv(txn_path)
    idn = pd.read_csv(idn_path)

    df = txn.merge(idn, on="TransactionID", how="left")
    del txn, idn

    df = downcast_dtypes(df)

    print(f"Merged shape: {df.shape}")
    print(f"Fraud rate: {df['isFraud'].mean() * 100:.3f}%")
    return df
