import os

# Paths
DATA_DIR = os.environ.get("FRAUD_DATA_DIR", "data")
OUTPUT_DIR = os.environ.get("FRAUD_OUTPUT_DIR", "outputs")

KAGGLE_COMPETITION = "ieee-fraud-detection"

# Preprocessing 
SAMPLE_FRAC = 1.0       # 1.0 = full dataset...Use e.g. 0.3 for a fast test run
MISSING_THRESH = 0.9    # drop columns with > this fraction missing
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Tuning
TUNE = True               # False = skip RandomizedSearchCV, use sane defaults
N_ITER = 10               # random search trials (10-15 = "light tuning")
CV_FOLDS = 3
N_JOBS = -1               # -1 = use all cores

SHAP_SAMPLE_SIZE = 2000
