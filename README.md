# IEEE-CIS Fraud Detection — Gradient Boosting Benchmark

Benchmarks **XGBoost**, **LightGBM**, and **CatBoost** on the IEEE-CIS Fraud
Detection dataset, comparing predictive performance, computational
efficiency, and feature importance. Runs locally on macOS (tested on
Apple Silicon M1) — no notebook or cloud environment required.

---
**Project structure**

```
config.py            all tunable settings in one place
utils.py             timing helper, dtype downcasting for memory efficiency
data_loader.py       Kaggle download + load/merge transaction+identity tables
preprocessing.py     missing-value handling, frequency encoding, train/val split
models.py            model classes + base params + light random-search grids
evaluate.py          custom random search, training/inference timing, scoring
explain.py           SHAP summary plot + top-10 feature export for the best model
main.py              orchestrates the full pipeline end to end
requirements.txt     pinned dependency versions (arm64-compatible)
setup.sh             one-shot macOS environment setup (libomp + venv + installs)
outputs/             results_comparison.csv, shap_summary_plot.png, top10_features.csv
```

## 1. Setup (macOS)

### Prerequisites
- macOS with [Homebrew](https://brew.sh) installed
- Python 3.9–3.12
- A free [Kaggle](https://kaggle.com) account, with the competition rules
  accepted on the [IEEE-CIS Fraud Detection page](https://www.kaggle.com/competitions/ieee-fraud-detection/rules)
  (required once, or the API download will fail)

### Step-by-step

**1. Run the setup script**
```bash
chmod +x setup.sh
./setup.sh
```
This installs `libomp` via Homebrew (required by XGBoost and LightGBM on
macOS — without it, both fail to import), creates a virtual environment in
`./venv`, and installs every pinned dependency from `requirements.txt`.

Let it run to completion uninterrupted — interrupting `brew install libomp`
partway through leaves the venv step un-run.

**2. Set up Kaggle credentials** (one-time)
- Kaggle → profile picture → **Settings** → **API** → **Create New Token**
  → downloads `kaggle.json`
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```
(Alternative: `export KAGGLE_USERNAME=...` and `export KAGGLE_KEY=...`
instead of the file, if preferred.)

**3. Activate the environment** (every new terminal session)
```bash
source venv/bin/activate
```
Confirm it worked:
```bash
which python   # should print .../fraud_detection_benchmark/venv/bin/python
```

**4. Run**
```bash
python main.py
```
First run downloads the dataset (~500MB, cached afterward at
`~/.cache/kagglehub/competitions/ieee-fraud-detection`). Subsequent runs can
skip the download entirely:
```bash
python main.py --data-dir ~/.cache/kagglehub/competitions/ieee-fraud-detection
```

**Useful flags** (all optional, override `config.py` defaults):

| Flag | Default | Purpose |
|---|---|---|
| `--sample-frac 0.3` | `1.0` | Use a subset of rows — faster, good for a first sanity-check run |
| `--no-tune` | tuning on | Skip random search, use each model's default params |
| `--n-iter 6` | `10` | Fewer random-search trials |
| `--cv-folds 2` | `3` | Fewer CV folds during search |
| `--n-jobs 4` | `-1` (all cores) | Cap CPU threads if memory gets tight |
| `--data-dir <path>` | none | Reuse an already-downloaded dataset |

**5. Check results**
```bash
open outputs/
```
Produces `results_comparison.csv`, `shap_summary_plot.png`, and
`top10_features.csv`.

---

## 2. What was implemented

### Dataset
IEEE-CIS Fraud Detection: `train_transaction.csv` (590,540 rows) left-joined
with `train_identity.csv` on `TransactionID`, giving 434 columns and a
3.499% fraud rate.

Since Kaggle's official `test_*.csv` files carry no public labels, ROC-AUC
and F1 can't be computed against them. Instead, the labeled training data
was split 80/20 (stratified on `isFraud`) into an internal train/validation
set — this is the only way to actually score the four required metrics for
this competition.

### Feature engineering (basic only, per scope)
1. **Drop high-missing columns** — any column with >90% missing values,
   fit on the training split only (12 columns dropped).
2. **Frequency encoding** — all 29 categorical (object dtype) columns
   mapped to their value counts, fit on train and applied to validation
   (no leakage).
3. **Missing value imputation** — remaining numeric NaNs filled with a
   constant sentinel (`-999`), letting tree-based models split on
   "missing" as its own signal.
4. **Class imbalance** — handled via `scale_pos_weight` (≈27.6, i.e.
   non-fraud outnumbers fraud ~27.6:1) passed to all three models, not via
   resampling.

Final feature set: **420 columns** (391 numeric + 29 frequency-encoded
categorical) after dropping `TransactionID` and the high-missing columns.

### Models & tuning
All three models — **XGBoost**, **LightGBM**, **CatBoost** — were tuned
with the same light random search: **10 trials, 3-fold stratified CV**,
scored on ROC-AUC, over a small hyperparameter grid per model (tree
depth/leaves, learning rate, subsample/colsample ratios, n_estimators or
iterations).

Note: the search was implemented as a small custom loop rather than
scikit-learn's `RandomizedSearchCV`, because newer scikit-learn's internal
`clone()` call is incompatible with `CatBoostClassifier` (a documented
library interaction issue, not specific to this dataset) — the custom loop
builds fresh model instances directly and works identically across all
three libraries.

### Evaluation
Four metrics only, per scope:
- **ROC-AUC** and **F1** on the held-out validation split
- **Training time** (full random search + final refit)
- **Inference time** (predicting on the validation set)

### Explainability
`shap.TreeExplainer` on the best model (by ROC-AUC), applied to a random
2,000-row sample of the validation set for speed. Produces one global SHAP
summary plot and a top-10 feature ranking by mean |SHAP value| — no local
explanations, waterfall, or dependence plots, per scope.

---

## 3. Results (full dataset, 590,540 rows)

Run configuration: `SAMPLE_FRAC=1.0`, `N_ITER=10`, `CV_FOLDS=3`, tuning on,
train/val split = 472,432 / 118,108 rows.

### Comparison table

| Model | ROC-AUC | F1 | Training Time | Inference Time |
|---|---|---|---|---|
| XGBoost | 0.9667 | 0.6399 | 1360.12s (~22.7 min) | 0.4906s |
| **LightGBM** | **0.9688** | **0.6580** | 1205.60s (~20.1 min) | 3.0651s |
| CatBoost | 0.9552 | 0.4627 | 2222.76s (~37.0 min) | **0.2694s** |

### Best hyperparameters found (10-trial random search)

- **XGBoost:** `n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=1.0`
- **LightGBM:** `n_estimators=600, num_leaves=127, learning_rate=0.05, subsample=1.0, colsample_bytree=1.0`
- **CatBoost:** `iterations=400, depth=8, learning_rate=0.1, l2_leaf_reg=7`

### Top 10 features (SHAP, best model = LightGBM)

| Rank | Feature | Mean Absolute SHAP Value |
|---|---|---|
| 1 | `C13` | 0.3020 |
| 2 | `TransactionAmt` | 0.2665 |
| 3 | `P_emaildomain` | 0.2420 |
| 4 | `card1` | 0.2366 |
| 5 | `C14` | 0.2172 |
| 6 | `TransactionDT` | 0.1919 |
| 7 | `card2` | 0.1915 |
| 8 | `V70` | 0.1849 |
| 9 | `D15` | 0.1723 |
| 10 | `card6` | 0.1638 |

**Why these matter:**
- **`C13`, `C14`** — Vesta's engineered "counting" features (e.g. counts of
  addresses/devices tied to a card). Unusual counts flag account or card
  reuse patterns typical of fraud rings.
- **`TransactionAmt`** — fraud transactions cluster at atypical amounts:
  small "card testing" charges or unusually round/high values.
- **`P_emaildomain`** — purchaser's email domain; certain domains
  (disposable/rare providers) correlate strongly with fraud.
- **`card1`, `card2`, `card6`** (frequency-encoded) — card issuer/type/network
  identifiers; rare card profiles are disproportionately fraudulent.
- **`V70`** — an anonymized Vesta-engineered feature; these frequently rank
  highly since Vesta already found them predictive before anonymizing them.
- **`D15`** — a time-delta feature (days since a previous transaction on
  this card/account); unusually short or long gaps signal fraud.
- **`TransactionDT`** — see methodology caveat below; treat this one with
  more suspicion than the others.

---

## 4. Conclusions

- **Best overall model:** **LightGBM** — highest ROC-AUC (0.9688) *and*
  highest F1 (0.6580); no tradeoff needed against the other two.
- **Fastest to train:** **LightGBM** (1205.60s), narrowly ahead of XGBoost.
- **Fastest at inference:** **CatBoost** (0.2694s), but its ROC-AUC and F1
  trail LightGBM by a large enough margin that the speed doesn't
  compensate.
- **Recommended for deployment:** **LightGBM**. It wins on accuracy,
  wins on training time, and its inference latency (≈26µs/transaction
  across the 118k-row validation set) is still comfortably fast enough for
  real-time transaction scoring. Unlike a smaller-sample run of this same
  benchmark (where XGBoost briefly led and CatBoost's inference speed made
  a real case for itself), the full-data result has no genuine
  accuracy-vs-speed tradeoff to weigh — LightGBM is the clear choice.

### Methodology caveat worth flagging in any write-up
The train/validation split here is a **random** stratified split, not a
**time-based** one. `TransactionDT` (a raw elapsed-seconds timestamp)
ranking #6 in feature importance suggests the models may be partly
exploiting temporal clustering of fraud (fraud often occurs in bursts) in a
way that wouldn't generalize to genuinely future transactions. Kaggle's
official test set is time-split from the training data specifically to
prevent this kind of leakage. This doesn't invalidate the benchmark
comparison between the three models (all three were evaluated identically),
but it does mean the absolute ROC-AUC/F1 numbers may be somewhat optimistic
relative to true out-of-time performance.

---
