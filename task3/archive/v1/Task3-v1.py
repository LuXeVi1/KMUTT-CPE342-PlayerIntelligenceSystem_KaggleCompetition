# Task3_two_stage_xgb_lgb_stacking.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
import xgboost as xgb
import lightgbm as lgb

# -----------------------------
# CONFIG / PATHS
# -----------------------------
DATA_PATH = "/content/drive/MyDrive/CPE342-Karena/CPE342-Hackathon/task3"
TRAIN_FP = os.path.join(DATA_PATH, "train.csv")
TEST_FP = os.path.join(DATA_PATH, "test.csv")
TARGET = "spending_30d"
RANDOM_STATE = 42
N_SPLITS = 5

# -----------------------------
# LOAD
# -----------------------------
train = pd.read_csv(TRAIN_FP)
test = pd.read_csv(TEST_FP)
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# -----------------------------
# PREPROCESS
# -----------------------------
# target and binary label
y = train[TARGET].copy()
y_binary = (y > 0).astype(int)   # 1 = spender, 0 = non-spender

# drop target from X
X = train.drop(columns=[TARGET])
X_test = test.copy()

# drop non-numeric columns (id-like). Keep any numeric-only columns.
non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print("Dropping non-numeric columns:", non_numeric)
X = X.drop(columns=non_numeric, errors="ignore")
X_test = X_test.drop(columns=non_numeric, errors="ignore")

# Fill missing with medians (numeric)
X = X.fillna(X.median(numeric_only=True))
X_test = X_test.fillna(X.median(numeric_only=True))

# convert DataFrame to numpy arrays for models (tree-based models don't need scaling)
X_vals = X.values
X_test_vals = X_test.values

# -----------------------------
# MODEL DEFINITIONS
# -----------------------------
# Classification models (predict whether player will spend > 0)
xgb_clf = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
)

lgb_clf = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
)

# Regression models (predict amount). We'll model log1p(y) to reduce skew.
xgb_reg = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.04,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_STATE,
)

lgb_reg = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.04,
    num_leaves=64,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_STATE,
)

# Stacking regressor that combines xgb_reg and lgb_reg, final estimator Ridge
stacking_reg = StackingRegressor(
    estimators=[("xgb", xgb_reg), ("lgb", lgb_reg)],
    final_estimator=Ridge(alpha=1.0),
    n_jobs=-1,
    passthrough=False,
)

# -----------------------------
# CROSS-VALIDATION: TWO-STAGE
# -----------------------------
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_maes = []
fold_norm_maes = []

# We'll store out-of-fold predictions (useful for ensembling & diagnostics)
oof_preds = np.zeros(len(X_vals))
oof_binary_preds = np.zeros(len(X_vals))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_vals)):
    print(f"\n--- Fold {fold+1} ---")
    X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
    y_tr, y_va = y.iloc[tr_idx].values, y.iloc[va_idx].values
    ybin_tr, ybin_va = y_binary.iloc[tr_idx].values, y_binary.iloc[va_idx].values

    # ---- Stage 1: train classifier on full training fold ----
    # Optionally set scale_pos_weight for xgb if very imbalanced
    pos = ybin_tr.sum()
    neg = len(ybin_tr) - pos
    if pos > 0:
        scale_pos_weight = neg / pos
    else:
        scale_pos_weight = 1.0

    xgb_clf.set_params(scale_pos_weight=scale_pos_weight)
    xgb_clf.fit(X_tr, ybin_tr)
    lgb_clf.fit(X_tr, ybin_tr)

    # Average classifier probabilities (simple ensemble of two classifiers)
    proba_va = 0.5 * xgb_clf.predict_proba(X_va)[:, 1] + 0.5 * lgb_clf.predict_proba(X_va)[:, 1]
    bin_pred_va = (proba_va >= 0.5).astype(int)  # threshold 0.5

    oof_binary_preds[va_idx] = bin_pred_va

    # ---- Stage 2: train regression only on positive (spender) examples from train fold ----
    pos_train_mask = (y_tr > 0)
    if pos_train_mask.sum() == 0:
        # No positive samples in this fold's train: fallback predictions -> zeros
        print("No positive targets in train fold; predicting zeros for regression in this fold.")
        preds_va = np.zeros(len(va_idx))
    else:
        # Prepare regression target as log1p
        y_reg_tr = np.log1p(y_tr[pos_train_mask])
        X_reg_tr = X_tr[pos_train_mask]

        # Fit stacking regressor on positive samples
        stacking_reg.fit(X_reg_tr, y_reg_tr)

        # For validation rows predicted as spender by classifier, predict regression
        preds_va = np.zeros(len(va_idx), dtype=float)
        spenders_idx = np.where(bin_pred_va == 1)[0]
        if len(spenders_idx) > 0:
            X_va_spenders = X_va[spenders_idx]
            logpreds = stacking_reg.predict(X_va_spenders)
            amt_preds = np.expm1(logpreds)   # revert log1p
            # clip to >= 0 (no negative amounts)
            amt_preds = np.clip(amt_preds, 0.0, None)
            preds_va[spenders_idx] = amt_preds
        else:
            # classifier predicted no spenders in val fold
            preds_va = np.zeros(len(va_idx))

    # OOF predictions
    oof_preds[va_idx] = preds_va

    # compute MAE on full validation fold (including zeros)
    mae = mean_absolute_error(y_va, preds_va)
    # compute normalized MAE (per problem formula)
    denom = y_va.sum()
    if denom == 0:
        norm_mae = 0.0
    else:
        norm_mae = 1.0 / (1.0 + (np.abs(y_va - preds_va).sum() / denom))

    print(f"Fold {fold+1} MAE: {mae:.4f}, Normalized MAE: {norm_mae:.6f} (denom sum(y_va)={denom:.2f})")
    fold_maes.append(mae)
    fold_norm_maes.append(norm_mae)

print("\n=== CV Summary ===")
print(f"Average MAE: {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f}")
print(f"Average Normalized MAE: {np.mean(fold_norm_maes):.6f} ± {np.std(fold_norm_maes):.6f}")

# -----------------------------
# TRAIN FINAL MODELS ON FULL DATA & PREDICT TEST SET
# -----------------------------
print("\nTraining final models on full training set...")

# Final Stage 1: classifiers on full data
pos_total = y_binary.sum()
neg_total = len(y_binary) - pos_total
scale_pos_weight_full = neg_total / pos_total if pos_total > 0 else 1.0
xgb_clf.set_params(scale_pos_weight=scale_pos_weight_full)
xgb_clf.fit(X_vals, y_binary)
lgb_clf.fit(X_vals, y_binary)

proba_test = 0.5 * xgb_clf.predict_proba(X_test_vals)[:, 1] + 0.5 * lgb_clf.predict_proba(X_test_vals)[:, 1]
bin_pred_test = (proba_test >= 0.5).astype(int)

# Final Stage 2: train stacking regressor on all positive samples (log1p target)
pos_mask_full = (y.values > 0)
if pos_mask_full.sum() > 0:
    X_reg_full = X_vals[pos_mask_full]
    y_reg_full = np.log1p(y.values[pos_mask_full])
    stacking_reg.fit(X_reg_full, y_reg_full)
    # predict for test rows that classifier said will spend
    preds_test = np.zeros(len(X_test_vals), dtype=float)
    spenders_test_idx = np.where(bin_pred_test == 1)[0]
    if len(spenders_test_idx) > 0:
        X_test_spenders = X_test_vals[spenders_test_idx]
        logpreds_test = stacking_reg.predict(X_test_spenders)
        amt_preds_test = np.expm1(logpreds_test)
        amt_preds_test = np.clip(amt_preds_test, 0.0, None)
        preds_test[spenders_test_idx] = amt_preds_test
else:
    # no positives in full train -> all zeros
    preds_test = np.zeros(len(X_test_vals))

# -----------------------------
# SAVE SUBMISSION
# -----------------------------
# Ensure non-negative and no NaNs
preds_test = np.nan_to_num(preds_test, nan=0.0, posinf=0.0, neginf=0.0)
preds_test = np.clip(preds_test, 0.0, None)

submission = pd.DataFrame({
    "id": [f"ANS{i+1:05d}" for i in range(len(preds_test))],
    "task3": preds_test
})

submission_fp = os.path.join(DATA_PATH, "submission_task3_v2.csv")
submission.to_csv(submission_fp, index=False)
print("Submission saved to:", submission_fp)
print(submission.head())
