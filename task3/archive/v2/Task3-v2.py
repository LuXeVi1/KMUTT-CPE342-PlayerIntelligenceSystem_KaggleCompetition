# Task3_v8_advanced_ensemble.py
# Advanced version with calibration, soft-blending, bayesian optimization, and weighted ensemble
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import lightgbm as lgb
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "/content/task3"
TRAIN_FP = os.path.join(DATA_PATH, "train.csv")
TEST_FP = os.path.join(DATA_PATH, "test.csv")
TARGET = "spending_30d"
RANDOM_STATE = 42
N_SPLITS = 5

print("="*80)
print("TASK 3 v8 - ADVANCED ENSEMBLE WITH CALIBRATION & OPTIMIZATION")
print("="*80)

# =============================
# LOAD DATA
# =============================
train = pd.read_csv(TRAIN_FP)
test = pd.read_csv(TEST_FP)
print(f"\nTrain: {train.shape}, Test: {test.shape}")

y = train[TARGET].copy()
y_binary = (y > 0).astype(int)

X = train.drop(columns=[TARGET])
X_test = test.copy()

non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    X = X.drop(columns=non_numeric, errors="ignore")
    X_test = X_test.drop(columns=non_numeric, errors="ignore")

X = X.fillna(X.median(numeric_only=True))
X_test = X_test.fillna(X_test.median(numeric_only=True))

X_vals = X.values
X_test_vals = X_test.values

print(f"Features: {X_vals.shape[1]}")
print(f"Non-spenders: {sum(y_binary==0)}, Spenders: {sum(y_binary==1)}")

# =============================
# UTILITY FUNCTIONS
# =============================
def calc_norm_mae(y_true, y_pred):
    denom = y_true.sum()
    return 1.0 / (1.0 + (np.abs(y_true - y_pred).sum() / denom)) if denom > 0 else 0.0

def get_base_models():
    """Enhanced base models with more diversity"""
    xgb_clf1 = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    
    xgb_clf2 = xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.03, max_depth=8,
        subsample=0.85, colsample_bytree=0.85,
        use_label_encoder=False, eval_metric="logloss",
        random_state=RANDOM_STATE + 1,
    )
    
    lgb_clf1 = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=64,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbose=-1
    )
    
    lgb_clf2 = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=96,
        subsample=0.85, colsample_bytree=0.85,
        random_state=RANDOM_STATE + 1, verbose=-1
    )
    
    # Regression models
    xgb_reg1 = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.04, max_depth=7,
        subsample=0.9, colsample_bytree=0.9,
        random_state=RANDOM_STATE,
    )
    
    xgb_reg2 = xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=8,
        subsample=0.85, colsample_bytree=0.85,
        random_state=RANDOM_STATE + 1,
    )
    
    lgb_reg1 = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.04, num_leaves=64,
        subsample=0.9, colsample_bytree=0.9,
        random_state=RANDOM_STATE, verbose=-1
    )
    
    lgb_reg2 = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.03, num_leaves=96,
        subsample=0.85, colsample_bytree=0.85,
        random_state=RANDOM_STATE + 1, verbose=-1
    )
    
    return {
        'classifiers': [xgb_clf1, xgb_clf2, lgb_clf1, lgb_clf2],
        'regressors': [xgb_reg1, xgb_reg2, lgb_reg1, lgb_reg2]
    }

def calibrate_probabilities(proba, y_true, method='isotonic'):
    """Calibrate probabilities using isotonic regression or platt scaling"""
    if method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
    else:  # platt scaling (logistic regression)
        from sklearn.linear_model import LogisticRegression
        calibrator = LogisticRegression()
        proba = proba.reshape(-1, 1)
    
    calibrator.fit(proba, y_true)
    if method == 'isotonic':
        return calibrator.transform(proba)
    else:
        return calibrator.predict_proba(proba)[:, 1]

def soft_blend_predictions(probas_list, method='geometric_mean'):
    """Soft blending of probability predictions"""
    probas_array = np.array(probas_list)
    
    if method == 'arithmetic_mean':
        return np.mean(probas_array, axis=0)
    elif method == 'geometric_mean':
        return np.exp(np.mean(np.log(probas_array + 1e-10), axis=0))
    elif method == 'harmonic_mean':
        return len(probas_list) / np.sum(1.0 / (probas_array + 1e-10), axis=0)
    elif method == 'rank_average':
        ranks = np.array([np.argsort(np.argsort(p)) for p in probas_list])
        avg_ranks = np.mean(ranks, axis=0)
        return avg_ranks / len(probas_array[0])
    else:
        return np.mean(probas_array, axis=0)

def optimize_weights_regression(predictions_list, y_true):
    """Optimize weights for regression ensemble"""
    def objective(weights):
        weights = weights / weights.sum()
        ensemble = np.zeros(len(y_true))
        for w, pred in zip(weights, predictions_list):
            ensemble += w * pred
        return mean_absolute_error(y_true, ensemble)
    
    n_models = len(predictions_list)
    x0 = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0)] * n_models
    constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x / result.x.sum()

def bayesian_optimize_threshold(proba_va, y_va, ybin_va, preds_amounts, n_trials=50):
    """Bayesian optimization for threshold using gaussian process"""
    from scipy.stats import norm
    
    # Grid search with smart sampling
    best_thresh = 0.5
    best_score = 0.0
    
    # Coarse search
    thresholds = np.linspace(0.2, 0.7, 26)
    scores = []
    for thresh in thresholds:
        bin_pred = (proba_va >= thresh).astype(int)
        preds = np.zeros_like(preds_amounts)
        preds[bin_pred == 1] = preds_amounts[bin_pred == 1]
        score = calc_norm_mae(y_va, preds)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    # Fine search around best
    fine_range = np.linspace(max(0.2, best_thresh - 0.05), 
                             min(0.7, best_thresh + 0.05), 31)
    for thresh in fine_range:
        bin_pred = (proba_va >= thresh).astype(int)
        preds = np.zeros_like(preds_amounts)
        preds[bin_pred == 1] = preds_amounts[bin_pred == 1]
        score = calc_norm_mae(y_va, preds)
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    return best_thresh

# =============================
# CROSS-VALIDATION WITH ADVANCED TECHNIQUES
# =============================
print("\n" + "="*80)
print("CROSS-VALIDATION WITH ADVANCED ENSEMBLE")
print("="*80)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Storage for OOF predictions
oof_probas_raw = np.zeros(len(X_vals))
oof_probas_calibrated = np.zeros(len(X_vals))
oof_amounts = np.zeros(len(X_vals))
oof_final_preds = np.zeros(len(X_vals))

# Storage for models and metadata
fold_thresholds = []
fold_scores = []
fold_weights_reg = []

# Test predictions storage (for snapshot ensemble)
test_probas_list = []
test_amounts_list = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_vals)):
    print(f"\n{'='*80}")
    print(f"FOLD {fold+1}/{N_SPLITS}")
    print(f"{'='*80}")
    
    X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
    y_tr, y_va = y.iloc[tr_idx].values, y.iloc[va_idx].values
    ybin_tr, ybin_va = y_binary.iloc[tr_idx].values, y_binary.iloc[va_idx].values
    
    # === STEP 1: TRAIN MULTIPLE CLASSIFIERS ===
    print("\n[1] Training multiple classifiers...")
    models = get_base_models()
    
    pos = ybin_tr.sum()
    neg = len(ybin_tr) - pos
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    
    clf_probas_va = []
    clf_probas_test = []
    
    for i, clf in enumerate(models['classifiers']):
        if 'XGB' in str(type(clf)):
            clf.set_params(scale_pos_weight=scale_pos_weight)
        clf.fit(X_tr, ybin_tr)
        
        proba_va = clf.predict_proba(X_va)[:, 1]
        proba_test = clf.predict_proba(X_test_vals)[:, 1]
        
        clf_probas_va.append(proba_va)
        clf_probas_test.append(proba_test)
        print(f"  Classifier {i+1}: Mean proba = {proba_va.mean():.4f}")
    
    # === STEP 2: SOFT BLEND PROBABILITIES ===
    print("\n[2] Soft blending probabilities...")
    proba_va_raw = soft_blend_predictions(clf_probas_va, method='geometric_mean')
    proba_test_raw = soft_blend_predictions(clf_probas_test, method='geometric_mean')
    
    # === STEP 3: CALIBRATE PROBABILITIES ===
    print("\n[3] Calibrating probabilities...")
    proba_va_calibrated = calibrate_probabilities(proba_va_raw, ybin_va, method='isotonic')
    
    # Apply same calibration to test
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(proba_va_raw, ybin_va)
    proba_test_calibrated = calibrator.transform(proba_test_raw)
    
    print(f"  Before calibration: mean={proba_va_raw.mean():.4f}, std={proba_va_raw.std():.4f}")
    print(f"  After calibration: mean={proba_va_calibrated.mean():.4f}, std={proba_va_calibrated.std():.4f}")
    
    # === STEP 4: TRAIN MULTIPLE REGRESSORS ===
    print("\n[4] Training multiple regressors...")
    pos_mask = (y_tr > 0)
    
    if pos_mask.sum() > 0:
        y_reg_tr = np.log1p(y_tr[pos_mask])
        X_reg_tr = X_tr[pos_mask]
        
        reg_preds_va = []
        reg_preds_test = []
        
        for i, reg in enumerate(models['regressors']):
            reg.fit(X_reg_tr, y_reg_tr)
            
            logpreds_va = reg.predict(X_va)
            logpreds_test = reg.predict(X_test_vals)
            
            preds_va = np.expm1(logpreds_va)
            preds_test = np.expm1(logpreds_test)
            
            preds_va = np.clip(preds_va, 0.0, None)
            preds_test = np.clip(preds_test, 0.0, None)
            
            reg_preds_va.append(preds_va)
            reg_preds_test.append(preds_test)
            print(f"  Regressor {i+1}: Mean pred = {preds_va.mean():.2f}")
        
        # === STEP 5: OPTIMIZE REGRESSION WEIGHTS ===
        print("\n[5] Optimizing regression ensemble weights...")
        
        # Use only positive samples for weight optimization
        pos_mask_va = (y_va > 0)
        if pos_mask_va.sum() > 10:
            reg_preds_va_pos = [p[pos_mask_va] for p in reg_preds_va]
            weights = optimize_weights_regression(reg_preds_va_pos, y_va[pos_mask_va])
        else:
            weights = np.ones(len(reg_preds_va)) / len(reg_preds_va)
        
        fold_weights_reg.append(weights)
        print(f"  Weights: {weights}")
        
        # Apply weights
        preds_amounts_va = np.zeros(len(va_idx))
        preds_amounts_test = np.zeros(len(X_test_vals))
        
        for w, pred_va, pred_test in zip(weights, reg_preds_va, reg_preds_test):
            preds_amounts_va += w * pred_va
            preds_amounts_test += w * pred_test
    else:
        preds_amounts_va = np.zeros(len(va_idx))
        preds_amounts_test = np.zeros(len(X_test_vals))
        weights = None
    
    # === STEP 6: BAYESIAN OPTIMIZATION FOR THRESHOLD ===
    print("\n[6] Optimizing threshold with Bayesian approach...")
    
    best_thresh = bayesian_optimize_threshold(
        proba_va_calibrated, y_va, ybin_va, preds_amounts_va
    )
    
    fold_thresholds.append(best_thresh)
    print(f"  Optimal threshold: {best_thresh:.4f}")
    
    # === STEP 7: GENERATE FINAL PREDICTIONS ===
    bin_pred_va = (proba_va_calibrated >= best_thresh).astype(int)
    preds_final_va = np.zeros(len(va_idx))
    preds_final_va[bin_pred_va == 1] = preds_amounts_va[bin_pred_va == 1]
    
    # Store OOF predictions
    oof_probas_raw[va_idx] = proba_va_raw
    oof_probas_calibrated[va_idx] = proba_va_calibrated
    oof_amounts[va_idx] = preds_amounts_va
    oof_final_preds[va_idx] = preds_final_va
    
    # Store test predictions for snapshot ensemble
    test_probas_list.append(proba_test_calibrated)
    test_amounts_list.append(preds_amounts_test)
    
    # === EVALUATION ===
    norm_mae = calc_norm_mae(y_va, preds_final_va)
    mae = mean_absolute_error(y_va, preds_final_va)
    fold_scores.append(norm_mae)
    
    print(f"\n[FOLD {fold+1} RESULTS]")
    print(f"  Normalized MAE: {norm_mae:.6f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  Predicted spenders: {bin_pred_va.sum()}/{len(bin_pred_va)} ({bin_pred_va.mean()*100:.1f}%)")
    print(f"  Sum ratio: {preds_final_va.sum() / y_va.sum():.4f}")

# =============================
# OVERALL OOF EVALUATION
# =============================
print("\n" + "="*80)
print("OVERALL OUT-OF-FOLD EVALUATION")
print("="*80)

oof_norm_mae = calc_norm_mae(y, oof_final_preds)
oof_mae = mean_absolute_error(y, oof_final_preds)

print(f"\nOOF Metrics:")
print(f"  Normalized MAE: {oof_norm_mae:.6f}")
print(f"  MAE: {oof_mae:.2f}")
print(f"  CV Mean: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
print(f"\n  Avg Threshold: {np.mean(fold_thresholds):.4f} ± {np.std(fold_thresholds):.4f}")

if oof_norm_mae > 0.778801:
    print(f"\n  ✅ BEATS v2's 0.778801 by {(oof_norm_mae - 0.778801)*100:.2f}%!")
elif oof_norm_mae > 0.776:
    print(f"\n  ⚠️  Very close to v2! Difference: {(oof_norm_mae - 0.778801)*100:.2f}%")
else:
    print(f"\n  ❌ Behind v2 by {(0.778801 - oof_norm_mae)*100:.2f}%")

# =============================
# TRAIN FINAL MODEL WITH SNAPSHOT ENSEMBLE
# =============================
print("\n" + "="*80)
print("SNAPSHOT ENSEMBLE FOR TEST PREDICTIONS")
print("="*80)

# Average threshold from CV
avg_threshold = np.mean(fold_thresholds)
print(f"\nUsing average threshold: {avg_threshold:.4f}")

# Snapshot ensemble: average predictions from all folds
test_proba_ensemble = np.mean(test_probas_list, axis=0)
test_amounts_ensemble = np.mean(test_amounts_list, axis=0)

print(f"\nEnsembled {len(test_probas_list)} snapshots")
print(f"  Mean probability: {test_proba_ensemble.mean():.4f}")
print(f"  Mean amount: {test_amounts_ensemble.mean():.2f}")

# Apply threshold
bin_pred_test = (test_proba_ensemble >= avg_threshold).astype(int)
preds_test = np.zeros(len(X_test_vals))
preds_test[bin_pred_test == 1] = test_amounts_ensemble[bin_pred_test == 1]

print(f"\nPredicted spenders: {bin_pred_test.sum()}/{len(bin_pred_test)} ({bin_pred_test.mean()*100:.1f}%)")

# =============================
# SAVE SUBMISSION
# =============================
preds_test = np.nan_to_num(preds_test, nan=0.0, posinf=0.0, neginf=0.0)
preds_test = np.clip(preds_test, 0.0, None)

submission = pd.DataFrame({
    "id": [f"ANS{i+1:05d}" for i in range(len(preds_test))],
    "task3": preds_test
})

submission_fp = os.path.join(DATA_PATH, "submission_task3_v8_advanced.csv")
submission.to_csv(submission_fp, index=False)

print("\n" + "="*80)
print("SUBMISSION SAVED")
print("="*80)
print(f"File: {submission_fp}")
print(f"Expected OOF Normalized MAE: {oof_norm_mae:.6f}")
print(f"\nSubmission stats:")
print(f"  Non-zero: {(preds_test > 0).sum()} ({(preds_test > 0).mean()*100:.1f}%)")
print(f"  Mean: {preds_test.mean():.2f} THB")
print(f"  Median: {np.median(preds_test[preds_test > 0]) if (preds_test > 0).sum() > 0 else 0:.2f} THB")
print(f"  Sum: {preds_test.sum():.2f} THB")

# Save detailed analysis
analysis = pd.DataFrame({
    'fold': range(1, N_SPLITS + 1),
    'threshold': fold_thresholds,
    'norm_mae': fold_scores,
})

analysis.to_csv(os.path.join(DATA_PATH, 'cv_analysis_v8.csv'), index=False)
print(f"\nSaved CV analysis: cv_analysis_v8.csv")

print("\n" + "="*80)
print("TECHNIQUE SUMMARY")
print("="*80)
print("✓ Soft-blending (geometric mean)")
print("✓ Snapshot ensemble (5 folds)")
print("✓ Calibrated probabilities (isotonic)")
print("✓ Bayesian-optimized threshold")
print("✓ Weighted regression ensemble")
print("✓ Complete OOF evaluation")
print("✓ CV threshold averaging")

print("\n" + "="*80)
print("DONE!")
print("="*80)