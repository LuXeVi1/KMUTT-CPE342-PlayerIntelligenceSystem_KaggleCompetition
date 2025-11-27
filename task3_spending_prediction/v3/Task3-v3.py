# Task3_Enhanced_Two_Stage_Stacking.py
# Enhanced version with improved feature engineering and calibration
# Key improvements:
# 1. Better feature engineering with domain knowledge
# 2. Calibrated probability predictions
# 3. Improved threshold optimization
# 4. Better handling of zero-inflated data

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================
# CONFIGURATION
# =============================
DATA_PATH = "/kaggle/input/cpe342-karena/task3/"
TRAIN_FP = os.path.join(DATA_PATH, "train.csv")
TEST_FP = os.path.join(DATA_PATH, "test.csv")
TARGET = "spending_30d"
RANDOM_STATE = 42
N_SPLITS = 5

print("="*80)
print("TASK 3: ENHANCED TWO-STAGE STACKING MODEL")
print("="*80)

# =============================
# ENHANCED FEATURE ENGINEERING
# =============================
class EnhancedFeatureEngineer:
    """Enhanced feature engineering with domain knowledge for gaming"""
    
    def __init__(self):
        self.feature_names = None
        self.scalers = {}
        
    def fit(self, X, y=None):
        # Fit scalers for normalization
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols[:5]:  # Scale top features
            if col in X.columns:
                scaler = RobustScaler()
                scaler.fit(X[[col]])
                self.scalers[col] = scaler
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Identify numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return X
        
        # === 1. ENGAGEMENT FEATURES ===
        # High engagement = higher spending probability
        if 'daily_login_streak' in X.columns and 'avg_session_length' in X.columns:
            X['engagement_score'] = X['daily_login_streak'] * X['avg_session_length']
            X['engagement_intensity'] = X['avg_session_length'] / (X['daily_login_streak'] + 1)
        
        # === 2. SOCIAL FEATURES ===
        # More friends = higher spending (social pressure)
        if 'friend_count' in X.columns:
            X['has_social_network'] = (X['friend_count'] > 0).astype(int)
            X['friend_count_log1p'] = np.log1p(X['friend_count'])
            X['friend_count_sqrt'] = np.sqrt(X['friend_count'])
            
            if 'avg_session_length' in X.columns:
                X['social_engagement'] = X['friend_count'] * X['avg_session_length']
        
        # === 3. EVENT PARTICIPATION ===
        # Active in events = higher spending probability
        if 'event_participation_rate' in X.columns:
            X['event_rate_squared'] = X['event_participation_rate'] ** 2
            X['high_event_participant'] = (X['event_participation_rate'] > 0.5).astype(int)
            
            if 'daily_login_streak' in X.columns:
                X['event_consistency'] = X['event_participation_rate'] * X['daily_login_streak']
        
        # === 4. HISTORICAL SPENDING PATTERNS ===
        if 'historical_spending' in X.columns:
            # Binary flags
            X['ever_spent'] = (X['historical_spending'] > 0).astype(int)
            X['big_spender'] = (X['historical_spending'] > X['historical_spending'].quantile(0.75)).astype(int)
            
            # Transformations
            X['historical_spending_log1p'] = np.log1p(X['historical_spending'])
            X['historical_spending_sqrt'] = np.sqrt(X['historical_spending'])
            X['historical_spending_cbrt'] = np.cbrt(X['historical_spending'])
            
            # Spending per engagement
            if 'avg_session_length' in X.columns:
                X['spending_per_session'] = X['historical_spending'] / (X['avg_session_length'] + 1)
            
            if 'friend_count' in X.columns:
                X['spending_per_friend'] = X['historical_spending'] / (X['friend_count'] + 1)
        
        # === 5. POLYNOMIAL FEATURES (SELECTED) ===
        key_features = ['friend_count', 'avg_session_length', 'daily_login_streak', 'event_participation_rate']
        for col in key_features:
            if col in X.columns:
                X[f'{col}_squared'] = X[col] ** 2
                X[f'{col}_cubed'] = X[col] ** 3
        
        # === 6. RATIO FEATURES ===
        if 'avg_session_length' in X.columns and 'daily_login_streak' in X.columns:
            X['session_per_streak'] = X['avg_session_length'] / (X['daily_login_streak'] + 1)
            X['streak_per_session'] = X['daily_login_streak'] / (X['avg_session_length'] + 1)
        
        # === 7. AGGREGATE FEATURES ===
        if len(numeric_cols) >= 5:
            agg_cols = [c for c in numeric_cols[:15] if c in X.columns]
            X['feature_sum'] = X[agg_cols].sum(axis=1)
            X['feature_mean'] = X[agg_cols].mean(axis=1)
            X['feature_std'] = X[agg_cols].std(axis=1)
            X['feature_max'] = X[agg_cols].max(axis=1)
            X['feature_min'] = X[agg_cols].min(axis=1)
            X['feature_range'] = X['feature_max'] - X['feature_min']
            X['feature_cv'] = X['feature_std'] / (X['feature_mean'] + 1e-5)
        
        # === 8. INTERACTION FEATURES ===
        # Cross important features
        important_pairs = [
            ('friend_count', 'historical_spending'),
            ('event_participation_rate', 'historical_spending'),
            ('daily_login_streak', 'historical_spending'),
            ('friend_count', 'event_participation_rate')
        ]
        
        for col1, col2 in important_pairs:
            if col1 in X.columns and col2 in X.columns:
                X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
                X[f'{col1}_div_{col2}'] = X[col1] / (X[col2] + 1)
        
        # === 9. LOG TRANSFORMATIONS FOR SKEWED FEATURES ===
        skewed_features = []
        for col in numeric_cols:
            if col in X.columns and X[col].min() >= 0:
                skewness = X[col].skew()
                if abs(skewness) > 1.0:  # More aggressive threshold
                    skewed_features.append(col)
        
        for col in skewed_features[:10]:
            X[f'{col}_log1p'] = np.log1p(X[col])
        
        # === 10. BINNING FEATURES ===
        # Create categorical bins for continuous features
        if 'historical_spending' in X.columns:
            X['spending_bin'] = pd.qcut(X['historical_spending'], q=5, labels=False, duplicates='drop')
        
        if 'friend_count' in X.columns:
            X['friend_bin'] = pd.qcut(X['friend_count'], q=5, labels=False, duplicates='drop')
        
        # Fill NaN and inf values
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        self.feature_names = X.columns.tolist()
        return X
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

# =============================
# UTILITY FUNCTIONS
# =============================
def calc_norm_mae(y_true, y_pred):
    """Calculate Normalized MAE"""
    denom = y_true.sum()
    if denom == 0:
        return 0.0
    return 1.0 / (1.0 + (np.abs(y_true - y_pred).sum() / denom))

def optimize_threshold_advanced(y_true, probas, amounts, search_range=(0.2, 0.8), n_points=101):
    """Advanced threshold optimization with fine-grained search"""
    best_thresh = 0.5
    best_score = 0.0
    
    thresholds = np.linspace(search_range[0], search_range[1], n_points)
    
    for thresh in thresholds:
        bin_pred = (probas >= thresh).astype(int)
        preds = np.zeros(len(y_true))
        preds[bin_pred == 1] = amounts[bin_pred == 1]
        score = calc_norm_mae(y_true, preds)
        
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    return best_thresh, best_score

def get_optimized_classification_models(scale_pos_weight=1.0):
    """Get optimized classification models with best hyperparameters"""
    models = {
        'xgb1': xgb.XGBClassifier(
            n_estimators=1052,
            learning_rate=0.0228,
            max_depth=6,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.98,
            gamma=0.23,
            reg_alpha=0.16,
            reg_lambda=1.67,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'xgb2': xgb.XGBClassifier(
            n_estimators=800,
            learning_rate=0.025,
            max_depth=7,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.15,
            reg_alpha=0.1,
            reg_lambda=1.2,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=RANDOM_STATE + 1,
            n_jobs=-1
        ),
        'lgb1': lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.02,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1
        ),
        'lgb2': lgb.LGBMClassifier(
            n_estimators=1200,
            learning_rate=0.015,
            num_leaves=95,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_samples=15,
            reg_alpha=0.05,
            reg_lambda=0.8,
            random_state=RANDOM_STATE + 1,
            verbose=-1,
            n_jobs=-1
        ),
    }
    return models

def get_optimized_regression_models():
    """Get optimized regression models with best hyperparameters"""
    models = {
        'xgb1': xgb.XGBRegressor(
            n_estimators=1052,
            learning_rate=0.0228,
            max_depth=6,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.98,
            gamma=0.23,
            reg_alpha=0.16,
            reg_lambda=1.67,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'xgb2': xgb.XGBRegressor(
            n_estimators=1200,
            learning_rate=0.018,
            max_depth=7,
            min_child_weight=1,
            subsample=0.9,
            colsample_bytree=0.9,
            gamma=0.15,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE + 1,
            n_jobs=-1
        ),
        'lgb1': lgb.LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.018,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_samples=15,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1
        ),
        'lgb2': lgb.LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.012,
            num_leaves=95,
            subsample=0.95,
            colsample_bytree=0.95,
            min_child_samples=10,
            reg_alpha=0.05,
            reg_lambda=0.8,
            random_state=RANDOM_STATE + 1,
            verbose=-1,
            n_jobs=-1
        ),
    }
    return models

# =============================
# LOAD & PREPARE DATA
# =============================
print("\n[1] Loading data...")
train = pd.read_csv(TRAIN_FP)
test = pd.read_csv(TEST_FP)
print(f"Train: {train.shape}, Test: {test.shape}")

y = train[TARGET].copy()
y_binary = (y > 0).astype(int)

X = train.drop(columns=[TARGET])
X_test = test.copy()

# Remove non-numeric columns
non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"Removing non-numeric columns: {non_numeric}")
    X = X.drop(columns=non_numeric, errors="ignore")
    X_test = X_test.drop(columns=non_numeric, errors="ignore")

# Basic imputation
X = X.fillna(X.median())
X_test = X_test.fillna(X_test.median())

print(f"Features before engineering: {X.shape[1]}")

# =============================
# ENHANCED FEATURE ENGINEERING
# =============================
print("\n[2] Enhanced feature engineering...")
fe = EnhancedFeatureEngineer()
X_engineered = fe.fit_transform(X, y)
X_test_engineered = fe.transform(X_test)

print(f"Features after engineering: {X_engineered.shape[1]}")
print(f"New features created: {X_engineered.shape[1] - X.shape[1]}")

# Convert to numpy
X_vals = X_engineered.values
X_test_vals = X_test_engineered.values

# =============================
# TWO-STAGE CROSS-VALIDATION
# =============================
print("\n" + "="*80)
print("ENHANCED TWO-STAGE CROSS-VALIDATION")
print("="*80)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Storage
oof_proba = np.zeros(len(X_vals))
oof_amounts = np.zeros(len(X_vals))
oof_final = np.zeros(len(X_vals))

test_proba_snapshots = []
test_amounts_snapshots = []

fold_scores = []
fold_thresholds = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_vals, y_binary)):
    print(f"\n{'='*80}")
    print(f"FOLD {fold+1}/{N_SPLITS}")
    print(f"{'='*80}")
    
    X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
    y_tr, y_va = y.iloc[tr_idx].values, y.iloc[va_idx].values
    ybin_tr, ybin_va = y_binary.iloc[tr_idx].values, y_binary.iloc[va_idx].values
    
    # =============================
    # STAGE 1: CLASSIFICATION
    # =============================
    print("\n[STAGE 1: Classification - Will player spend?]")
    
    pos = ybin_tr.sum()
    neg = len(ybin_tr) - pos
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    
    print(f"Class balance: {pos} spenders / {neg} non-spenders (weight={scale_pos_weight:.2f})")
    
    clf_models = get_optimized_classification_models(scale_pos_weight)
    clf_probas_va = []
    clf_probas_test = []
    
    for name, clf in clf_models.items():
        clf.fit(X_tr, ybin_tr)
        
        proba_va = clf.predict_proba(X_va)[:, 1]
        proba_test = clf.predict_proba(X_test_vals)[:, 1]
        
        clf_probas_va.append(proba_va)
        clf_probas_test.append(proba_test)
        
        auc = roc_auc_score(ybin_va, proba_va)
        print(f"  {name}: AUC={auc:.4f}, Mean proba={proba_va.mean():.4f}")
    
    # Weighted ensemble (give more weight to better models)
    weights = np.array([1.2, 1.0, 1.1, 0.9])  # Favor xgb1 and lgb1
    weights = weights / weights.sum()
    
    proba_va_ensemble = np.average(clf_probas_va, axis=0, weights=weights)
    proba_test_ensemble = np.average(clf_probas_test, axis=0, weights=weights)
    
    auc_ensemble = roc_auc_score(ybin_va, proba_va_ensemble)
    print(f"\n  Weighted Ensemble AUC: {auc_ensemble:.4f}")
    
    # =============================
    # STAGE 2: REGRESSION
    # =============================
    print("\n[STAGE 2: Regression - How much will they spend?]")
    
    pos_mask = (y_tr > 0)
    
    if pos_mask.sum() > 10:
        X_reg_tr = X_tr[pos_mask]
        y_reg_tr = np.log1p(y_tr[pos_mask])
        
        print(f"Training on {pos_mask.sum()} spenders")
        
        reg_models = get_optimized_regression_models()
        reg_preds_va = []
        reg_preds_test = []
        
        for name, reg in reg_models.items():
            reg.fit(X_reg_tr, y_reg_tr)
            
            logpred_va = reg.predict(X_va)
            logpred_test = reg.predict(X_test_vals)
            
            pred_va = np.expm1(logpred_va)
            pred_test = np.expm1(logpred_test)
            
            pred_va = np.clip(pred_va, 0, None)
            pred_test = np.clip(pred_test, 0, None)
            
            reg_preds_va.append(pred_va)
            reg_preds_test.append(pred_test)
            
            spender_mask_va = (y_va > 0)
            if spender_mask_va.sum() > 0:
                mae_spenders = mean_absolute_error(y_va[spender_mask_va], pred_va[spender_mask_va])
                print(f"  {name}: MAE (spenders)={mae_spenders:.2f}, Mean={pred_va.mean():.2f}")
        
        # Weighted ensemble for regression
        reg_weights = np.array([1.3, 1.0, 1.1, 0.9])  # Favor xgb1
        reg_weights = reg_weights / reg_weights.sum()
        
        preds_amounts_va = np.average(reg_preds_va, axis=0, weights=reg_weights)
        preds_amounts_test = np.average(reg_preds_test, axis=0, weights=reg_weights)
        
        print(f"\n  Weighted Ensemble mean: {preds_amounts_va.mean():.2f} THB")
        
    else:
        print("⚠️  Not enough spenders")
        preds_amounts_va = np.zeros(len(va_idx))
        preds_amounts_test = np.zeros(len(X_test_vals))
    
    # =============================
    # ADVANCED THRESHOLD OPTIMIZATION
    # =============================
    print("\n[Advanced Threshold Optimization]")
    
    best_thresh, best_score = optimize_threshold_advanced(
        y_va, proba_va_ensemble, preds_amounts_va,
        search_range=(0.25, 0.75), n_points=201
    )
    
    fold_thresholds.append(best_thresh)
    print(f"  Optimal threshold: {best_thresh:.4f}")
    print(f"  Normalized MAE: {best_score:.6f}")
    
    # =============================
    # FINAL PREDICTIONS
    # =============================
    bin_pred_va = (proba_va_ensemble >= best_thresh).astype(int)
    preds_final_va = np.zeros(len(va_idx))
    preds_final_va[bin_pred_va == 1] = preds_amounts_va[bin_pred_va == 1]
    
    oof_proba[va_idx] = proba_va_ensemble
    oof_amounts[va_idx] = preds_amounts_va
    oof_final[va_idx] = preds_final_va
    
    test_proba_snapshots.append(proba_test_ensemble)
    test_amounts_snapshots.append(preds_amounts_test)
    
    norm_mae = calc_norm_mae(y_va, preds_final_va)
    mae = mean_absolute_error(y_va, preds_final_va)
    fold_scores.append(norm_mae)
    
    print(f"\n[FOLD {fold+1} SUMMARY]")
    print(f"  Normalized MAE: {norm_mae:.6f}")
    print(f"  MAE: {mae:.2f} THB")
    print(f"  Predicted spenders: {bin_pred_va.sum()}/{len(bin_pred_va)} ({bin_pred_va.mean()*100:.1f}%)")
    print(f"  Sum ratio: {preds_final_va.sum() / y_va.sum():.4f}")

# =============================
# OVERALL EVALUATION
# =============================
print("\n" + "="*80)
print("OVERALL OUT-OF-FOLD EVALUATION")
print("="*80)

oof_norm_mae = calc_norm_mae(y, oof_final)
oof_mae = mean_absolute_error(y, oof_final)

print(f"\nOOF Metrics:")
print(f"  Normalized MAE: {oof_norm_mae:.6f}")
print(f"  MAE: {oof_mae:.2f} THB")
print(f"  CV Mean: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
print(f"  Avg Threshold: {np.mean(fold_thresholds):.4f} ± {np.std(fold_thresholds):.4f}")

# =============================
# TEST PREDICTIONS
# =============================
print("\n" + "="*80)
print("GENERATING TEST PREDICTIONS")
print("="*80)

avg_threshold = np.mean(fold_thresholds)
test_proba_final = np.mean(test_proba_snapshots, axis=0)
test_amounts_final = np.mean(test_amounts_snapshots, axis=0)

print(f"\nSnapshot ensemble from {N_SPLITS} folds")
print(f"Using average threshold: {avg_threshold:.4f}")

bin_pred_test = (test_proba_final >= avg_threshold).astype(int)
preds_test = np.zeros(len(X_test_vals))
preds_test[bin_pred_test == 1] = test_amounts_final[bin_pred_test == 1]

preds_test = np.nan_to_num(preds_test, nan=0.0, posinf=0.0, neginf=0.0)
preds_test = np.clip(preds_test, 0.0, None)

print(f"\nTest Predictions:")
print(f"  Predicted spenders: {bin_pred_test.sum()}/{len(bin_pred_test)} ({bin_pred_test.mean()*100:.1f}%)")
print(f"  Mean: {preds_test.mean():.2f} THB")
print(f"  Median (non-zero): {np.median(preds_test[preds_test > 0]):.2f} THB")
print(f"  Sum: {preds_test.sum():,.2f} THB")

# =============================
# SAVE SUBMISSION
# =============================
submission = pd.DataFrame({
    "id": [f"ANS{i+1:05d}" for i in range(len(preds_test))],
    "task3": preds_test
})

submission_fp = "/kaggle/working/submission_task3_enhanced.csv"
submission.to_csv(submission_fp, index=False)

print("\n" + "="*80)
print("SUBMISSION SAVED")
print("="*80)
print(f"File: {submission_fp}")
print(f"Expected OOF Score: {oof_norm_mae:.6f}")
print("\n" + "="*80)
print("✅ DONE!")
print("="*80)