# ============================================================================
# TASK 3: SPENDING PREDICTION — V3 OPTIMIZED
# Improvements:
# - Modular Object-Oriented Design
# - Robust Path Handling (Local vs Colab)
# - Optimized Feature Engineering (Vectorized)
# - Two-Stage Stacking (Classification + Regression)
# - Advanced Threshold Optimization
# - Clean Logging & Error Handling
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd
import warnings
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.base import clone

# Boosting Libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    # Environment Detection
    IS_COLAB = 'google.colab' in sys.modules
    
    if IS_COLAB:
        BASE_DIR = Path("/content/drive/MyDrive/ML-Task3-V3-Optimized")
        DATA_DIR = Path("/content/task3")
    else:
        BASE_DIR = Path("./output_task3")
        DATA_DIR = Path("./data")
        
    TRAIN_PATH = DATA_DIR / "train.csv"
    TEST_PATH = DATA_DIR / "test.csv"
    
    # Model Parameters
    N_FOLDS = 5
    RANDOM_STATE = 42
    N_JOBS = -1
    TARGET_COL = "spending_30d"
    
    def __init__(self):
        self.BASE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"📂 Working Directory: {self.BASE_DIR}")
        print(f"📂 Data Directory: {self.DATA_DIR}")

# ============================================================
# UTILITIES
# ============================================================
class Logger:
    @staticmethod
    def section(title):
        print(f"\n{'=' * 80}")
        print(f"🚀 {title.upper()}")
        print(f"{'=' * 80}")

    @staticmethod
    def info(msg):
        print(f"✓ {msg}")

    @staticmethod
    def warn(msg):
        print(f"⚠️ {msg}")

    @staticmethod
    def error(msg):
        print(f"❌ {msg}")

def calc_norm_mae(y_true, y_pred):
    """Calculate Normalized MAE"""
    denom = y_true.sum()
    if denom == 0: return 0.0
    return 1.0 / (1.0 + (np.abs(y_true - y_pred).sum() / denom))

# ============================================================
# FEATURE ENGINEERING
# ============================================================
class FeatureEngineer:
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Helper for safe division
        def safe_div(a, b):
            return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b!=0)
        
        # 1. Engagement
        if 'daily_login_streak' in X and 'avg_session_length' in X:
            X['engagement_score'] = X['daily_login_streak'] * X['avg_session_length']
            X['engagement_intensity'] = safe_div(X['avg_session_length'], X['daily_login_streak'] + 1)
            
        # 2. Social
        if 'friend_count' in X:
            X['has_social_network'] = (X['friend_count'] > 0).astype(int)
            X['friend_count_log1p'] = np.log1p(X['friend_count'])
            if 'avg_session_length' in X:
                X['social_engagement'] = X['friend_count'] * X['avg_session_length']
                
        # 3. Event Participation
        if 'event_participation_rate' in X:
            X['event_rate_sq'] = X['event_participation_rate'] ** 2
            if 'daily_login_streak' in X:
                X['event_consistency'] = X['event_participation_rate'] * X['daily_login_streak']
                
        # 4. Historical Spending (Crucial)
        if 'historical_spending' in X:
            X['ever_spent'] = (X['historical_spending'] > 0).astype(int)
            X['historical_spending_log1p'] = np.log1p(X['historical_spending'])
            
            if 'avg_session_length' in X:
                X['spending_per_session'] = safe_div(X['historical_spending'], X['avg_session_length'] + 1)
                
        # 5. Polynomials (Vectorized)
        poly_cols = ['friend_count', 'avg_session_length', 'daily_login_streak']
        for c in poly_cols:
            if c in X:
                X[f'{c}_sq'] = X[c] ** 2
                
        # 6. Aggregates
        num_cols = X.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 5:
            # Select a subset to avoid noise
            subset = num_cols[:10] 
            X['feat_mean'] = X[subset].mean(axis=1)
            X['feat_max'] = X[subset].max(axis=1)
            
        # Fill NaNs
        X = X.fillna(0)
        return X

# ============================================================
# CORE PIPELINE
# ============================================================
class SpendingPredictionPipeline:
    def __init__(self, config):
        self.cfg = config
        self.fe = FeatureEngineer()
        self.models_clf = {}
        self.models_reg = {}
        
    def load_data(self):
        Logger.section("Loading Data")
        
        if not self.cfg.TRAIN_PATH.exists():
            if Path("train.csv").exists():
                self.cfg.TRAIN_PATH = Path("train.csv")
            else:
                raise FileNotFoundError(f"Train file not found at {self.cfg.TRAIN_PATH}")
                
        if not self.cfg.TEST_PATH.exists():
            if Path("test.csv").exists():
                self.cfg.TEST_PATH = Path("test.csv")
        
        self.train_df = pd.read_csv(self.cfg.TRAIN_PATH)
        self.test_df = pd.read_csv(self.cfg.TEST_PATH) if self.cfg.TEST_PATH.exists() else None
        
        Logger.info(f"Train shape: {self.train_df.shape}")
        if self.test_df: Logger.info(f"Test shape: {self.test_df.shape}")
        
        # Prepare Data
        self.y = self.train_df[self.cfg.TARGET_COL].copy()
        self.y_binary = (self.y > 0).astype(int)
        
        self.X = self.train_df.drop(columns=[self.cfg.TARGET_COL])
        
        # Drop IDs
        drop_cols = ['id', 'player_id']
        self.X = self.X.drop(columns=drop_cols, errors='ignore')
        
        if self.test_df is not None:
            self.test_ids = self.test_df['id']
            self.X_test = self.test_df.drop(columns=drop_cols, errors='ignore')
        else:
            self.X_test = None

    def preprocess(self):
        Logger.section("Feature Engineering")
        
        # Basic Imputation before FE
        num_cols = self.X.select_dtypes(include=[np.number]).columns
        self.X[num_cols] = self.X[num_cols].fillna(self.X[num_cols].median())
        if self.X_test is not None:
            self.X_test[num_cols] = self.X_test[num_cols].fillna(self.X[num_cols].median())
            
        # Advanced FE
        self.X = self.fe.transform(self.X)
        if self.X_test is not None:
            self.X_test = self.fe.transform(self.X_test)
            
        Logger.info(f"Features: {self.X.shape[1]}")
        
        # Convert to numpy for training
        self.X_vals = self.X.values
        self.X_test_vals = self.X_test.values if self.X_test is not None else None

    def _get_clf_models(self, scale_pos_weight):
        return {
            'xgb': xgb.XGBClassifier(
                n_estimators=800, learning_rate=0.02, max_depth=6, 
                scale_pos_weight=scale_pos_weight, eval_metric='logloss',
                random_state=self.cfg.RANDOM_STATE, n_jobs=self.cfg.N_JOBS
            ),
            'lgb': lgb.LGBMClassifier(
                n_estimators=800, learning_rate=0.02, num_leaves=63,
                scale_pos_weight=scale_pos_weight, verbose=-1,
                random_state=self.cfg.RANDOM_STATE, n_jobs=self.cfg.N_JOBS
            )
        }

    def _get_reg_models(self):
        return {
            'xgb': xgb.XGBRegressor(
                n_estimators=800, learning_rate=0.02, max_depth=6,
                objective='reg:squarederror', random_state=self.cfg.RANDOM_STATE, n_jobs=self.cfg.N_JOBS
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=800, learning_rate=0.02, num_leaves=63,
                verbose=-1, random_state=self.cfg.RANDOM_STATE, n_jobs=self.cfg.N_JOBS
            )
        }

    def train_cv(self):
        Logger.section("Two-Stage Cross-Validation")
        
        skf = StratifiedKFold(n_splits=self.cfg.N_FOLDS, shuffle=True, random_state=self.cfg.RANDOM_STATE)
        
        self.oof_preds = np.zeros(len(self.X_vals))
        self.test_preds_accum = np.zeros(len(self.X_test_vals)) if self.X_test_vals is not None else None
        
        fold_scores = []
        
        for fold, (tr_idx, va_idx) in enumerate(skf.split(self.X_vals, self.y_binary), 1):
            print(f"\n📄 Fold {fold}/{self.cfg.N_FOLDS}")
            
            X_tr, X_va = self.X_vals[tr_idx], self.X_vals[va_idx]
            y_tr, y_va = self.y.iloc[tr_idx].values, self.y.iloc[va_idx].values
            ybin_tr, ybin_va = self.y_binary.iloc[tr_idx].values, self.y_binary.iloc[va_idx].values
            
            # --- Stage 1: Classification ---
            pos_ratio = (len(ybin_tr) - ybin_tr.sum()) / ybin_tr.sum()
            clf_models = self._get_clf_models(pos_ratio)
            
            clf_preds_va = []
            clf_preds_test = []
            
            for name, model in clf_models.items():
                model.fit(X_tr, ybin_tr)
                clf_preds_va.append(model.predict_proba(X_va)[:, 1])
                if self.X_test_vals is not None:
                    clf_preds_test.append(model.predict_proba(self.X_test_vals)[:, 1])
            
            # Average Probabilities
            prob_va = np.mean(clf_preds_va, axis=0)
            prob_test = np.mean(clf_preds_test, axis=0) if clf_preds_test else None
            
            print(f"  Clf AUC: {roc_auc_score(ybin_va, prob_va):.4f}")
            
            # --- Stage 2: Regression ---
            # Train only on spenders
            mask_spend = y_tr > 0
            X_tr_reg = X_tr[mask_spend]
            y_tr_reg = np.log1p(y_tr[mask_spend]) # Log transform target
            
            reg_models = self._get_reg_models()
            reg_preds_va = []
            reg_preds_test = []
            
            for name, model in reg_models.items():
                model.fit(X_tr_reg, y_tr_reg)
                
                # Predict on ALL validation data
                pred_va_log = model.predict(X_va)
                reg_preds_va.append(np.expm1(pred_va_log)) # Inverse log
                
                if self.X_test_vals is not None:
                    pred_test_log = model.predict(self.X_test_vals)
                    reg_preds_test.append(np.expm1(pred_test_log))
            
            # Average Amounts
            amt_va = np.mean(reg_preds_va, axis=0)
            amt_va = np.clip(amt_va, 0, None) # No negative spending
            
            amt_test = np.mean(reg_preds_test, axis=0) if reg_preds_test else None
            if amt_test is not None: amt_test = np.clip(amt_test, 0, None)
            
            # --- Threshold Optimization ---
            best_thresh = 0.5
            best_score = 0
            
            # Search for best threshold to multiply prob * amount
            # Or simpler: if prob > thresh then amount, else 0
            thresholds = np.linspace(0.2, 0.8, 50)
            for t in thresholds:
                final_pred = np.where(prob_va >= t, amt_va, 0)
                score = calc_norm_mae(y_va, final_pred)
                if score > best_score:
                    best_score = score
                    best_thresh = t
            
            print(f"  Best Threshold: {best_thresh:.3f} | NormMAE: {best_score:.4f}")
            fold_scores.append(best_score)
            
            # Store Predictions
            final_pred_va = np.where(prob_va >= best_thresh, amt_va, 0)
            self.oof_preds[va_idx] = final_pred_va
            
            if self.test_preds_accum is not None:
                final_pred_test = np.where(prob_test >= best_thresh, amt_test, 0)
                self.test_preds_accum += final_pred_test / self.cfg.N_FOLDS
                
        Logger.info(f"Mean CV NormMAE: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
        
    def save_submission(self):
        if self.test_preds_accum is None: return
        
        Logger.section("Saving Submission")
        
        # Create ID list if needed
        if hasattr(self, 'test_ids'):
            ids = self.test_ids
        else:
            ids = [f"ANS{i+1:05d}" for i in range(len(self.test_preds_accum))]
            
        sub = pd.DataFrame({
            "id": ids,
            "task3": self.test_preds_accum
        })
        
        sub_path = self.cfg.BASE_DIR / "submission_Task3_v3.csv"
        sub.to_csv(sub_path, index=False)
        Logger.info(f"Saved to {sub_path}")
        Logger.info(f"Prediction Stats:\n Mean: {sub['task3'].mean():.2f}\n Max: {sub['task3'].max():.2f}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    config = Config()
    pipeline = SpendingPredictionPipeline(config)
    
    try:
        pipeline.load_data()
        pipeline.preprocess()
        pipeline.train_cv()
        pipeline.save_submission()
        
        Logger.section("Done")
        print("✅ Task 3 V3 Optimized Finished.")
        
    except Exception as e:
        Logger.error(f"Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()
