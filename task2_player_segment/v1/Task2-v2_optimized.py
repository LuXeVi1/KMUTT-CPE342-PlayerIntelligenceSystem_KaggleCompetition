# ============================================================================
# TASK 2: PLAYER SEGMENT CLASSIFICATION — V2 OPTIMIZED
# Improvements:
# - Modular Object-Oriented Design
# - Robust Path Handling (Local vs Colab)
# - Optimized Feature Engineering
# - Advanced Ensemble Modeling with Voting
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer, KNNImputer

# Boosting Libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, ADASYN

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    # Environment Detection
    IS_COLAB = 'google.colab' in sys.modules
    
    if IS_COLAB:
        BASE_DIR = Path("/content/drive/MyDrive/ML-Task2-V2-Optimized")
        DATA_DIR = Path("/content/task2")
    else:
        BASE_DIR = Path("./output_task2")
        DATA_DIR = Path("./data") # Assumes data is in 'data' folder
        
    TRAIN_PATH = DATA_DIR / "train.csv"
    TEST_PATH = DATA_DIR / "test.csv"
    
    # Model Parameters
    N_FOLDS = 5
    RANDOM_STATE = 42
    N_JOBS = -1
    TARGET_COL = "segment"
    
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

# ============================================================
# CORE PIPELINE
# ============================================================
class PlayerSegmentationPipeline:
    def __init__(self, config):
        self.cfg = config
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        
    def load_data(self):
        Logger.section("Loading Data")
        
        # Robust file loading
        if not self.cfg.TRAIN_PATH.exists():
            # Fallback for local testing
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
        if self.test_df is not None:
            Logger.info(f"Test shape: {self.test_df.shape}")
            
        # Clean Target
        self.train_df = self.train_df.dropna(subset=[self.cfg.TARGET_COL]).reset_index(drop=True)
        self.y = self.train_df[self.cfg.TARGET_COL].copy()
        self.X = self.train_df.drop(columns=[self.cfg.TARGET_COL])
        
        # Save IDs for submission
        self.test_ids = self.test_df['id'] if self.test_df is not None and 'id' in self.test_df.columns else None
        
        # Drop ID columns for training
        drop_cols = ['id', 'player_id']
        self.X = self.X.drop(columns=drop_cols, errors='ignore')
        if self.test_df is not None:
            self.X_test = self.test_df.drop(columns=drop_cols, errors='ignore')
        else:
            self.X_test = None

    def _engineer_features(self, df):
        """Vectorized and optimized feature engineering."""
        df = df.copy()
        
        # Helper for safe division
        def safe_div(a, b, fill=0):
            return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b!=0)

        # --- Engagement ---
        if 'avg_session_duration' in df and 'play_frequency' in df:
            df['engagement_score'] = df['avg_session_duration'] * df['play_frequency']
            df['engagement_intensity'] = safe_div(df['avg_session_duration'], df['play_frequency'] + 1)
            
        if 'total_playtime_hours' in df and 'account_age_days' in df:
            df['playtime_per_day'] = safe_div(df['total_playtime_hours'], df['account_age_days'] + 1)
            
        # --- Spending ---
        if 'total_spending_thb' in df:
            if 'avg_monthly_spending' in df:
                df['spending_months'] = safe_div(df['total_spending_thb'], df['avg_monthly_spending'] + 1)
            if 'account_age_days' in df:
                df['spending_per_day'] = safe_div(df['total_spending_thb'], df['account_age_days'] + 1)
            if 'total_playtime_hours' in df:
                df['spending_per_hour'] = safe_div(df['total_spending_thb'], df['total_playtime_hours'] + 1)
                
        # --- Social ---
        if 'friend_count' in df:
            if 'team_play_percentage' in df:
                df['social_engagement'] = df['friend_count'] * df['team_play_percentage'] / 100
            if 'chat_activity_score' in df:
                df['social_interaction'] = df['friend_count'] * df['chat_activity_score']
                
        # --- Competitive ---
        if 'win_rate_ranked' in df and 'ranked_participation_rate' in df:
            df['competitive_success'] = df['win_rate_ranked'] * df['ranked_participation_rate']
            
        # --- Preferences ---
        pref_cols = [c for c in df.columns if 'item_type_preference' in c]
        if pref_cols:
            df['preference_diversity'] = df[pref_cols].std(axis=1)
            df['max_preference'] = df[pref_cols].max(axis=1)
            
        return df

    def preprocess(self):
        Logger.section("Preprocessing & Feature Engineering")
        
        # 1. Feature Engineering
        self.X = self._engineer_features(self.X)
        if self.X_test is not None:
            self.X_test = self._engineer_features(self.X_test)
        Logger.info(f"Features after engineering: {self.X.shape[1]}")
        
        # 2. Identify Column Types
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # 3. Imputation
        # Numeric: Median
        imputer_num = SimpleImputer(strategy='median')
        self.X[numeric_cols] = imputer_num.fit_transform(self.X[numeric_cols])
        if self.X_test is not None:
            self.X_test[numeric_cols] = imputer_num.transform(self.X_test[numeric_cols])
            
        # Categorical: Constant 'Unknown'
        self.X[categorical_cols] = self.X[categorical_cols].fillna('Unknown')
        if self.X_test is not None:
            self.X_test[categorical_cols] = self.X_test[categorical_cols].fillna('Unknown')
            
        # 4. Encoding
        for col in categorical_cols:
            le = LabelEncoder()
            # Fit on both train and test to catch all categories if possible, or handle unseen
            combined = pd.concat([self.X[col], self.X_test[col]], axis=0).astype(str) if self.X_test is not None else self.X[col].astype(str)
            le.fit(combined)
            
            self.X[col] = le.transform(self.X[col].astype(str))
            if self.X_test is not None:
                self.X_test[col] = le.transform(self.X_test[col].astype(str))
            self.label_encoders[col] = le
            
        Logger.info("Imputation and Encoding complete")
        
        # 5. Scaling
        self.X_scaled = self.scaler.fit_transform(self.X)
        if self.X_test is not None:
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
        # 6. Class Balancing (SMOTE)
        class_counts = self.y.value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min()
        Logger.info(f"Class Imbalance Ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 1.5:
            Logger.info("Applying SMOTE...")
            sm = SMOTE(random_state=self.cfg.RANDOM_STATE, k_neighbors=5)
            self.X_res, self.y_res = sm.fit_resample(self.X_scaled, self.y)
        else:
            self.X_res, self.y_res = self.X_scaled, self.y.values
            
    def train(self):
        Logger.section("Training Ensemble Model")
        
        # Define Models
        xgb_clf = xgb.XGBClassifier(
            objective="multi:softprob", num_class=4, n_estimators=600, learning_rate=0.03,
            max_depth=8, subsample=0.85, colsample_bytree=0.85, random_state=self.cfg.RANDOM_STATE, n_jobs=self.cfg.N_JOBS
        )
        
        lgb_clf = lgb.LGBMClassifier(
            objective="multiclass", num_class=4, n_estimators=600, learning_rate=0.03,
            num_leaves=80, subsample=0.85, colsample_bytree=0.85, random_state=self.cfg.RANDOM_STATE, n_jobs=self.cfg.N_JOBS, verbose=-1
        )
        
        cat_clf = CatBoostClassifier(
            iterations=600, learning_rate=0.03, depth=8, loss_function="MultiClass",
            random_seed=self.cfg.RANDOM_STATE, verbose=0, thread_count=self.cfg.N_JOBS
        )
        
        self.model = VotingClassifier(
            estimators=[('xgb', xgb_clf), ('lgb', lgb_clf), ('cat', cat_clf)],
            voting='soft', weights=[1.2, 1.0, 1.1], n_jobs=self.cfg.N_JOBS
        )
        
        # Cross-Validation
        kf = StratifiedKFold(n_splits=self.cfg.N_FOLDS, shuffle=True, random_state=self.cfg.RANDOM_STATE)
        scores_macro = []
        scores_weighted = []
        
        print(f"Starting {self.cfg.N_FOLDS}-Fold CV...")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_res, self.y_res), 1):
            X_tr, X_val = self.X_res[train_idx], self.X_res[val_idx]
            y_tr, y_val = self.y_res[train_idx], self.y_res[val_idx]
            
            self.model.fit(X_tr, y_tr)
            preds = self.model.predict(X_val)
            
            f1_m = f1_score(y_val, preds, average='macro')
            f1_w = f1_score(y_val, preds, average='weighted')
            scores_macro.append(f1_m)
            scores_weighted.append(f1_w)
            
            print(f"  Fold {fold}: F1-Macro = {f1_m:.4f} | F1-Weighted = {f1_w:.4f}")
            
        Logger.info(f"Avg F1-Macro: {np.mean(scores_macro):.4f} ± {np.std(scores_macro):.4f}")
        Logger.info(f"Avg F1-Weighted: {np.mean(scores_weighted):.4f} ± {np.std(scores_weighted):.4f}")
        
        # Final Training
        Logger.info("Retraining on full dataset...")
        self.model.fit(self.X_res, self.y_res)
        
        # Save Model
        joblib.dump(self.model, self.cfg.BASE_DIR / "ensemble_model.pkl")
        Logger.info("Model saved.")

    def predict(self):
        if self.X_test is None: return
        
        Logger.section("Generating Submission")
        
        preds = self.model.predict(self.X_test_scaled)
        
        # Prepare IDs
        if self.test_ids is not None:
            ids = self.test_ids
        else:
            # Fallback ID generation matching the original format
            ids = [f"ANS{i+1:05d}" for i in range(len(preds))]
            
        submission = pd.DataFrame({
            "id": ids,
            "task2": preds
        })
        
        sub_path = self.cfg.BASE_DIR / "submission_Task2_v2.csv"
        submission.to_csv(sub_path, index=False)
        
        Logger.info(f"Submission saved to {sub_path}")
        Logger.info(f"Prediction Counts:\n{submission['task2'].value_counts()}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    config = Config()
    pipeline = PlayerSegmentationPipeline(config)
    
    try:
        pipeline.load_data()
        pipeline.preprocess()
        pipeline.train()
        pipeline.predict()
        
        Logger.section("Pipeline Complete")
        print("✅ Task 2 V2 Optimized Finished Successfully.")
        
    except Exception as e:
        Logger.error(f"Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()
