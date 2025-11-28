# ============================================================
# Task 1 — V3.2 OPTIMIZED: Modular & Efficient Pipeline
# Improvements:
# - Object-Oriented Structure for better maintainability
# - Automatic Environment Detection (Colab vs Local)
# - Optimized Feature Engineering & Selection
# - Enhanced Error Handling & Logging
# - Configurable Paths
# ============================================================

import os
import sys
import copy
import joblib
import numpy as np
import pandas as pd
import warnings
import time
from pathlib import Path
from datetime import datetime

# ML Libraries
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import ADASYN, SMOTE

# Boosting Libraries
from lightgbm import LGBMClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    # Paths
    IS_COLAB = 'google.colab' in sys.modules
    
    if IS_COLAB:
        BASE_DIR = Path("/content/drive/MyDrive/ML-Task1-V3-Optimized")
        DATA_DIR = Path("/content/task1")
    else:
        # Local paths - adjust as needed
        BASE_DIR = Path("./output")
        DATA_DIR = Path("./data") # Assumes data is in a 'data' folder relative to script
        
    TRAIN_PATH = DATA_DIR / "train.csv"
    TEST_PATH = DATA_DIR / "test.csv"
    
    # Model Parameters
    N_FOLDS = 5
    RANDOM_STATE = 42
    N_JOBS = -1
    
    # Feature Selection
    TOP_N_FEATURES = 60
    
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

def sanitize_for_pickle(model_obj):
    """Cleans model objects for pickling by removing unpicklable attributes."""
    try:
        for attr in ['callbacks', '_callbacks', 'learning_rates', '_callbacks_list']:
            if hasattr(model_obj, attr):
                try: setattr(model_obj, attr, None)
                except: pass
        if hasattr(model_obj, "__dict__"):
            for k, v in list(model_obj.__dict__.items()):
                if callable(v) or ('<function' in repr(v)) or ('<lambda' in repr(v)):
                    try: setattr(model_obj, k, None)
                    except: pass
    except:
        pass
    return model_obj

# ============================================================
# CORE PIPELINE
# ============================================================
class CheaterDetectionPipeline:
    def __init__(self, config):
        self.cfg = config
        self.models = {}
        self.selected_features = []
        self.best_weights = None
        self.best_threshold = 0.5
        self.meta_model = None
        
    def load_data(self):
        Logger.section("Loading Data")
        
        # Check if files exist
        if not self.cfg.TRAIN_PATH.exists():
            Logger.error(f"Train file not found at {self.cfg.TRAIN_PATH}")
            # Fallback for local testing if files are in current dir
            if Path("train.csv").exists():
                self.cfg.TRAIN_PATH = Path("train.csv")
                Logger.info(f"Found train.csv in current directory")
            else:
                raise FileNotFoundError(f"Could not find train.csv")

        if not self.cfg.TEST_PATH.exists():
            if Path("test.csv").exists():
                self.cfg.TEST_PATH = Path("test.csv")
            else:
                Logger.warn(f"Test file not found at {self.cfg.TEST_PATH}")

        train = pd.read_csv(self.cfg.TRAIN_PATH)
        test = pd.read_csv(self.cfg.TEST_PATH) if self.cfg.TEST_PATH.exists() else None
        
        mask = train["is_cheater"].notna()
        train = train[mask].reset_index(drop=True)
        
        self.y_train = train["is_cheater"].astype(int).values
        self.X_train = train.drop(columns=["id", "player_id", "is_cheater"], errors='ignore')
        
        if test is not None:
            self.X_test = test.drop(columns=["id", "player_id"], errors='ignore')
            self.test_ids = test["id"]
        else:
            self.X_test = None
            
        Logger.info(f"Train shape: {self.X_train.shape}")
        if self.X_test is not None:
            Logger.info(f"Test shape: {self.X_test.shape}")
        Logger.info(f"Cheater rate: {self.y_train.mean():.2%}")

    def _add_missing_indicators(self, df):
        """Adds indicators for strategically missing values."""
        strategic_cols = ['reports_received', 'device_changes_count', 'account_age_days']
        for col in strategic_cols:
            if col in df.columns:
                df[f'{col}_is_missing'] = df[col].isna().astype(int)
        return df

    def _impute_data(self, X_train, X_test):
        """Neutral imputation strategy."""
        X_train_imp = X_train.copy()
        X_test_imp = X_test.copy() if X_test is not None else None
        
        # 1. KNN for performance metrics (correlated)
        perf_cols = [c for c in ['kill_death_ratio', 'headshot_percentage', 'accuracy_score', 
                               'win_rate', 'kill_consistency', 'damage_per_round'] 
                    if c in X_train.columns]
        
        if perf_cols:
            knn = KNNImputer(n_neighbors=15, weights='distance')
            X_train_imp[perf_cols] = knn.fit_transform(X_train[perf_cols])
            if X_test_imp is not None:
                X_test_imp[perf_cols] = knn.transform(X_test[perf_cols])
                
        # 2. Median for others (neutral)
        numeric_cols = X_train_imp.select_dtypes(include=[np.number]).columns
        remaining_cols = [c for c in numeric_cols if c not in perf_cols]
        
        if remaining_cols:
            med = SimpleImputer(strategy='median')
            X_train_imp[remaining_cols] = med.fit_transform(X_train[remaining_cols])
            if X_test_imp is not None:
                X_test_imp[remaining_cols] = med.transform(X_test[remaining_cols])
                
        return X_train_imp, X_test_imp

    def _engineer_features(self, df):
        """Creates enhanced features efficiently."""
        df = df.copy()
        
        # Helper to safely get columns
        def get_col(name, default=0):
            return df[name] if name in df.columns else default
            
        # Vectorized feature creation
        acc = get_col("accuracy_score")
        hs = get_col("headshot_percentage")
        kd = get_col("kill_death_ratio")
        reports = get_col("reports_received")
        crosshair = get_col("crosshair_placement")
        
        # Interaction features
        df["aim_efficiency"] = acc * hs / 100
        df["kill_effectiveness"] = kd * hs / 100
        df["reports_x_crosshair"] = reports * crosshair
        df["reports_x_hs"] = reports * hs / 10
        df["reports_x_kd"] = reports * kd
        
        # Boolean flags (as int)
        df["superhuman_aim"] = ((acc > 75) & (hs > 75)).astype(int)
        df["extreme_performance"] = ((kd > 8) & (hs > 70)).astype(int)
        df["high_risk_pattern"] = ((reports > 18) & (get_col("device_changes_count") > 12)).astype(int)
        
        # Squared features for non-linear effects
        df["reports_squared"] = reports ** 2
        df["hs_squared"] = (hs / 100) ** 2
        
        # Aggregate flags
        flag_cols = [c for c in df.columns if 'flag' in c or c in 
                    ['superhuman_aim', 'extreme_performance', 'high_risk_pattern']]
        if flag_cols:
            df["total_red_flags"] = df[flag_cols].sum(axis=1)
            
        return df

    def preprocess(self):
        Logger.section("Preprocessing & Feature Engineering")
        
        # 1. Missing Indicators
        self.X_train = self._add_missing_indicators(self.X_train)
        if self.X_test is not None:
            self.X_test = self._add_missing_indicators(self.X_test)
            
        # 2. Imputation
        self.X_train, self.X_test = self._impute_data(self.X_train, self.X_test)
        Logger.info("Applied neutral imputation")
        
        # 3. Feature Engineering
        self.X_train = self._engineer_features(self.X_train)
        if self.X_test is not None:
            self.X_test = self._engineer_features(self.X_test)
        Logger.info(f"Total features after engineering: {self.X_train.shape[1]}")

    def select_features(self):
        Logger.section("Feature Selection")
        
        # Correlation
        corr = pd.concat([self.X_train, pd.Series(self.y_train, name="target")], axis=1).corr()["target"].abs()
        high_corr = corr[corr > 0.01].index.drop("target").tolist()
        
        # Tree-based importance
        tree = ExtraTreesClassifier(n_estimators=100, max_depth=10, n_jobs=self.cfg.N_JOBS, random_state=self.cfg.RANDOM_STATE)
        tree.fit(self.X_train, self.y_train)
        importances = pd.Series(tree.feature_importances_, index=self.X_train.columns)
        top_imp = importances.nlargest(self.cfg.TOP_N_FEATURES).index.tolist()
        
        # Variance
        var_sel = VarianceThreshold(threshold=0.01)
        var_sel.fit(self.X_train)
        high_var = self.X_train.columns[var_sel.get_support()].tolist()
        
        # Intersection
        self.selected_features = list(set(high_corr) & set(top_imp) & set(high_var))
        
        # Force include critical features
        critical = ['reports_received', 'crosshair_placement', 'headshot_percentage', 
                   'kill_death_ratio', 'total_red_flags']
        for c in critical:
            if c in self.X_train.columns and c not in self.selected_features:
                self.selected_features.append(c)
                
        # Fallback
        if not self.selected_features:
            self.selected_features = top_imp[:30]
            
        Logger.info(f"Selected {len(self.selected_features)} features")
        
        # Filter data
        self.X_train = self.X_train[self.selected_features]
        if self.X_test is not None:
            self.X_test = self.X_test[self.selected_features]

    def train_models(self):
        Logger.section("Training Models (5-Fold)")
        
        kf = StratifiedKFold(n_splits=self.cfg.N_FOLDS, shuffle=True, random_state=self.cfg.RANDOM_STATE)
        
        # Initialize containers
        self.val_preds = np.zeros((len(self.X_train), 5)) # 5 models
        self.test_preds = np.zeros((len(self.X_test), 5)) if self.X_test is not None else None
        
        scale_pos = np.sum(self.y_train == 0) / np.sum(self.y_train == 1)
        
        # Model Definitions
        model_defs = {
            'rf': RandomForestClassifier(n_estimators=300, max_depth=12, class_weight='balanced_subsample', n_jobs=self.cfg.N_JOBS, random_state=self.cfg.RANDOM_STATE),
            'xgb': xgb.XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=6, scale_pos_weight=scale_pos*0.65, n_jobs=self.cfg.N_JOBS, random_state=self.cfg.RANDOM_STATE, eval_metric="logloss"),
            'lgb': LGBMClassifier(n_estimators=800, learning_rate=0.05, max_depth=7, scale_pos_weight=scale_pos*0.7, random_state=self.cfg.RANDOM_STATE, verbose=-1),
            'cat': CatBoostClassifier(iterations=800, learning_rate=0.05, depth=7, class_weights=[1.0, scale_pos*0.65], verbose=0, random_seed=self.cfg.RANDOM_STATE),
            'xgb2': xgb.XGBClassifier(n_estimators=800, learning_rate=0.04, max_depth=7, scale_pos_weight=scale_pos*0.75, n_jobs=self.cfg.N_JOBS, random_state=123, eval_metric="logloss")
        }
        
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train, self.y_train), 1):
            print(f"\n📄 Fold {fold}/{self.cfg.N_FOLDS}")
            
            X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr, y_val = self.y_train[train_idx], self.y_train[val_idx]
            
            # Resampling (ADASYN)
            try:
                resampler = ADASYN(sampling_strategy=0.55, random_state=self.cfg.RANDOM_STATE, n_neighbors=8)
                X_tr_res, y_tr_res = resampler.fit_resample(X_tr, y_tr)
            except:
                resampler = SMOTE(sampling_strategy=0.55, random_state=self.cfg.RANDOM_STATE)
                X_tr_res, y_tr_res = resampler.fit_resample(X_tr, y_tr)
                
            # Scaling
            scaler = RobustScaler()
            X_tr_scaled = scaler.fit_transform(X_tr_res)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(self.X_test) if self.X_test is not None else None
            
            # Train each model
            for idx, (name, model) in enumerate(model_defs.items()):
                print(f"  > {name.upper()}", end="... ")
                m = copy.deepcopy(model)
                
                try:
                    if name in ['xgb', 'xgb2']:
                        m.fit(X_tr_scaled, y_tr_res, eval_set=[(X_val_scaled, y_val)], 
                              callbacks=[xgb.callback.EarlyStopping(rounds=50)], verbose=False)
                    elif name == 'lgb':
                        m.fit(X_tr_scaled, y_tr_res, eval_set=[(X_val_scaled, y_val)], 
                              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)], verbose=False)
                    elif name == 'cat':
                        m.fit(X_tr_scaled, y_tr_res, eval_set=(X_val_scaled, y_val), early_stopping_rounds=50, verbose=False)
                    else:
                        m.fit(X_tr_scaled, y_tr_res)
                        
                    # Predict
                    self.val_preds[val_idx, idx] = m.predict_proba(X_val_scaled)[:, 1]
                    if self.test_preds is not None:
                        self.test_preds[:, idx] += m.predict_proba(X_test_scaled)[:, 1] / self.cfg.N_FOLDS
                        
                    # Save model
                    joblib.dump(sanitize_for_pickle(m), self.cfg.BASE_DIR / f"{name}_fold{fold}.pkl")
                    print("✓")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
            
            # Evaluate Fold
            fold_pred = self.val_preds[val_idx].mean(axis=1)
            best_f2 = 0
            for t in np.linspace(0.3, 0.7, 41):
                f2 = fbeta_score(y_val, (fold_pred >= t).astype(int), beta=2)
                if f2 > best_f2: best_f2 = f2
            
            fold_scores.append(best_f2)
            print(f"  ★ Fold F2 Score: {best_f2:.4f}")
            
        Logger.info(f"Mean F2 Score: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
        
        # Save predictions
        np.save(self.cfg.BASE_DIR / "val_preds.npy", self.val_preds)
        if self.test_preds is not None:
            np.save(self.cfg.BASE_DIR / "test_preds.npy", self.test_preds)

    def optimize_ensemble(self):
        Logger.section("Ensemble Optimization")
        
        # 1. Weighted Average Optimization
        weights_options = [
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.15, 0.25, 0.25, 0.2, 0.15],
            [0.1, 0.3, 0.3, 0.2, 0.1]
        ]
        
        best_w_f2 = 0
        best_w_thresh = 0.5
        
        for w in weights_options:
            weighted_pred = np.average(self.val_preds, axis=1, weights=w)
            for t in np.linspace(0.3, 0.7, 81):
                f2 = fbeta_score(self.y_train, (weighted_pred >= t).astype(int), beta=2)
                if f2 > best_w_f2:
                    best_w_f2 = f2
                    best_w_thresh = t
                    self.best_weights = w
                    
        Logger.info(f"Best Weighted F2: {best_w_f2:.4f} (Threshold: {best_w_thresh:.3f})")
        
        # 2. Meta-Model (Stacking)
        self.meta_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", use_label_encoder=False, random_state=42
        )
        self.meta_model.fit(self.val_preds, self.y_train)
        
        meta_pred = self.meta_model.predict_proba(self.val_preds)[:, 1]
        best_m_f2 = 0
        best_m_thresh = 0.5
        
        for t in np.linspace(0.3, 0.7, 81):
            f2 = fbeta_score(self.y_train, (meta_pred >= t).astype(int), beta=2)
            if f2 > best_m_f2:
                best_m_f2 = f2
                best_m_thresh = t
                
        Logger.info(f"Best Meta-Model F2: {best_m_f2:.4f} (Threshold: {best_m_thresh:.3f})")
        
        # Decision
        if best_m_f2 > best_w_f2:
            self.use_meta = True
            self.best_threshold = best_m_thresh
            Logger.info("Selected: Meta-Model")
        else:
            self.use_meta = False
            self.best_threshold = best_w_thresh
            Logger.info("Selected: Weighted Average")

    def generate_submission(self):
        if self.X_test is None: return
        
        Logger.section("Generating Submission")
        
        # Get probabilities
        if self.use_meta:
            final_probs = self.meta_model.predict_proba(self.test_preds)[:, 1]
        else:
            final_probs = np.average(self.test_preds, axis=1, weights=self.best_weights)
            
        # Conservative Threshold Adjustment
        expected_rate = self.y_train.mean()
        sorted_probs = np.sort(final_probs)[::-1]
        target_idx = int(len(final_probs) * expected_rate)
        dist_thresh = sorted_probs[min(target_idx, len(sorted_probs)-1)]
        
        # Blend thresholds (Bias towards distribution for safety)
        final_thresh = (self.best_threshold * 0.4) + (dist_thresh * 0.6)
        final_thresh = np.clip(final_thresh, 0.35, 0.65)
        
        predictions = (final_probs >= final_thresh).astype(int)
        pred_rate = predictions.mean()
        
        Logger.info(f"Final Threshold: {final_thresh:.3f}")
        Logger.info(f"Predicted Rate: {pred_rate:.2%} (Expected: {expected_rate:.2%})")
        
        # Save
        sub = pd.DataFrame({"id": self.test_ids, "task1": predictions})
        sub_path = self.cfg.BASE_DIR / "submission_optimized.csv"
        sub.to_csv(sub_path, index=False)
        Logger.info(f"Saved to {sub_path}")

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    config = Config()
    pipeline = CheaterDetectionPipeline(config)
    
    try:
        pipeline.load_data()
        pipeline.preprocess()
        pipeline.select_features()
        pipeline.train_models()
        pipeline.optimize_ensemble()
        pipeline.generate_submission()
        
        Logger.section("Done")
        print("✅ Pipeline completed successfully.")
        
    except Exception as e:
        Logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
