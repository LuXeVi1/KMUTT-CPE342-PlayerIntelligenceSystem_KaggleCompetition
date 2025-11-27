# ============================================================
# Task 1 — IMPROVED Stacking Pipeline with Advanced Features
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import fbeta_score, roc_auc_score, precision_recall_curve
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1) SETUP
# ============================================================
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

DRIVE_FOLDER = "/content/drive/MyDrive/ML-Task1-IMPROVED-STACKING"
os.makedirs(DRIVE_FOLDER, exist_ok=True)

TRAIN_PATH = "/content/task1/train.csv"
TEST_PATH = "/content/task1/test.csv"

print("=" * 80)
print("🚀 IMPROVED STACKING PIPELINE")
print("=" * 80)
print(f"Saving to: {DRIVE_FOLDER}\n")

# ============================================================
# 2) LOAD DATA
# ============================================================
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

mask = train["is_cheater"].notna()
train = train[mask].reset_index(drop=True)

y_all = train["is_cheater"].astype(int).values
X_all = train.drop(columns=["id", "player_id", "is_cheater"])
X_test = test.drop(columns=["id", "player_id"])

print(f"✓ Train shape: {X_all.shape}")
print(f"✓ Test shape: {X_test.shape}")
print(f"✓ Class distribution: {np.bincount(y_all)}\n")

# ============================================================
# 3) ADVANCED FEATURE ENGINEERING
# ============================================================
print("=" * 80)
print("🛠️ ADVANCED FEATURE ENGINEERING")
print("=" * 80)

def create_advanced_features(df):
    """Create comprehensive cheat detection features"""
    c = df.columns
    
    # A. SUPERHUMAN PERFORMANCE
    if all(x in c for x in ["accuracy_score", "headshot_percentage"]):
        df["aim_efficiency"] = df["accuracy_score"] * df["headshot_percentage"] / 100
    
    if all(x in c for x in ["kill_death_ratio", "headshot_percentage"]):
        df["kill_effectiveness"] = df["kill_death_ratio"] * df["headshot_percentage"] / 100
    
    if all(x in c for x in ["damage_per_round", "kill_death_ratio", "win_rate"]):
        df["combat_dominance"] = df["damage_per_round"] * df["kill_death_ratio"] * df["win_rate"] / 10000
    
    if all(x in c for x in ["aiming_smoothness", "spray_control_score", "crosshair_placement"]):
        df["mechanical_perfection"] = df["aiming_smoothness"] * df["spray_control_score"] * df["crosshair_placement"]
    
    if all(x in c for x in ["accuracy_score", "headshot_percentage", "crosshair_placement"]):
        df["aim_perfection_score"] = df["accuracy_score"] * df["headshot_percentage"] * df["crosshair_placement"] / 10000
    
    # B. INCONSISTENCY DETECTION
    if all(x in c for x in ["game_sense_score", "level"]):
        df["skill_vs_experience"] = df["game_sense_score"] / (df["level"] / 100 + 0.01)
    
    if all(x in c for x in ["win_rate", "account_age_days"]):
        df["perf_vs_age"] = df["win_rate"] / (df["account_age_days"] / 365 + 0.1)
    
    if all(x in c for x in ["headshot_percentage", "kill_consistency"]):
        df["hs_consistency_anomaly"] = df["headshot_percentage"] / (df["kill_consistency"] + 0.01)
    
    if all(x in c for x in ["kill_death_ratio", "win_rate", "account_age_days"]):
        df["skill_explosion"] = (df["kill_death_ratio"] * df["win_rate"]) / (df["account_age_days"] + 10)
    
    # C. BEHAVIORAL ANOMALIES
    if all(x in c for x in ["reports_received", "device_changes_count"]):
        df["report_severity"] = df["reports_received"] * df["device_changes_count"]
    
    if all(x in c for x in ["sessions_per_day", "avg_session_length_min", "communication_rate"]):
        df["grind_pattern"] = df["sessions_per_day"] * df["avg_session_length_min"] * (1 - df["communication_rate"])
    
    if all(x in c for x in ["night_play_ratio", "sessions_per_day", "team_play_score"]):
        df["night_farming"] = df["night_play_ratio"] * df["sessions_per_day"] * (1 - df["team_play_score"])
    
    if all(x in c for x in ["kill_death_ratio", "win_rate", "friend_network_size", "communication_rate"]):
        df["antisocial_elite"] = (df["kill_death_ratio"] * df["win_rate"]) / ((df["friend_network_size"] + 1) * (df["communication_rate"] + 0.1))
    
    if all(x in c for x in ["device_changes_count", "level_progression_speed", "reports_received"]):
        df["device_hopping"] = df["device_changes_count"] * df["level_progression_speed"] * (df["reports_received"] + 1)
    
    # D. SKILL IMBALANCES
    if all(x in c for x in ["aiming_smoothness", "crosshair_placement", "movement_pattern_score"]):
        df["aim_movement_disconnect"] = np.abs((df["aiming_smoothness"] + df["crosshair_placement"]) / 2 - df["movement_pattern_score"])
    
    if all(x in c for x in ["aiming_smoothness", "spray_control_score", "game_sense_score", "map_knowledge"]):
        df["mech_strat_imbalance"] = np.abs((df["aiming_smoothness"] + df["spray_control_score"]) / 2 - (df["game_sense_score"] + df["map_knowledge"]) / 2)
    
    if all(x in c for x in ["clutch_success_rate", "team_play_score"]):
        df["clutch_teamplay_paradox"] = df["clutch_success_rate"] / (df["team_play_score"] + 0.1)
    
    # E. EXTREME PERFORMANCE FLAGS
    if "headshot_percentage" in c:
        df["extreme_hs"] = (df["headshot_percentage"] > 70).astype(int)
    if "kill_death_ratio" in c:
        df["extreme_kd"] = (df["kill_death_ratio"] > 8).astype(int)
    if "accuracy_score" in c:
        df["extreme_accuracy"] = (df["accuracy_score"] > 70).astype(int)
    if "win_rate" in c:
        df["extreme_win"] = (df["win_rate"] > 85).astype(int)
    if "reports_received" in c:
        df["high_reports"] = (df["reports_received"] > 15).astype(int)
    if "account_age_days" in c:
        df["new_account"] = (df["account_age_days"] < 30).astype(int)
    
    flag_cols = ["extreme_hs", "extreme_kd", "extreme_accuracy", "extreme_win", "high_reports", "new_account"]
    available_flags = [col for col in flag_cols if col in df.columns]
    if available_flags:
        df["total_suspicion_flags"] = df[available_flags].sum(axis=1)
    
    # F. INTERACTION FEATURES (Top correlations)
    if all(x in c for x in ["reports_received", "crosshair_placement"]):
        df["reports_x_crosshair"] = df["reports_received"] * df["crosshair_placement"]
    
    if all(x in c for x in ["reports_received", "headshot_percentage"]):
        df["reports_x_hs"] = df["reports_received"] * df["headshot_percentage"] / 10
    
    if all(x in c for x in ["reports_received", "kill_death_ratio"]):
        df["reports_x_kd"] = df["reports_received"] * df["kill_death_ratio"]
    
    if all(x in c for x in ["crosshair_placement", "accuracy_score"]):
        df["crosshair_x_accuracy"] = df["crosshair_placement"] * df["accuracy_score"] / 100
    
    if all(x in c for x in ["game_sense_score", "map_knowledge"]):
        df["gamesense_x_map"] = df["game_sense_score"] * df["map_knowledge"]
    
    # G. POLYNOMIAL FEATURES
    if "reports_received" in c:
        df["reports_squared"] = df["reports_received"] ** 2
    if "headshot_percentage" in c:
        df["hs_squared"] = (df["headshot_percentage"] / 100) ** 2
    if "kill_death_ratio" in c:
        df["kd_squared"] = (df["kill_death_ratio"] / 12) ** 2
    
    # H. RATIO FEATURES
    if all(x in c for x in ["damage_per_round", "kill_death_ratio"]):
        df["kill_efficiency"] = df["damage_per_round"] / (df["kill_death_ratio"] + 0.1)
    
    if all(x in c for x in ["survival_time_avg", "damage_per_round"]):
        df["survival_effectiveness"] = df["survival_time_avg"] * df["damage_per_round"] / 1000
    
    if all(x in c for x in ["utility_usage_rate", "game_sense_score"]):
        df["utility_mastery"] = df["utility_usage_rate"] * df["game_sense_score"]
    
    return df

X_all = create_advanced_features(X_all)
X_test = create_advanced_features(X_test)

print(f"✓ Features after engineering: {X_all.shape[1]}\n")

# ============================================================
# 4) IMPUTATION
# ============================================================
print("=" * 80)
print("🔧 MISSING VALUE IMPUTATION")
print("=" * 80)

numeric_cols = X_all.select_dtypes(include=np.number).columns.tolist()
imputer = SimpleImputer(strategy="median")

X_all[numeric_cols] = imputer.fit_transform(X_all[numeric_cols])
X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])

print(f"✓ Imputed {len(numeric_cols)} numeric features\n")

# ============================================================
# 5) IMPROVED FEATURE SELECTION
# ============================================================
print("=" * 80)
print("🎯 ADVANCED FEATURE SELECTION")
print("=" * 80)

# Correlation-based selection (threshold = 0.01)
corr = pd.concat([X_all, pd.Series(y_all, name="is_cheater")], axis=1).corr()["is_cheater"].abs()
high_corr_features = corr[corr > 0.01].index.drop("is_cheater").tolist()

# Tree-based importance
from sklearn.ensemble import ExtraTreesClassifier
tree_selector = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
tree_selector.fit(X_all, y_all)
importances = pd.Series(tree_selector.feature_importances_, index=X_all.columns)
top_importance_features = importances.nlargest(50).index.tolist()

# Combine both methods
selected_features = list(set(high_corr_features) | set(top_importance_features))

X_all_selected = X_all[selected_features]
X_test_selected = X_test[selected_features]

print(f"✓ Selected {len(selected_features)} features")
print(f"  - From correlation: {len(high_corr_features)}")
print(f"  - From tree importance: {len(top_importance_features)}\n")

# Save feature names
np.save(os.path.join(DRIVE_FOLDER, "selected_features.npy"), np.array(selected_features))

# ============================================================
# 6) STRATIFIED K-FOLD WITH DIVERSE BASE MODELS
# ============================================================
print("=" * 80)
print("🤖 TRAINING DIVERSE BASE MODELS (5-FOLD)")
print("=" * 80)

N_FOLDS = 5
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Use moderate SMOTE (0.55 ratio)
smote_tomek = SMOTETomek(sampling_strategy=0.55, random_state=42)

# Prepare arrays for stacking
val_preds_stack = np.zeros((len(X_all_selected), 5))  # 5 models
test_preds_stack = np.zeros((len(X_test_selected), 5))

# Calculate scale_pos_weight
scale_pos = np.sum(y_all == 0) / np.sum(y_all == 1)

# Define diverse base models
models = {
    'rf': RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ),
    'xgb': xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=7,
        subsample=0.75,
        colsample_bytree=0.75,
        gamma=0.3,
        reg_alpha=1.0,
        reg_lambda=3.0,
        scale_pos_weight=scale_pos * 0.6,
        random_state=42,
        tree_method='hist'
    ),
    'lgb': LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=20,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_samples=50,
        reg_alpha=1.0,
        reg_lambda=3.0,
        scale_pos_weight=scale_pos * 0.65,
        random_state=42,
        verbose=-1
    ),
    'cat': CatBoostClassifier(
        iterations=400,
        learning_rate=0.05,
        depth=5,
        l2_leaf_reg=7,
        subsample=0.75,
        class_weights=[1.0, scale_pos * 0.6],
        verbose=0,
        random_seed=42
    ),
    'gbm': GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.75,
        random_state=42
    )
}

fold_f2_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all_selected, y_all), 1):
    print(f"\n{'─' * 80}")
    print(f"📊 Fold {fold}/{N_FOLDS}")
    print(f"{'─' * 80}")
    
    X_tr, X_val = X_all_selected.iloc[train_idx], X_all_selected.iloc[val_idx]
    y_tr, y_val = y_all[train_idx], y_all[val_idx]
    
    # Apply SMOTE
    X_tr_res, y_tr_res = smote_tomek.fit_resample(X_tr, y_tr)
    print(f"After SMOTETomek: {len(y_tr_res)} samples")
    
    # Scale features
    scaler = RobustScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_res)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Train each model
    for idx, (name, model) in enumerate(models.items()):
        print(f"  Training {name.upper()}...", end=" ")
        
        model_fold = clone(model)
        
        if name == 'lgb':
            model_fold.fit(
                X_tr_scaled, y_tr_res,
                eval_set=[(X_val_scaled, y_val)],
                callbacks=[early_stopping(30), log_evaluation(0)]
            )
        elif name == 'cat':
            model_fold.fit(
                X_tr_scaled, y_tr_res,
                eval_set=Pool(X_val_scaled, y_val),
                early_stopping_rounds=30,
                verbose=0
            )
        else:
            model_fold.fit(X_tr_scaled, y_tr_res)
        
        # Save model
        joblib.dump(model_fold, os.path.join(DRIVE_FOLDER, f"{name}_fold{fold}.pkl"))
        
        # Predictions
        val_preds_stack[val_idx, idx] = model_fold.predict_proba(X_val_scaled)[:, 1]
        test_preds_stack[:, idx] += model_fold.predict_proba(X_test_scaled)[:, 1] / N_FOLDS
        
        print("✓")
    
    # Calculate fold F2 with simple averaging
    fold_pred_proba = val_preds_stack[val_idx].mean(axis=1)
    fold_pred = (fold_pred_proba >= 0.5).astype(int)
    fold_f2 = fbeta_score(y_val, fold_pred, beta=2)
    fold_f2_scores.append(fold_f2)
    print(f"  Fold F2 (simple avg): {fold_f2:.4f}")

print(f"\n{'=' * 80}")
print(f"Mean Fold F2: {np.mean(fold_f2_scores):.4f} ± {np.std(fold_f2_scores):.4f}")
print(f"{'=' * 80}\n")

# Save stacked predictions
np.save(os.path.join(DRIVE_FOLDER, "val_preds_stack.npy"), val_preds_stack)
np.save(os.path.join(DRIVE_FOLDER, "test_preds_stack.npy"), test_preds_stack)

# ============================================================
# 7) ADVANCED META-MODEL
# ============================================================
print("=" * 80)
print("🧠 TRAINING META-MODEL")
print("=" * 80)

# Try multiple meta-models
meta_models = {
    'logistic': LogisticRegression(
        C=0.1,
        max_iter=2000,
        random_state=42,
        class_weight='balanced'
    ),
    'ridge': RidgeClassifier(
        alpha=1.0,
        random_state=42,
        class_weight='balanced'
    ),
    'xgb': xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        scale_pos_weight=scale_pos * 0.7,
        random_state=42
    )
}

best_meta_name = None
best_meta_model = None
best_meta_f2 = 0

for name, meta in meta_models.items():
    meta.fit(val_preds_stack, y_all)
    
    if hasattr(meta, 'predict_proba'):
        val_pred_proba = meta.predict_proba(val_preds_stack)[:, 1]
    else:
        val_pred_proba = meta.decision_function(val_preds_stack)
        val_pred_proba = (val_pred_proba - val_pred_proba.min()) / (val_pred_proba.max() - val_pred_proba.min())
    
    # Find best threshold
    best_t = 0.5
    best_f2 = 0
    for t in np.linspace(0.3, 0.7, 41):
        pred = (val_pred_proba >= t).astype(int)
        f2 = fbeta_score(y_all, pred, beta=2)
        if f2 > best_f2:
            best_f2 = f2
            best_t = t
    
    print(f"  {name}: F2 = {best_f2:.4f} @ threshold = {best_t:.3f}")
    
    if best_f2 > best_meta_f2:
        best_meta_f2 = best_f2
        best_meta_model = meta
        best_meta_name = name
        best_threshold = best_t

print(f"\n✓ Best meta-model: {best_meta_name.upper()}")
print(f"✓ Best F2: {best_meta_f2:.4f}")
print(f"✓ Best threshold: {best_threshold:.3f}\n")

# Save best meta-model
joblib.dump(best_meta_model, os.path.join(DRIVE_FOLDER, "meta_model_best.pkl"))

# ============================================================
# 8) CALIBRATED PREDICTIONS
# ============================================================
print("=" * 80)
print("🔮 GENERATING CALIBRATED PREDICTIONS")
print("=" * 80)

if hasattr(best_meta_model, 'predict_proba'):
    test_prob = best_meta_model.predict_proba(test_preds_stack)[:, 1]
else:
    test_prob = best_meta_model.decision_function(test_preds_stack)
    test_prob = (test_prob - test_prob.min()) / (test_prob.max() - test_prob.min())

# Strategy: Match expected distribution
expected_cheat_rate = np.mean(y_all)
sorted_proba = np.sort(test_prob)[::-1]
target_count = int(len(test_prob) * expected_cheat_rate)
distribution_threshold = sorted_proba[target_count] if target_count < len(sorted_proba) else 0.5

# Use max of tuned threshold and distribution threshold
final_threshold = max(best_threshold, distribution_threshold, best_threshold * 1.1)

test_pred = (test_prob >= final_threshold).astype(int)

predicted_rate = test_pred.sum() / len(test_pred)

print(f"✓ Tuned threshold: {best_threshold:.3f}")
print(f"✓ Distribution threshold: {distribution_threshold:.3f}")
print(f"✓ Final threshold: {final_threshold:.3f}")
print(f"✓ Expected cheat rate: {expected_cheat_rate:.2%}")
print(f"✓ Predicted cheat rate: {predicted_rate:.2%}")
print(f"✓ Deviation: {abs(predicted_rate - expected_cheat_rate):.2%}\n")

# ============================================================
# 9) CREATE SUBMISSION
# ============================================================
print("=" * 80)
print("💾 CREATING SUBMISSION")
print("=" * 80)

submission = pd.DataFrame({
    "id": test["id"],
    "task1": test_pred
})

sub_path = os.path.join(DRIVE_FOLDER, "submission_task1_improved_stacking.csv")
submission.to_csv(sub_path, index=False)

print(f"✓ Submission saved to: {sub_path}")
print(f"\n📊 Prediction distribution:")
print(submission['task1'].value_counts())
print(f"\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)