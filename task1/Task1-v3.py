# ============================================================
# Task 1 — V2 FIXED: Reduced False Positives
# แก้ไขปัญหา: ทาย cheater มากเกินไป + missing value bias
# ============================================================

import os
import copy
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import xgboost as xgb

from imblearn.over_sampling import ADASYN

# ============================================================
# 1) SETUP
# ============================================================
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

DRIVE_FOLDER = "/content/drive/MyDrive/ML-Task1-V2-FIXED"
os.makedirs(DRIVE_FOLDER, exist_ok=True)

TRAIN_PATH = "/content/task1/train.csv"
TEST_PATH = "/content/task1/test.csv"

print("=" * 80)
print("🚀 V2 FIXED - REDUCED FALSE POSITIVES & MISSING VALUE BIAS")
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
print(f"✓ Class distribution: {np.bincount(y_all)}")
print(f"✓ Cheater rate: {y_all.mean():.2%}\n")

# ============================================================
# 3) 🔧 FIX 1: SMARTER MISSING VALUE INDICATORS
# ============================================================
print("=" * 80)
print("🔍 FIX 1: SELECTIVE MISSING VALUE INDICATORS")
print("=" * 80)

def add_selective_missing_indicators(df, train_y=None):
    """
    เพิ่ม missing indicators เฉพาะ features ที่ missing มีความสัมพันธ์กับ cheater
    """
    # Features ที่ missing อาจบ่งบอกพฤติกรรมจริง (ไม่ใช่แค่ข้อมูลหาย)
    strategic_missing_cols = [
        'reports_received',      # cheater อาจถูก report บ่อย
        'device_changes_count',  # cheater อาจเปลี่ยน device บ่อย
        'account_age_days',      # account ใหม่อาจน่าสงสัย
    ]
    
    for col in strategic_missing_cols:
        if col in df.columns:
            df[f'{col}_is_missing'] = df[col].isna().astype(int)
    
    return df

X_all = add_selective_missing_indicators(X_all, y_all)
X_test = add_selective_missing_indicators(X_test)

print(f"✓ Added {len([c for c in X_all.columns if '_is_missing' in c])} strategic missing indicators\n")

# ============================================================
# 4) 🔧 FIX 2: NEUTRAL IMPUTATION STRATEGY
# ============================================================
print("=" * 80)
print("🔧 FIX 2: NEUTRAL MISSING VALUE IMPUTATION")
print("=" * 80)

def neutral_imputation(X_train, X_test):
    """
    ใช้ median/mean ที่เป็นกลาง ไม่ให้ missing value มีผลต่อการทาย
    """
    X_train_imp = X_train.copy()
    X_test_imp = X_test.copy()
    
    # Performance metrics - KNN imputer (เพราะมีความสัมพันธ์กัน)
    performance_features = [
        'kill_death_ratio', 'headshot_percentage', 'accuracy_score',
        'win_rate', 'kill_consistency', 'damage_per_round'
    ]
    
    # Behavioral metrics - Median (ค่ากลาง ไม่ลำเอียง)
    behavioral_features = [
        'sessions_per_day', 'avg_session_length_min', 'night_play_ratio',
        'communication_rate', 'team_play_score', 'friend_network_size'
    ]
    
    # Suspicious metrics - Median (ไม่ให้ missing = cheater)
    suspicious_features = [
        'reports_received', 'device_changes_count', 'account_age_days',
        'level', 'level_progression_speed'
    ]
    
    # 1. KNN for performance
    perf_cols = [c for c in performance_features if c in X_train.columns]
    if perf_cols:
        knn_imputer = KNNImputer(n_neighbors=15, weights='distance')
        X_train_imp[perf_cols] = knn_imputer.fit_transform(X_train[perf_cols])
        X_test_imp[perf_cols] = knn_imputer.transform(X_test[perf_cols])
    
    # 2. Median for behavioral (neutral)
    behav_cols = [c for c in behavioral_features if c in X_train.columns]
    if behav_cols:
        median_imputer = SimpleImputer(strategy='median')
        X_train_imp[behav_cols] = median_imputer.fit_transform(X_train[behav_cols])
        X_test_imp[behav_cols] = median_imputer.transform(X_test[behav_cols])
    
    # 3. Median for suspicious (neutral - ไม่ให้ missing = suspicious)
    susp_cols = [c for c in suspicious_features if c in X_train.columns]
    if susp_cols:
        median_imputer = SimpleImputer(strategy='median')
        X_train_imp[susp_cols] = median_imputer.fit_transform(X_train[susp_cols])
        X_test_imp[susp_cols] = median_imputer.transform(X_test[susp_cols])
    
    # 4. Fill remaining with median
    numeric_cols = X_train_imp.select_dtypes(include=[np.number]).columns
    remaining_imputer = SimpleImputer(strategy='median')
    X_train_imp[numeric_cols] = remaining_imputer.fit_transform(X_train_imp[numeric_cols])
    X_test_imp[numeric_cols] = remaining_imputer.transform(X_test_imp[numeric_cols])
    
    return X_train_imp, X_test_imp

X_all, X_test = neutral_imputation(X_all, X_test)
print("✓ Applied neutral imputation strategy\n")

# ============================================================
# 5) ENHANCED FEATURE ENGINEERING
# ============================================================
print("=" * 80)
print("🛠️ ENHANCED FEATURE ENGINEERING")
print("=" * 80)

def create_enhanced_features(df):
    df = df.copy()
    c = df.columns

    # High-precision cheating indicators
    if all(x in c for x in ["accuracy_score", "headshot_percentage"]):
        df["aim_efficiency"] = df["accuracy_score"] * df["headshot_percentage"] / 100
        df["superhuman_aim"] = ((df["accuracy_score"] > 75) & (df["headshot_percentage"] > 75)).astype(int)

    if all(x in c for x in ["kill_death_ratio", "headshot_percentage"]):
        df["kill_effectiveness"] = df["kill_death_ratio"] * df["headshot_percentage"] / 100
        df["extreme_performance"] = ((df["kill_death_ratio"] > 8) & (df["headshot_percentage"] > 70)).astype(int)

    if all(x in c for x in ["aiming_smoothness", "spray_control_score", "crosshair_placement"]):
        df["mechanical_perfection"] = df["aiming_smoothness"] * df["spray_control_score"] * df["crosshair_placement"]
        df["perfect_mechanics"] = ((df["aiming_smoothness"] > 0.92) &
                                   (df["spray_control_score"] > 0.92) &
                                   (df["crosshair_placement"] > 0.92)).astype(int)

    if all(x in c for x in ["game_sense_score", "level"]):
        df["skill_vs_experience"] = df["game_sense_score"] / (df["level"] / 100 + 0.01)
        df["rapid_skill_gain"] = ((df["game_sense_score"] > 0.85) & (df["level"] < 35)).astype(int)

    if all(x in c for x in ["win_rate", "account_age_days"]):
        df["perf_vs_age"] = df["win_rate"] / (df["account_age_days"] / 365 + 0.1)
        df["new_account_domination"] = ((df["win_rate"] > 85) & (df["account_age_days"] < 50)).astype(int)

    if all(x in c for x in ["reports_received", "device_changes_count"]):
        df["report_severity"] = df["reports_received"] * df["device_changes_count"]
        df["high_risk_pattern"] = ((df["reports_received"] > 18) & (df["device_changes_count"] > 12)).astype(int)

    if all(x in c for x in ["reports_received", "crosshair_placement"]):
        df["reports_x_crosshair"] = df["reports_received"] * df["crosshair_placement"]

    if all(x in c for x in ["reports_received", "headshot_percentage"]):
        df["reports_x_hs"] = df["reports_received"] * df["headshot_percentage"] / 10

    if all(x in c for x in ["reports_received", "kill_death_ratio"]):
        df["reports_x_kd"] = df["reports_received"] * df["kill_death_ratio"]

    if all(x in c for x in ["crosshair_placement", "accuracy_score"]):
        df["crosshair_x_accuracy"] = df["crosshair_placement"] * df["accuracy_score"] / 100

    if all(x in c for x in ["game_sense_score", "map_knowledge"]):
        df["strategic_mastery"] = df["game_sense_score"] * df["map_knowledge"]

    if "reports_received" in c:
        df["reports_squared"] = df["reports_received"] ** 2
        df["high_report_flag"] = (df["reports_received"] > 20).astype(int)

    if "headshot_percentage" in c:
        df["hs_squared"] = (df["headshot_percentage"] / 100) ** 2

    if "kill_death_ratio" in c:
        df["kd_squared"] = (df["kill_death_ratio"] / 12) ** 2

    # Flag aggregation
    flag_cols = [col for col in df.columns if '_flag' in col or 'indicator' in col or
                 col in ['superhuman_aim', 'extreme_performance', 'perfect_mechanics', 
                         'high_risk_pattern', 'rapid_skill_gain', 'new_account_domination']]
    if flag_cols:
        df["total_red_flags"] = df[flag_cols].sum(axis=1)

    return df

X_all = create_enhanced_features(X_all)
X_test = create_enhanced_features(X_test)
print(f"✓ Features after engineering: {X_all.shape[1]}\n")

# ============================================================
# 6) IMPROVED FEATURE SELECTION
# ============================================================
print("=" * 80)
print("🎯 ADVANCED FEATURE SELECTION")
print("=" * 80)

corr = pd.concat([X_all, pd.Series(y_all, name="is_cheater")], axis=1).corr()["is_cheater"].abs()
high_corr_features = corr[corr > 0.01].index.drop("is_cheater").tolist()

tree_selector = ExtraTreesClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    random_state=42,
    n_jobs=-1
)
tree_selector.fit(X_all, y_all)
importances = pd.Series(tree_selector.feature_importances_, index=X_all.columns)
top_importance_features = importances.nlargest(60).index.tolist()

var_selector = VarianceThreshold(threshold=0.01)
var_selector.fit(X_all)
high_var_features = X_all.columns[var_selector.get_support()].tolist()

selected_features = list(set(high_corr_features) & set(top_importance_features) & set(high_var_features))

critical_features = [
    'reports_received', 'crosshair_placement', 'headshot_percentage',
    'kill_death_ratio', 'reports_x_crosshair', 'reports_x_hs',
    'game_sense_score', 'account_age_days', 'accuracy_score',
    'total_red_flags'
]
for cf in critical_features:
    if cf in X_all.columns and cf not in selected_features:
        selected_features.append(cf)

if len(selected_features) == 0:
    selected_features = top_importance_features[:35]

X_all_selected = X_all[selected_features]
X_test_selected = X_test[selected_features]

print(f"✓ Selected {len(selected_features)} features\n")
np.save(os.path.join(DRIVE_FOLDER, "selected_features.npy"), np.array(selected_features))

# ============================================================
# 7) OPTIMIZED K-FOLD TRAINING
# ============================================================
print("=" * 80)
print("🤖 TRAINING OPTIMIZED MODELS (5-FOLD)")
print("=" * 80)

N_FOLDS = 5
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Moderate resampling (ไม่มากเกินไป)
resampler = ADASYN(sampling_strategy=0.55, random_state=42, n_neighbors=8)

val_preds_stack = np.zeros((len(X_all_selected), 5))
test_preds_stack = np.zeros((len(X_test_selected), 5))

scale_pos = np.sum(y_all == 0) / np.sum(y_all == 1)

models = {
    'rf': RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    ),
    'xgb': {
        'model': xgb.XGBClassifier(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=6,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.15,
            reg_alpha=3.0,
            reg_lambda=5.0,
            scale_pos_weight=scale_pos * 0.65,
            random_state=42,
            tree_method='hist',
            use_label_encoder=False,
            eval_metric="logloss"
        )
    },
    'lgb': {
        'model': LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=30,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=40,
            reg_alpha=3.0,
            reg_lambda=5.0,
            scale_pos_weight=scale_pos * 0.7,
            random_state=42,
            verbose=-1
        )
    },
    'cat': {
        'model': CatBoostClassifier(
            iterations=800,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=12,
            subsample=0.8,
            class_weights=[1.0, scale_pos * 0.65],
            verbose=0,
            random_seed=42
        )
    },
    'xgb2': {
        'model': xgb.XGBClassifier(
            n_estimators=800,
            learning_rate=0.04,
            max_depth=7,
            min_child_weight=5,
            subsample=0.75,
            colsample_bytree=0.75,
            gamma=0.25,
            reg_alpha=4.0,
            reg_lambda=6.0,
            scale_pos_weight=scale_pos * 0.75,
            random_state=123,
            tree_method='hist',
            use_label_encoder=False,
            eval_metric="logloss"
        )
    }
}

def sanitize_for_pickle(model_obj):
    try:
        for attr in ['callbacks', '_callbacks', 'learning_rates', '_callbacks_list']:
            if hasattr(model_obj, attr):
                try:
                    setattr(model_obj, attr, None)
                except:
                    pass
        if hasattr(model_obj, "__dict__"):
            for k, v in list(model_obj.__dict__.items()):
                if callable(v) or ('<function' in repr(v)) or ('<lambda' in repr(v)):
                    try:
                        setattr(model_obj, k, None)
                    except:
                        pass
    except:
        pass
    return model_obj

fold_f2_scores = []
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all_selected, y_all), 1):
    print(f"\n{'─' * 80}")
    print(f"📊 Fold {fold}/{N_FOLDS}")
    print(f"{'─' * 80}")

    X_tr, X_val = X_all_selected.iloc[train_idx], X_all_selected.iloc[val_idx]
    y_tr, y_val = y_all[train_idx], y_all[val_idx]

    try:
        X_tr_res, y_tr_res = resampler.fit_resample(X_tr, y_tr)
        print(f"After ADASYN: {len(y_tr_res)} samples")
    except:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(sampling_strategy=0.55, random_state=42)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
        print(f"After SMOTE: {len(y_tr_res)} samples")

    scaler = RobustScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_res)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test_selected)

    for idx, (name, model_config) in enumerate(models.items()):
        print(f"  Training {name.upper()}...", end=" ")
        
        model = model_config['model'] if isinstance(model_config, dict) else model_config
        model_fold = copy.deepcopy(model)

        try:
            if name == 'lgb':
                model_fold.fit(
                    X_tr_scaled, y_tr_res,
                    eval_set=[(X_val_scaled, y_val)],
                    early_stopping_rounds=80,
                    verbose=False
                )
            elif name == 'cat':
                model_fold.fit(
                    X_tr_scaled, y_tr_res,
                    eval_set=(X_val_scaled, y_val),
                    early_stopping_rounds=80,
                    verbose=0
                )
            elif 'xgb' in name:
                model_fold.fit(
                    X_tr_scaled, y_tr_res,
                    eval_set=[(X_val_scaled, y_val)],
                    early_stopping_rounds=80,
                    verbose=False
                )
            else:
                model_fold.fit(X_tr_scaled, y_tr_res)
        except:
            try:
                model_fold.fit(X_tr_scaled, y_tr_res)
            except Exception as e:
                print(f"\n    ERROR: {e}")
                continue

        try:
            model_to_save = sanitize_for_pickle(model_fold)
            joblib.dump(model_to_save, os.path.join(DRIVE_FOLDER, f"{name}_fold{fold}.pkl"))
        except:
            pass

        try:
            proba_val = model_fold.predict_proba(X_val_scaled)[:, 1]
            proba_test = model_fold.predict_proba(X_test_scaled)[:, 1]
        except:
            proba_val = np.zeros(len(X_val_scaled))
            proba_test = np.zeros(len(X_test_scaled))

        val_preds_stack[val_idx, idx] = proba_val
        test_preds_stack[:, idx] += proba_test / N_FOLDS

        print("✓")

    # Fold evaluation
    fold_pred_proba = val_preds_stack[val_idx].mean(axis=1)
    best_f2 = 0
    best_t = 0.5
    best_precision = 0
    best_recall = 0
    
    for t in np.linspace(0.3, 0.7, 81):
        pred = (fold_pred_proba >= t).astype(int)
        f2 = fbeta_score(y_val, pred, beta=2)
        if f2 > best_f2:
            best_f2 = f2
            best_t = t
            best_precision = precision_score(y_val, pred)
            best_recall = recall_score(y_val, pred)

    fold_f2_scores.append(best_f2)
    fold_metrics.append({
        'f2': best_f2,
        'threshold': best_t,
        'precision': best_precision,
        'recall': best_recall
    })
    
    print(f"  Fold F2: {best_f2:.4f} @ threshold {best_t:.3f}")
    print(f"  Precision: {best_precision:.4f}, Recall: {best_recall:.4f}")

print(f"\n{'=' * 80}")
print(f"Mean Fold F2: {np.mean(fold_f2_scores):.4f} ± {np.std(fold_f2_scores):.4f}")
print(f"Mean Threshold: {np.mean([m['threshold'] for m in fold_metrics]):.3f}")
print(f"{'=' * 80}\n")

np.save(os.path.join(DRIVE_FOLDER, "val_preds_stack.npy"), val_preds_stack)
np.save(os.path.join(DRIVE_FOLDER, "test_preds_stack.npy"), test_preds_stack)

# ============================================================
# 8) OPTIMIZED META-MODEL
# ============================================================
print("=" * 80)
print("🧠 TRAINING OPTIMIZED META-MODEL")
print("=" * 80)

weights_options = [
    [0.15, 0.22, 0.22, 0.22, 0.19],
    [0.12, 0.24, 0.24, 0.22, 0.18],
    [0.10, 0.25, 0.25, 0.20, 0.20],
]

best_weighted_f2 = 0
best_weights = weights_options[0]
best_weighted_threshold = 0.5

for weights in weights_options:
    val_pred_weighted = np.average(val_preds_stack, axis=1, weights=weights)
    for t in np.linspace(0.3, 0.7, 81):
        pred = (val_pred_weighted >= t).astype(int)
        f2 = fbeta_score(y_all, pred, beta=2)
        if f2 > best_weighted_f2:
            best_weighted_f2 = f2
            best_weighted_threshold = t
            best_weights = weights

print(f"✓ Best Weighted F2: {best_weighted_f2:.4f} @ threshold = {best_weighted_threshold:.3f}")
print(f"✓ Best weights: {best_weights}")

meta_xgb = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=3.0,
    reg_lambda=4.0,
    scale_pos_weight=scale_pos * 0.75,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
meta_xgb.fit(val_preds_stack, y_all)
sanitize_for_pickle(meta_xgb)
joblib.dump(meta_xgb, os.path.join(DRIVE_FOLDER, "meta_model.pkl"))

val_pred_meta = meta_xgb.predict_proba(val_preds_stack)[:, 1]
best_meta_f2 = 0
best_meta_threshold = 0.5

for t in np.linspace(0.3, 0.7, 81):
    pred = (val_pred_meta >= t).astype(int)
    f2 = fbeta_score(y_all, pred, beta=2)
    if f2 > best_meta_f2:
        best_meta_f2 = f2
        best_meta_threshold = t

print(f"✓ Meta XGBoost F2: {best_meta_f2:.4f} @ threshold = {best_meta_threshold:.3f}")

if best_meta_f2 > best_weighted_f2:
    print(f"\n✅ Using Meta XGBoost (F2: {best_meta_f2:.4f})")
    use_meta = True
    best_f2 = best_meta_f2
    best_threshold = best_meta_threshold
else:
    print(f"\n✅ Using Weighted Average (F2: {best_weighted_f2:.4f})")
    use_meta = False
    best_f2 = best_weighted_f2
    best_threshold = best_weighted_threshold
    np.save(os.path.join(DRIVE_FOLDER, "best_weights.npy"), np.array(best_weights))

print()

# ============================================================
# 9) 🔧 FIX 3: CONSERVATIVE THRESHOLD STRATEGY
# ============================================================
print("=" * 80)
print("🔮 FIX 3: CONSERVATIVE PREDICTION STRATEGY")
print("=" * 80)

if use_meta:
    test_prob = meta_xgb.predict_proba(test_preds_stack)[:, 1]
else:
    test_prob = np.average(test_preds_stack, axis=1, weights=best_weights)

expected_cheat_rate = np.mean(y_all)

# ใช้ threshold ที่ conservative กว่า
# เพิ่ม weight ให้ expected distribution มากขึ้น
sorted_proba = np.sort(test_prob)[::-1]
target_count = int(len(test_prob) * expected_cheat_rate)
distribution_threshold = sorted_proba[min(target_count, len(sorted_proba) - 1)]

# Conservative strategy: ให้น้ำหนัก distribution มากกว่า
recall_weight = 0.4  # ลดลงจาก 0.8
precision_weight = 0.6  # เพิ่มขึ้นจาก 0.2

adjusted_threshold = (best_threshold * recall_weight + distribution_threshold * precision_weight)

# Ensure reasonable bounds
final_threshold = max(0.35, min(adjusted_threshold, 0.65))

# Additional check: ถ้า predicted rate ยังสูงเกินไป ให้ปรับ threshold ขึ้น
test_pred_initial = (test_prob >= final_threshold).astype(int)
initial_rate = test_pred_initial.sum() / len(test_pred_initial)

if initial_rate > expected_cheat_rate * 1.15:  # ถ้าเกิน 15%
    # หา threshold ที่ให้ rate ใกล้เคียง expected มากขึ้น
    for candidate_t in np.linspace(final_threshold, 0.7, 50):
        candidate_pred = (test_prob >= candidate_t).astype(int)
        candidate_rate = candidate_pred.sum() / len(candidate_pred)
        if abs(candidate_rate - expected_cheat_rate) < abs(initial_rate - expected_cheat_rate):
            final_threshold = candidate_t
            break

test_pred = (test_prob >= final_threshold).astype(int)
predicted_rate = test_pred.sum() / len(test_pred)

print(f"✓ Best tuned threshold: {best_threshold:.3f}")
print(f"✓ Distribution threshold: {distribution_threshold:.3f}")
print(f"✓ Adjusted threshold: {adjusted_threshold:.3f}")
print(f"✓ Final threshold: {final_threshold:.3f}")
print(f"✓ Expected cheat rate: {expected_cheat_rate:.2%}")
print(f"✓ Predicted cheat rate: {predicted_rate:.2%}")
print(f"✓ Deviation: {abs(predicted_rate - expected_cheat_rate):.2%}")

# Color-coded feedback
if abs(predicted_rate - expected_cheat_rate) < 0.05:
    print("✅ EXCELLENT: Deviation < 5%")
elif abs(predicted_rate - expected_cheat_rate) < 0.10:
    print("✓ GOOD: Deviation < 10%")
else:
    print("⚠️  WARNING: Deviation > 10%")

print()

# ============================================================
# 10) CREATE SUBMISSION
# ============================================================
print("=" * 80)
print("💾 CREATING SUBMISSION")
print("=" * 80)

submission = pd.DataFrame({
    "id": test["id"],
    "task1": test_pred
})
submission_with_proba = pd.DataFrame({
    "id": test["id"],
    "task1": test_pred,
    "cheater_probability": test_prob
})

sub_path = os.path.join(DRIVE_FOLDER, "submission_task1_v2_fixed.csv")
sub_path_proba = os.path.join(DRIVE_FOLDER, "submission_with_probabilities.csv")

submission.to_csv(sub_path, index=False)
submission_with_proba.to_csv(sub_path_proba, index=False)

print(f"✓ Submission saved to: {sub_path}")
print(f"✓ Submission with probabilities saved to: {sub_path_proba}")
print(f"\n📊 Prediction distribution:")
print(submission['task1'].value_counts())
print(f"✓ Cheaters detected: {test_pred.sum()} ({predicted_rate:.2%})")

print(f"\n📈 Probability distribution:")
print(f"  Min: {test_prob.min():.4f}")
print(f"  25%: {np.percentile(test_prob, 25):.4f}")
print(f"  50%: {np.percentile(test_prob, 50):.4f}")
print(f"  75%: {np.percentile(test_prob, 75):.4f}")
print(f"  Max: {test_prob.max():.4f}")

high_conf_cheaters = (test_prob >= 0.7).sum()
medium_conf_cheaters = ((test_prob >= final_threshold) & (test_prob < 0.7)).sum()
high_conf_legit = (test_prob <= 0.3).sum()
uncertain = ((test_prob > 0.3) & (test_prob < final_threshold)).sum()

print(f"\n🎯 Confidence breakdown:")
print(f"  High confidence cheaters (prob >= 0.7): {high_conf_cheaters} ({high_conf_cheaters/len(test_prob)*100:.1f}%)")
print(f"  Medium confidence cheaters ({final_threshold:.2f} <= prob < 0.7): {medium_conf_cheaters} ({medium_conf_cheaters/len(test_prob)*100:.1f}%)")
print(f"  High confidence legit (prob <= 0.3): {high_conf_legit} ({high_conf_legit/len(test_prob)*100:.1f}%)")
print(f"  Uncertain (0.3 < prob < {final_threshold:.2f}): {uncertain} ({uncertain/len(test_prob)*100:.1f}%)")

print(f"\n✅ V2 FIXED PIPELINE COMPLETED!")
print(f"🎯 Key improvements:")
print(f"   - Selective missing indicators (only strategic ones)")
print(f"   - Neutral imputation (no missing = cheater bias)")
print(f"   - Conservative threshold ({final_threshold:.3f} vs previous 0.291)")
print(f"   - Better balanced predicted rate")

print("=" * 80)

# ============================================================
# 11) FEATURE IMPORTANCE ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("📊 TOP 20 FEATURE IMPORTANCE")
print("=" * 80)

top_20_features = importances.nlargest(20)
for idx, (feat, imp) in enumerate(top_20_features.items(), 1):
    print(f"{idx:2d}. {feat:40s} {imp:.6f}")

print("\n" + "=" * 80)
print("✅ ALL DONE! Ready to submit.")
print("=" * 80)