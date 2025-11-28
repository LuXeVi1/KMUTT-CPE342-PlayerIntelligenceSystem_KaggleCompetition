# ============================================================================
# TASK 2: PLAYER SEGMENT CLASSIFICATION 
# ============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATA
# ============================================================================
DATA_PATH = "/content/task2"
train = pd.read_csv(f"{DATA_PATH}/train.csv")
test = pd.read_csv(f"{DATA_PATH}/test.csv")

print("="*80)
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("="*80)

# ============================================================================
# 2. ADVANCED MISSING VALUE HANDLING
# ============================================================================
TARGET = "segment"
y = train[TARGET].copy()
X = train.drop(columns=[TARGET])

# Remove rows with missing target
if y.isna().sum() > 0:
    non_null_idx = y.dropna().index
    X = X.loc[non_null_idx]
    y = y.loc[non_null_idx]
    print(f"Removed {len(train) - len(X)} rows with missing target")

# Separate numeric and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"\nNumeric features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")

# Remove ID columns
id_cols = ['id', 'player_id']
X = X.drop(columns=id_cols, errors='ignore')
test = test.drop(columns=id_cols, errors='ignore')

# Update column lists after dropping IDs
numeric_cols = [col for col in numeric_cols if col not in id_cols]
categorical_cols = [col for col in categorical_cols if col not in id_cols]

# ============================================================================
# 3. ENHANCED FEATURE ENGINEERING (Before handling missing values)
# ============================================================================
def create_advanced_features(df):
    """Create domain-specific features based on game mechanics"""
    
    # === ENGAGEMENT FEATURES ===
    if "avg_session_duration" in df.columns and "play_frequency" in df.columns:
        df["engagement_score"] = df["avg_session_duration"] * df["play_frequency"]
        df["engagement_intensity"] = df["avg_session_duration"] / (df["play_frequency"] + 1)
    
    if "total_playtime_hours" in df.columns and "account_age_days" in df.columns:
        df["playtime_per_day"] = df["total_playtime_hours"] / (df["account_age_days"] + 1)
    
    if "login_streak" in df.columns and "play_frequency" in df.columns:
        df["consistency_score"] = df["login_streak"] * df["play_frequency"]
    
    # === SPENDING BEHAVIOR FEATURES ===
    if "total_spending_thb" in df.columns and "avg_monthly_spending" in df.columns:
        df["spending_months"] = df["total_spending_thb"] / (df["avg_monthly_spending"] + 1)
        df["spending_acceleration"] = df["avg_monthly_spending"] / (df["total_spending_thb"] / 12 + 1)
    
    if "total_spending_thb" in df.columns and "account_age_days" in df.columns:
        df["spending_per_day"] = df["total_spending_thb"] / (df["account_age_days"] + 1)
    
    if "spending_frequency" in df.columns and "avg_monthly_spending" in df.columns:
        df["avg_transaction_value"] = df["avg_monthly_spending"] / (df["spending_frequency"] + 1)
    
    if "total_spending_thb" in df.columns and "total_playtime_hours" in df.columns:
        df["spending_per_hour"] = df["total_spending_thb"] / (df["total_playtime_hours"] + 1)
    
    # === SOCIAL FEATURES ===
    if "friend_count" in df.columns and "team_play_percentage" in df.columns:
        df["social_engagement"] = df["friend_count"] * df["team_play_percentage"] / 100
    
    if "friend_count" in df.columns and "chat_activity_score" in df.columns:
        df["social_interaction"] = df["friend_count"] * df["chat_activity_score"]
    
    if "gifts_sent_received" in df.columns and "friend_count" in df.columns:
        df["gift_per_friend"] = df["gifts_sent_received"] / (df["friend_count"] + 1)
    
    if "friend_invites_sent" in df.columns and "friend_count" in df.columns:
        df["friend_conversion_rate"] = df["friend_count"] / (df["friend_invites_sent"] + 1)
    
    # === COMPETITIVE FEATURES ===
    if "ranked_participation_rate" in df.columns and "tournament_entries" in df.columns:
        df["competitive_intensity"] = df["ranked_participation_rate"] * df["tournament_entries"]
    
    if "win_rate_ranked" in df.columns and "ranked_participation_rate" in df.columns:
        df["competitive_success"] = df["win_rate_ranked"] * df["ranked_participation_rate"]
    
    if "competitive_rank" in df.columns and "win_rate_ranked" in df.columns:
        df["rank_performance"] = df["competitive_rank"] * df["win_rate_ranked"]
    
    # === COLLECTION & ACHIEVEMENT FEATURES ===
    if "achievement_completion_rate" in df.columns and "collection_progress" in df.columns:
        df["completion_dedication"] = df["achievement_completion_rate"] * df["collection_progress"]
    
    if "rare_items_count" in df.columns and "collection_progress" in df.columns:
        df["collector_intensity"] = df["rare_items_count"] * df["collection_progress"]
    
    if "speed_of_progression" in df.columns and "achievement_completion_rate" in df.columns:
        df["progression_efficiency"] = df["speed_of_progression"] * df["achievement_completion_rate"]
    
    # === PREFERENCE FEATURES ===
    pref_cols = ["item_type_preference_cosmetic", "item_type_preference_performance", 
                 "item_type_preference_social"]
    if all(col in df.columns for col in pref_cols):
        df["preference_diversity"] = df[pref_cols].std(axis=1)
        df["max_preference"] = df[pref_cols].max(axis=1)
        df["cosmetic_focus"] = df["item_type_preference_cosmetic"] / (df[pref_cols].sum(axis=1) + 1)
    
    # === RECENCY & ACTIVITY ===
    if "days_since_last_login" in df.columns and "login_streak" in df.columns:
        df["activity_recency_ratio"] = df["login_streak"] / (df["days_since_last_login"] + 1)
    
    if "peak_concurrent_hours" in df.columns and "avg_session_duration" in df.columns:
        df["peak_intensity"] = df["peak_concurrent_hours"] / (df["avg_session_duration"] / 60 + 1)
    
    # === VIP & LOYALTY ===
    if "vip_tier" in df.columns and "total_spending_thb" in df.columns:
        df["vip_spending_ratio"] = df["total_spending_thb"] / (df["vip_tier"] + 1)
    
    return df

# Apply feature engineering to both train and test
print("\n" + "="*80)
print("Creating advanced features...")
X = create_advanced_features(X)
test = create_advanced_features(test)
print(f"Total features after engineering: {len(X.columns)}")

# ============================================================================
# 4. INTELLIGENT MISSING VALUE HANDLING
# ============================================================================
print("\n" + "="*80)
print("Handling missing values intelligently...")

# Update numeric columns list after feature engineering
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Strategy: Use median for most, but mean for certain features
mean_impute_cols = ['avg_session_duration', 'avg_monthly_spending', 'avg_match_length']
for col in numeric_cols:
    if col in X.columns:
        if col in mean_impute_cols:
            fill_value = X[col].mean()
        else:
            fill_value = X[col].median()
        
        X[col] = X[col].fillna(fill_value)
        if col in test.columns:
            test[col] = test[col].fillna(fill_value)

# Handle categorical columns - fill with 'Unknown' instead of dropping
for col in categorical_cols:
    if col in X.columns:
        X[col] = X[col].fillna('Unknown')
        if col in test.columns:
            test[col] = test[col].fillna('Unknown')

# Encode categorical variables
print("\nEncoding categorical variables...")
label_encoders = {}
for col in categorical_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        if col in test.columns:
            # Handle unseen categories in test
            test[col] = test[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
            test[col] = le.transform(test[col].astype(str))
        label_encoders[col] = le

print(f"Final feature count: {len(X.columns)}")

# ============================================================================
# 5. FEATURE SCALING
# ============================================================================
print("\n" + "="*80)
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)

# ============================================================================
# 6. HANDLE CLASS IMBALANCE WITH SMOTE
# ============================================================================
print("\n" + "="*80)
print("Analyzing class distribution...")
class_counts = y.value_counts().sort_index()
print("Class distribution:")
for cls, count in class_counts.items():
    print(f"  Class {cls}: {count} ({count/len(y)*100:.2f}%)")

imbalance_ratio = class_counts.max() / class_counts.min()
print(f"\nImbalance ratio: {imbalance_ratio:.2f}")

if imbalance_ratio > 1.5:
    from imblearn.over_sampling import SMOTE
    print("\nApplying SMOTE to balance classes...")
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_scaled, y)
    
    print("Class distribution after SMOTE:")
    unique, counts = np.unique(y_res, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} ({count/len(y_res)*100:.2f}%)")
else:
    X_res, y_res = X_scaled, y.values
    print("Skipped SMOTE (balance acceptable)")

# ============================================================================
# 7. COMPUTE CLASS WEIGHTS (for models that support it)
# ============================================================================
class_weights = compute_class_weight('balanced', classes=np.unique(y_res), y=y_res)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print(f"\nClass weights: {class_weight_dict}")

# ============================================================================
# 8. DEFINE OPTIMIZED MODELS
# ============================================================================
print("\n" + "="*80)
print("Initializing ensemble models...")

# XGBoost with optimized parameters
xgb_model = xgb.XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    num_class=4,
    n_estimators=500,
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=2,
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    tree_method='hist',
    n_jobs=-1
)

# LightGBM with optimized parameters
lgb_model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=4,
    n_estimators=500,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=80,
    min_data_in_leaf=30,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# CatBoost with optimized parameters
cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3,
    loss_function="MultiClass",
    random_seed=42,
    verbose=0,
    thread_count=-1
)

# Weighted Voting Ensemble
ensemble = VotingClassifier(
    estimators=[
        ("xgb", xgb_model), 
        ("lgb", lgb_model), 
        ("cat", cat_model)
    ],
    voting="soft",
    weights=[1.2, 1.0, 1.1],  # Give slightly more weight to XGB and CatBoost
    n_jobs=-1
)

# ============================================================================
# 9. STRATIFIED K-FOLD CROSS-VALIDATION
# ============================================================================
print("\n" + "="*80)
print("Starting Stratified K-Fold Cross-Validation (5 folds)...")
print("="*80)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores_macro = []
f1_scores_weighted = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_res, y_res), 1):
    print(f"\n--- Fold {fold}/5 ---")
    X_train, X_val = X_res[train_idx], X_res[val_idx]
    y_train, y_val = y_res[train_idx], y_res[val_idx]
    
    # Train ensemble
    ensemble.fit(X_train, y_train)
    
    # Predict
    y_pred = ensemble.predict(X_val)
    
    # Calculate F1 scores
    f1_macro = f1_score(y_val, y_pred, average="macro")
    f1_weighted = f1_score(y_val, y_pred, average="weighted")
    
    f1_scores_macro.append(f1_macro)
    f1_scores_weighted.append(f1_weighted)
    
    print(f"F1 (macro): {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")

print("\n" + "="*80)
print("CROSS-VALIDATION RESULTS")
print("="*80)
print(f"Average F1 (macro):    {np.mean(f1_scores_macro):.4f} ± {np.std(f1_scores_macro):.4f}")
print(f"Average F1 (weighted): {np.mean(f1_scores_weighted):.4f} ± {np.std(f1_scores_weighted):.4f}")
print("="*80)

# ============================================================================
# 10. TRAIN FINAL MODEL ON ALL DATA
# ============================================================================
print("\nTraining final model on all data...")
ensemble.fit(X_res, y_res)
print("✓ Final model trained successfully")

# ============================================================================
# 11. MAKE PREDICTIONS & CREATE SUBMISSION
# ============================================================================
print("\nMaking predictions on test set...")
y_test_pred = ensemble.predict(test_scaled)

# Create submission with formatted ID
submission = pd.DataFrame({
    "id": [f"ANS{i+1:05d}" for i in range(len(y_test_pred))],
    "task2": y_test_pred
})

# Save submission
submission_path = f"{DATA_PATH}/submission_Task2_v2.csv"
submission.to_csv(submission_path, index=False)

print("\n" + "="*80)
print("SUBMISSION SUMMARY")
print("="*80)
print(f"✓ Submission file saved: {submission_path}")
print(f"✓ Total predictions: {len(submission)}")
print("\nPrediction distribution:")
pred_counts = pd.Series(y_test_pred).value_counts().sort_index()
for cls, count in pred_counts.items():
    print(f"  Class {cls}: {count} ({count/len(y_test_pred)*100:.2f}%)")

print("\nFirst 10 predictions:")
print(submission.head(10))
print("="*80)
print("\n✓ Task 2 v2 completed successfully!")