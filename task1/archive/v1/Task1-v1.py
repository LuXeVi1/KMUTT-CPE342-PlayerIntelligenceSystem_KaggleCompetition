# ============================================================
# Task 1 — Full Pipeline (Preprocess → K-Fold Training → Meta)
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.base import clone

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from google.colab import drive

# ============================================================
# 1) MOUNT DRIVE + SET PATH
# ============================================================

drive.mount("/content/drive", force_remount=True)

DRIVE_FOLDER = "/content/drive/MyDrive/ML-Task1-FULL-PIPELINE"
os.makedirs(DRIVE_FOLDER, exist_ok=True)

TRAIN_PATH = "/content/task1/train.csv"
TEST_PATH  = "/content/task1/test.csv"

print("Saving to:", DRIVE_FOLDER)


# ============================================================
# 2) LOAD RAW DATA
# ============================================================

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

mask = train["is_cheater"].notna()
train = train[mask].reset_index(drop=True)

y_all = train["is_cheater"].astype(int).values

X_all = train.drop(columns=["id", "player_id", "is_cheater"])
X_test = test.drop(columns=["id", "player_id"])


# ============================================================
# 3) IMPUTE + FEATURE ENGINEERING
# ============================================================

def add_features(df):
    c = df.columns
    if "accuracy_score" in c and "headshot_percentage" in c:
        df["aim_eff"] = df["accuracy_score"] * df["headshot_percentage"]
    if "kill_death_ratio" in c and "first_blood_rate" in c:
        df["aggr"] = df["kill_death_ratio"] * df["first_blood_rate"]
    if "kill_consistency" in c and "movement_pattern_score" in c:
        df["skill_cons"] = df["kill_consistency"] * df["movement_pattern_score"]
    if "game_sense_score" in c and "map_knowledge" in c:
        df["strat_score"] = df["game_sense_score"] * df["map_knowledge"]
    if "sessions_per_day" in c and "avg_session_length_min" in c:
        df["stability"] = df["sessions_per_day"] * df["avg_session_length_min"]
    if "damage_per_round" in c and "survival_time_avg" in c:
        df["combat"] = df["damage_per_round"] * df["survival_time_avg"]
    if "aiming_smoothness" in c and "spray_control_score" in c:
        df["mech_skill"] = df["aiming_smoothness"] * df["spray_control_score"]
    return df


imputer = SimpleImputer(strategy="median")
numeric_cols = X_all.select_dtypes(include=np.number).columns.tolist()

X_all[numeric_cols] = imputer.fit_transform(X_all[numeric_cols])
X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])

X_all = add_features(X_all)
X_test = add_features(X_test)


# ============================================================
# 4) FEATURE SELECTION
# ============================================================

corr = pd.concat([X_all, pd.Series(y_all, name="is_cheater")], axis=1).corr()["is_cheater"].abs()
selected_features = corr[corr > 0.02].index.drop("is_cheater").tolist()

X_all = X_all[selected_features]
X_test = X_test[selected_features]

np.save(os.path.join(DRIVE_FOLDER, "selected_features.npy"), np.array(selected_features))

print(f"Selected {len(selected_features)} features")


# ============================================================
# 5) TRAIN BASE MODELS (5-FOLD)
# ============================================================

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
smote = SMOTE(random_state=42)
scaler = StandardScaler()

val_preds_stack = np.zeros((len(X_all), 3))
test_preds_stack = np.zeros((len(X_test), 3))

# Define base models
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
lgb = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=64, random_state=42)
cat = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, verbose=0, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y_all), 1):
    print(f"\n=========== Fold {fold} ===========")

    X_tr, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
    y_tr, y_val = y_all[train_idx], y_all[val_idx]

    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

    X_tr_scaled = scaler.fit_transform(X_tr_res)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # clone models to ensure independence
    rf_f = clone(rf)
    lgb_f = clone(lgb)
    cat_f = clone(cat)

    rf_f.fit(X_tr_scaled, y_tr_res)

    lgb_f.fit(X_tr_scaled, y_tr_res,
        eval_set=[(X_val_scaled, y_val)],
        callbacks=[early_stopping(stopping_rounds=30), log_evaluation(0)]
    )

    cat_f.fit(X_tr_scaled, y_tr_res,
        eval_set=Pool(X_val_scaled, y_val),
        early_stopping_rounds=30,
        verbose=0
    )

    # Save each fold
    joblib.dump(rf_f,  os.path.join(DRIVE_FOLDER, f"rf_fold{fold}.pkl"))
    joblib.dump(lgb_f, os.path.join(DRIVE_FOLDER, f"lgb_fold{fold}.pkl"))
    joblib.dump(cat_f, os.path.join(DRIVE_FOLDER, f"cat_fold{fold}.pkl"))

    # Collect stacked preds
    val_preds_stack[val_idx, 0] = rf_f.predict_proba(X_val_scaled)[:, 1]
    val_preds_stack[val_idx, 1] = lgb_f.predict_proba(X_val_scaled)[:, 1]
    val_preds_stack[val_idx, 2] = cat_f.predict_proba(X_val_scaled)[:, 1]

    test_preds_stack[:, 0] += rf_f.predict_proba(X_test_scaled)[:, 1] / kf.n_splits
    test_preds_stack[:, 1] += lgb_f.predict_proba(X_test_scaled)[:, 1] / kf.n_splits
    test_preds_stack[:, 2] += cat_f.predict_proba(X_test_scaled)[:, 1] / kf.n_splits


np.save(os.path.join(DRIVE_FOLDER, "val_preds_stack.npy"), val_preds_stack)
np.save(os.path.join(DRIVE_FOLDER, "test_preds_stack.npy"), test_preds_stack)

print("\nSaved stacked predictions.")


# ============================================================
# 6) TRAIN META-MODEL (LOGISTIC REGRESSION)
# ============================================================

meta_model = LogisticRegression(max_iter=2000)
meta_model.fit(val_preds_stack, y_all)

val_pred = meta_model.predict(val_preds_stack)
f2 = fbeta_score(y_all, val_pred, beta=2)
print(f"\nMeta-model Validation F2 = {f2:.4f}")

joblib.dump(meta_model, os.path.join(DRIVE_FOLDER, "meta_model.pkl"))


# ============================================================
# 7) THRESHOLD TUNING
# ============================================================

best_thresh = 0.5
best_f2 = 0

val_prob = meta_model.predict_proba(val_preds_stack)[:, 1]

for t in np.arange(0.1, 0.9, 0.05):
    pred_t = (val_prob >= t).astype(int)
    f2_t = fbeta_score(y_all, pred_t, beta=2)
    if f2_t > best_f2:
        best_f2 = f2_t
        best_thresh = t

print(f"\nBest threshold = {best_thresh}, F2 = {best_f2:.4f}")


# ============================================================
# 8) TEST PREDICTION + SUBMISSION
# ============================================================

test_prob = meta_model.predict_proba(test_preds_stack)[:, 1]
test_pred = (test_prob >= best_thresh).astype(int)

# Rename column to task1
submission = pd.DataFrame({
    "id": test["id"],
    "task1": test_pred
})

sub_path = os.path.join(DRIVE_FOLDER, "submission_task1_full_pipeline.csv")
submission.to_csv(sub_path, index=False)

print("\nSubmission saved to:", sub_path)
print("\nPipeline Completed Successfully!")

