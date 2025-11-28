# ============================================================================
# 🔒 TASK 5 (V5): Account Security Monitoring - Conservative Ensemble
# เพิ่มเฉพาะ features ที่จำเป็น + ปรับแต่ง hyperparameters
# ============================================================================

import os
import numpy as np
import pandas as pd
import gc

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

import joblib

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DATA_PATH = "/content/task5"
TEST_CSV = os.path.join(DATA_PATH, "test.csv")
OUTPUT_CSV = os.path.join(DATA_PATH, "submission_task5_V5.csv")

RANDOM_STATE = 42
CONTAMINATION = 0.045  # ปรับจาก 0.05 เพื่อความระมัดระวัง
OCSVM_NU = CONTAMINATION
SAMPLE_SIZE_FOR_OCSVM = 15000

# Ensemble weights (ปรับได้)
ISO_WEIGHT = 0.35
OCSVM_WEIGHT = 0.30
AE_WEIGHT = 0.35

# ----------------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------------
df = pd.read_csv(TEST_CSV)
print(f"Loaded: {df.shape}")
print(f"Duplicated rows: {df.duplicated().sum()}")

# Remove duplicates if any
if df.duplicated().sum() > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"After deduplication: {df.shape}")

# Identify ID column
id_col = None
for c in ['id', 'Id', 'ID', 'player_id']:
    if c in df.columns:
        id_col = c
        break

# ----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING (Conservative - เพิ่มเฉพาะที่จำเป็น)
# ----------------------------------------------------------------------------
def create_essential_features(df):
    """สร้างเฉพาะ features ที่มีความหมายชัดเจนต่อ anomaly detection"""
    features = []
    
    # Base features (numeric only)
    base_features = df.select_dtypes(include=[np.number])
    if id_col and id_col in base_features.columns:
        base_features = base_features.drop(columns=[id_col])
    features.append(base_features)
    
    # === Critical temporal patterns (เฉพาะที่สำคัญ) ===
    
    # 1. Skill trajectory trend (ตรวจจับการเปลี่ยนแปลง skill ผิดปกติ)
    skill_cols = ['skill_rating_1', 'skill_rating_2', 'skill_rating_3', 'skill_rating_4']
    if all(c in df.columns for c in skill_cols):
        skill_data = df[skill_cols].values
        features.append(pd.DataFrame({
            'skill_mean': np.nanmean(skill_data, axis=1),
            'skill_std': np.nanstd(skill_data, axis=1),
            'skill_range': np.nanmax(skill_data, axis=1) - np.nanmin(skill_data, axis=1)
        }))
    
    # 2. Login behavior changes (ตรวจจับการ login ผิดปกติ)
    login_cols = ['login_count_1', 'login_count_2', 'login_count_3', 'login_count_4']
    if all(c in df.columns for c in login_cols):
        login_data = df[login_cols].values
        features.append(pd.DataFrame({
            'login_volatility': np.nanstd(login_data, axis=1) / (np.nanmean(login_data, axis=1) + 1e-5)
        }))
    
    # 3. Location consistency (ตรวจจับการเปลี่ยน location ผิดปกติ)
    loc_change_cols = ['location_changes_1', 'location_changes_2', 'location_changes_3', 'location_changes_4']
    if all(c in df.columns for c in loc_change_cols):
        loc_data = df[loc_change_cols].values
        features.append(pd.DataFrame({
            'location_changes_max': np.nanmax(loc_data, axis=1)
        }))
    
    # 4. Purchase behavior (ตรวจจับการซื้อขายผิดปกติ)
    purchase_freq_cols = ['purchase_frequency_1', 'purchase_frequency_2', 'purchase_frequency_3', 'purchase_frequency_4']
    purchase_amt_cols = ['purchase_amount_mean_1', 'purchase_amount_mean_2', 'purchase_amount_mean_3', 'purchase_amount_mean_4']
    
    if all(c in df.columns for c in purchase_freq_cols):
        purchase_freq_data = df[purchase_freq_cols].values
        features.append(pd.DataFrame({
            'purchase_freq_spike': np.nanmax(purchase_freq_data, axis=1) - np.nanmin(purchase_freq_data, axis=1)
        }))
    
    if all(c in df.columns for c in purchase_amt_cols):
        purchase_amt_data = df[purchase_amt_cols].values
        features.append(pd.DataFrame({
            'purchase_amt_spike': np.nanmax(purchase_amt_data, axis=1) - np.nanmin(purchase_amt_data, axis=1)
        }))
    
    # 5. Device and IP entropy (ตรวจจับการใช้หลาย device/IP)
    device_cols = ['device_count_1', 'device_count_2', 'device_count_3', 'device_count_4']
    entropy_cols = ['ip_hash_entropy_1', 'ip_hash_entropy_2', 'ip_hash_entropy_3', 'ip_hash_entropy_4']
    
    if all(c in df.columns for c in device_cols):
        device_data = df[device_cols].values
        features.append(pd.DataFrame({
            'device_count_max': np.nanmax(device_data, axis=1)
        }))
    
    if all(c in df.columns for c in entropy_cols):
        entropy_data = df[entropy_cols].values
        features.append(pd.DataFrame({
            'ip_entropy_mean': np.nanmean(entropy_data, axis=1)
        }))
    
    # Combine all features
    X = pd.concat(features, axis=1)
    return X

X = create_essential_features(df)
print(f"Total features: {X.shape[1]} (original: {df.select_dtypes(include=[np.number]).shape[1] - 1})")

# ----------------------------------------------------------------------------
# 3. MISSING VALUES (Simple median fill)
# ----------------------------------------------------------------------------
X = X.fillna(X.median())

feature_columns = X.columns.tolist()

# ----------------------------------------------------------------------------
# 4. SCALING
# ----------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)

gc.collect()

# ----------------------------------------------------------------------------
# 5. ISOLATION FOREST (ปรับ hyperparameters)
# ----------------------------------------------------------------------------
print("\n[1/3] Training Isolation Forest...")
iso = IsolationForest(
    n_estimators=200,  # เพิ่มจาก 100 → 200
    contamination=CONTAMINATION,
    max_samples='auto',
    max_features=1.0,  # ใช้ features ทั้งหมด
    bootstrap=True,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
iso.fit(X_scaled)
iso_scores_raw = -iso.decision_function(X_scaled)

del iso
gc.collect()

# ----------------------------------------------------------------------------
# 6. ONE-CLASS SVM
# ----------------------------------------------------------------------------
print("[2/3] Training One-Class SVM...")

if X_scaled.shape[0] > SAMPLE_SIZE_FOR_OCSVM:
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(
        np.arange(X_scaled.shape[0]),
        size=SAMPLE_SIZE_FOR_OCSVM,
        replace=False
    )
    X_ocsvm_train = X_scaled[idx]
else:
    X_ocsvm_train = X_scaled

ocsvm = OneClassSVM(kernel="rbf", nu=OCSVM_NU, gamma="auto")  # เปลี่ยนจาก scale → auto
ocsvm.fit(X_ocsvm_train)

# Batch prediction for memory safety
batch_size = 2048
ocsvm_scores_raw = []
for i in range(0, len(X_scaled), batch_size):
    batch_scores = -ocsvm.decision_function(X_scaled[i:i+batch_size])
    ocsvm_scores_raw.append(batch_scores)

ocsvm_scores_raw = np.concatenate(ocsvm_scores_raw)

# ----------------------------------------------------------------------------
# 7. AUTOENCODER (ปรับ architecture และ training)
# ----------------------------------------------------------------------------
print("[3/3] Training Autoencoder...")

input_dim = X_scaled.shape[1]
encoding_dim = max(16, int(input_dim / 3))  # เปลี่ยนจาก /4 → /3 (เก็บข้อมูลมากขึ้น)

def build_autoencoder(input_dim, encoding_dim):
    inp = layers.Input(shape=(input_dim,))
    
    # Encoder
    x = layers.Dense(int(input_dim*0.75), activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)  # Dropout เบาๆ
    
    x = layers.Dense(int(input_dim/2), activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    # Bottleneck
    encoded = layers.Dense(encoding_dim, activation="relu")(x)
    
    # Decoder
    x = layers.Dense(int(input_dim/2), activation="relu")(encoded)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(int(input_dim*0.75), activation="relu")(x)
    
    out = layers.Dense(input_dim, activation="linear")(x)
    
    model = models.Model(inputs=inp, outputs=out)
    return model

ae = build_autoencoder(input_dim, encoding_dim)
ae.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="mse"
)

X_train, X_val = train_test_split(X_scaled, test_size=0.15, random_state=RANDOM_STATE)

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-6,
    verbose=1
)

ae.fit(
    X_train, X_train,
    epochs=200,
    batch_size=64,
    validation_data=(X_val, X_val),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Reconstruction errors (batch)
recon_errors = []
for i in range(0, len(X_scaled), batch_size):
    X_batch = X_scaled[i:i+batch_size]
    X_recon = ae.predict(X_batch, batch_size=batch_size, verbose=0)
    batch_err = np.mean((X_batch - X_recon) ** 2, axis=1)
    recon_errors.append(batch_err)

recon_errors = np.concatenate(recon_errors)

del X_train, X_val
gc.collect()

# ----------------------------------------------------------------------------
# 8. SCORE NORMALIZATION
# ----------------------------------------------------------------------------
def minmax_norm(arr):
    arr = np.array(arr, dtype=float)
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)

iso_norm = minmax_norm(iso_scores_raw)
ocsvm_norm = minmax_norm(ocsvm_scores_raw)
ae_norm = minmax_norm(recon_errors)

# ----------------------------------------------------------------------------
# 9. WEIGHTED ENSEMBLE
# ----------------------------------------------------------------------------
ensemble_score = (
    ISO_WEIGHT * iso_norm + 
    OCSVM_WEIGHT * ocsvm_norm + 
    AE_WEIGHT * ae_norm
)

threshold = np.quantile(ensemble_score, 1 - CONTAMINATION)
pred_anomaly = (ensemble_score >= threshold).astype(int)

# Statistics
print(f"\n{'='*60}")
print(f"Anomaly Detection Results:")
print(f"{'='*60}")
print(f"Total samples: {len(pred_anomaly)}")
print(f"Detected anomalies: {pred_anomaly.sum()} ({100*pred_anomaly.mean():.2f}%)")
print(f"Threshold: {threshold:.6f}")
print(f"Score range: [{ensemble_score.min():.6f}, {ensemble_score.max():.6f}]")
print(f"\nIndividual model statistics:")
print(f"  ISO  - mean: {iso_norm.mean():.4f}, std: {iso_norm.std():.4f}")
print(f"  OCSVM- mean: {ocsvm_norm.mean():.4f}, std: {ocsvm_norm.std():.4f}")
print(f"  AE   - mean: {ae_norm.mean():.4f}, std: {ae_norm.std():.4f}")
print(f"{'='*60}\n")

# ----------------------------------------------------------------------------
# 10. SAVE SUBMISSION
# ----------------------------------------------------------------------------
if id_col:
    ids = df[id_col].astype(str).values
else:
    ids = [f"ANS{i+1:05d}" for i in range(len(pred_anomaly))]

submission = pd.DataFrame({
    "id": ids,
    "task5": pred_anomaly
})

submission.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved submission: {OUTPUT_CSV}")
print(submission.head(10))
print(f"\nAnomaly distribution in first 100:")
print(submission.head(100)['task5'].value_counts())

# ----------------------------------------------------------------------------
# 11. SAVE MODELS (V5)
# ----------------------------------------------------------------------------
joblib.dump(scaler, os.path.join(DATA_PATH, "scaler_V5.joblib"))
joblib.dump(ocsvm, os.path.join(DATA_PATH, "ocsvm_model_V5.joblib"))
ae.save(os.path.join(DATA_PATH, "autoencoder_model_V5.keras"))
joblib.dump(feature_columns, os.path.join(DATA_PATH, "feature_columns_V5.joblib"))

print(f"\n✅ Saved all model artifacts (V5) to: {DATA_PATH}")