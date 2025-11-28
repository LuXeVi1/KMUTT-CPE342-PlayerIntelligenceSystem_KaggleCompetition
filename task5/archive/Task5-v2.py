# ============================================================================
# 🔒 TASK 5 (V2): Account Security Monitoring - Optimized Ensemble
# Improved Feature Engineering + Robust Scaling + Rank Averaging
# ============================================================================

import os
import numpy as np
import pandas as pd
import gc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
# Adjust paths as needed
BASE_DIR = r"c:\Users\akani\OneDrive\Documents\Antigracity\Player-Intelligence-System---CPE342-ML---ML01"
DATA_PATH = os.path.join(BASE_DIR, "train", "task5")
TEST_CSV = os.path.join(DATA_PATH, "test.csv")
OUTPUT_CSV = os.path.join(DATA_PATH, "submission_task5_v2.csv")

RANDOM_STATE = 42
CONTAMINATION = 0.045  # Estimated anomaly rate
OCSVM_NU = CONTAMINATION
SAMPLE_SIZE_FOR_OCSVM = 20000  # Increased sample size

# Ensemble weights (can be tuned)
ISO_WEIGHT = 0.40
OCSVM_WEIGHT = 0.30
AE_WEIGHT = 0.30

# ----------------------------------------------------------------------------
# 1. DATA LOADING & PREPROCESSING
# ----------------------------------------------------------------------------
def load_data(path):
    print(f"Loading data from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    df = pd.read_csv(path)
    print(f"Loaded shape: {df.shape}")
    
    # Identify ID column
    id_col = 'id'
    if 'id' not in df.columns:
        # Try to find other ID candidates or create one
        for c in ['Id', 'ID', 'player_id']:
            if c in df.columns:
                id_col = c
                break
    
    return df, id_col

# ----------------------------------------------------------------------------
# 2. ADVANCED FEATURE ENGINEERING
# ----------------------------------------------------------------------------
def create_features(df, id_col):
    print("Generating features...")
    X = df.copy()
    
    # Drop ID for training
    if id_col and id_col in X.columns:
        X = X.drop(columns=[id_col])
    
    # Helper to get columns by suffix
    def get_cols(suffix):
        return [c for c in X.columns if c.endswith(suffix)]
    
    # Helper to get columns by prefix (without suffix)
    def get_series_cols(prefix):
        return [f"{prefix}_{i}" for i in range(1, 5) if f"{prefix}_{i}" in X.columns]

    # 1. Temporal Aggregations (Mean, Std, Trend)
    # Identify time-series groups (columns ending in _1, _2, _3, _4)
    prefixes = set()
    for c in X.columns:
        if c[-2:] in ['_1', '_2', '_3', '_4']:
            prefixes.add(c[:-2])
            
    for prefix in prefixes:
        cols = [f"{prefix}_{i}" for i in range(1, 5)]
        if all(c in X.columns for c in cols):
            data = X[cols].values
            
            # Mean and Std
            X[f'{prefix}_mean'] = np.nanmean(data, axis=1)
            X[f'{prefix}_std'] = np.nanstd(data, axis=1)
            
            # Trend (Last - First)
            X[f'{prefix}_trend'] = data[:, 3] - data[:, 0]
            
            # Max Jump (Max absolute difference between consecutive periods)
            diffs = np.diff(data, axis=1)
            X[f'{prefix}_max_jump'] = np.nanmax(np.abs(diffs), axis=1)

    # 2. Risk Flags Score
    risk_flags = ['vpn_usage', 'suspicious_login_time', 'mass_item_sale', 'password_changed']
    available_flags = [c for c in risk_flags if c in X.columns]
    if available_flags:
        X['risk_score'] = X[available_flags].sum(axis=1)
    
    # 3. Specific Domain Features
    
    # Skill Volatility vs Activity
    if 'skill_rating_std' in X.columns and 'activity_level_mean' in X.columns:
        X['skill_instability_per_activity'] = X['skill_rating_std'] / (X['activity_level_mean'] + 1e-5)

    # High Spending with Low Activity (Potential hacked account draining)
    if 'purchase_amount_mean_mean' in X.columns and 'activity_level_mean' in X.columns:
        X['spending_per_activity'] = X['purchase_amount_mean_mean'] / (X['activity_level_mean'] + 1e-5)
        
    # Location & Device Entropy
    if 'location_changes_mean' in X.columns and 'device_count_mean' in X.columns:
        X['access_complexity'] = X['location_changes_mean'] * X['device_count_mean']

    # Fill NaNs
    X = X.fillna(X.median())
    
    # Drop original time-series columns to reduce noise and dimensionality
    # (Optional: keep them if models can handle high dim, but aggregating is usually better for anomalies)
    # For now, we keep everything + new features, but let's drop the raw ones if we have aggregates
    # Actually, let's keep them, IsoForest handles high dim okay.
    
    print(f"Features generated: {X.shape[1]}")
    return X

# ----------------------------------------------------------------------------
# 3. MODELS
# ----------------------------------------------------------------------------

def train_isolation_forest(X):
    print("Training Isolation Forest...")
    iso = IsolationForest(
        n_estimators=300,
        contamination=CONTAMINATION,
        max_samples=0.8, # Subsample to reduce overfitting
        max_features=1.0,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    iso.fit(X)
    # Decision function: lower is more anomalous. We want higher = anomaly.
    # Scikit-learn: average path length. Negative is anomaly.
    # So -decision_function gives positive scores for anomalies.
    scores = -iso.decision_function(X)
    return scores

def train_ocsvm(X):
    print("Training One-Class SVM...")
    # Scale specifically for SVM (it's sensitive)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Sample if too large
    if X_scaled.shape[0] > SAMPLE_SIZE_FOR_OCSVM:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(np.arange(X_scaled.shape[0]), size=SAMPLE_SIZE_FOR_OCSVM, replace=False)
        X_train = X_scaled[idx]
    else:
        X_train = X_scaled
        
    ocsvm = OneClassSVM(kernel="rbf", nu=OCSVM_NU, gamma="scale")
    ocsvm.fit(X_train)
    
    # Batch prediction
    scores = []
    batch_size = 5000
    for i in range(0, len(X_scaled), batch_size):
        batch = X_scaled[i:i+batch_size]
        scores.append(-ocsvm.decision_function(batch))
    
    return np.concatenate(scores)

def train_autoencoder(X):
    print("Training Autoencoder...")
    # Scale to [0, 1] or Standard. MinMax is often good for AE reconstruction.
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    input_dim = X_scaled.shape[1]
    
    # Architecture
    input_layer = layers.Input(shape=(input_dim,))
    
    # Encoder
    x = layers.Dense(128, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Bottleneck
    encoded = layers.Dense(32, activation='relu')(x)
    
    # Decoder
    x = layers.Dense(64, activation='relu')(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    output_layer = layers.Dense(input_dim, activation='sigmoid')(x) # Sigmoid for MinMax scaled data
    
    autoencoder = models.Model(input_layer, output_layer)
    
    autoencoder.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
    
    # Train
    X_train, X_val = train_test_split(X_scaled, test_size=0.1, random_state=RANDOM_STATE)
    
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    
    autoencoder.fit(
        X_train, X_train,
        epochs=100,
        batch_size=128,
        validation_data=(X_val, X_val),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Calculate reconstruction error
    reconstructions = autoencoder.predict(X_scaled, batch_size=512)
    mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
    
    return mse, autoencoder

# ----------------------------------------------------------------------------
# 4. MAIN PIPELINE
# ----------------------------------------------------------------------------
def main():
    # 1. Load
    df, id_col = load_data(TEST_CSV)
    ids = df[id_col] if id_col else df.index
    
    # 2. Features
    X = create_features(df, id_col)
    
    # 3. Scaling (Robust Scaler for outliers)
    print("Scaling data...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 4. Train Models & Get Scores
    scores_iso = train_isolation_forest(X_scaled)
    scores_ocsvm = train_ocsvm(X_scaled)
    scores_ae, ae_model = train_autoencoder(X_scaled)
    
    # 5. Rank Averaging (More robust than MinMax averaging)
    print("Ensembling scores...")
    rank_iso = rankdata(scores_iso)
    rank_ocsvm = rankdata(scores_ocsvm)
    rank_ae = rankdata(scores_ae)
    
    # Normalize ranks to 0-1
    n = len(X)
    norm_iso = rank_iso / n
    norm_ocsvm = rank_ocsvm / n
    norm_ae = rank_ae / n
    
    # Weighted Ensemble
    final_score = (
        ISO_WEIGHT * norm_iso +
        OCSVM_WEIGHT * norm_ocsvm +
        AE_WEIGHT * norm_ae
    )
    
    # 6. Thresholding
    threshold = np.quantile(final_score, 1 - CONTAMINATION)
    predictions = (final_score >= threshold).astype(int)
    
    print(f"Anomalies detected: {predictions.sum()} ({predictions.mean()*100:.2f}%)")
    
    # 7. Save Submission
    submission = pd.DataFrame({
        "id": ids,
        "task5": predictions
    })
    
    submission.to_csv(OUTPUT_CSV, index=False)
    print(f"Submission saved to {OUTPUT_CSV}")
    
    # 8. Save Artifacts
    print("Saving artifacts...")
    joblib.dump(scaler, os.path.join(DATA_PATH, "scaler_v2.joblib"))
    joblib.dump(scores_iso, os.path.join(DATA_PATH, "scores_iso.joblib"))
    ae_model.save(os.path.join(DATA_PATH, "autoencoder_v2.keras"))

if __name__ == "__main__":
    main()
