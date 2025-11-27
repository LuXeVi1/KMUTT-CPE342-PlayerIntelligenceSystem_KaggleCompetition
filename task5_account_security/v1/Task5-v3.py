# ============================================================================
# 🔒 TASK 5 (V3): Account Security Monitoring - Advanced Ensemble
# Enhanced Efficiency, Robust Feature Engineering & Rank-Based Ensemble
# Optimized for Colab & Local Execution
# ============================================================================

import os
import gc
import sys
import time
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers

# ----------------------------------------------------------------------------
# LOGGING SETUP
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
class Config:
    # 1. Paths
    # Check if running in Colab
    IS_COLAB = 'google.colab' in sys.modules
    
    if IS_COLAB:
        BASE_DIR = "/content"
        DATA_DIR = "/content/task5"
    else:
        # Local execution: Try to find the dataset relative to this script
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        # Assuming standard structure: project/task5_account_security/v1/script.py
        # And data in: project/train/task5
        PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
        DATA_DIR = os.path.join(PROJECT_ROOT, "train", "task5")
        
        # Fallback if not found
        if not os.path.exists(DATA_DIR):
            DATA_DIR = CURRENT_DIR

    TEST_CSV = os.path.join(DATA_DIR, "test.csv")
    OUTPUT_CSV = os.path.join(DATA_DIR, "submission_task5_v3.csv")
    
    # 2. Hyperparameters
    RANDOM_STATE = 42
    CONTAMINATION = 0.045     # Expected anomaly rate
    OCSVM_NU = 0.045
    SAMPLE_SIZE_FOR_OCSVM = 25000  # Subsample size for SVM (efficiency)
    
    # 3. Ensemble Weights (Rank-based)
    ISO_WEIGHT = 0.40
    OCSVM_WEIGHT = 0.30
    AE_WEIGHT = 0.30
    
    # 4. Training
    BATCH_SIZE = 2048
    AE_EPOCHS = 150
    AE_BATCH_SIZE = 128

config = Config()

# ----------------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------------
def load_data(path):
    logger.info(f"Loading data from: {path}")
    if not os.path.exists(path):
        logger.error(f"❌ File not found: {path}")
        if config.IS_COLAB:
            logger.info("💡 Tip: If running on Colab, make sure to upload 'test.csv' to '/content/task5/'")
        raise FileNotFoundError(f"File not found: {path}")
    
    df = pd.read_csv(path)
    logger.info(f"✅ Data loaded. Shape: {df.shape}")
    
    # Identify ID column
    id_col = 'id'
    candidates = ['id', 'Id', 'ID', 'player_id']
    for c in candidates:
        if c in df.columns:
            id_col = c
            break
            
    return df, id_col

# ----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING (Vectorized & Efficient)
# ----------------------------------------------------------------------------
def engineer_features(df, id_col):
    logger.info("🛠️  Starting feature engineering...")
    X = df.copy()
    
    # Drop ID
    if id_col and id_col in X.columns:
        X = X.drop(columns=[id_col])
        
    # --- 1. Temporal Patterns (Vectorized) ---
    # Find groups of columns like feature_1, feature_2, ...
    prefixes = set()
    for c in X.columns:
        if c.endswith('_1') or c.endswith('_2') or c.endswith('_3') or c.endswith('_4'):
            # Extract prefix (everything before the last underscore)
            parts = c.split('_')
            if len(parts) > 1:
                prefix = "_".join(parts[:-1])
                prefixes.add(prefix)
            
    new_features = {}
    
    for prefix in prefixes:
        cols = [f"{prefix}_{i}" for i in range(1, 5)]
        # Only process if all 4 time steps exist
        if all(c in X.columns for c in cols):
            data = X[cols].values
            
            # Basic stats
            mean_val = np.nanmean(data, axis=1)
            new_features[f'{prefix}_mean'] = mean_val
            new_features[f'{prefix}_std'] = np.nanstd(data, axis=1)
            new_features[f'{prefix}_max'] = np.nanmax(data, axis=1)
            
            # Trend: Last - First
            new_features[f'{prefix}_trend'] = data[:, 3] - data[:, 0]
            
            # Volatility: Max absolute jump between consecutive periods
            diffs = np.diff(data, axis=1)
            new_features[f'{prefix}_volatility'] = np.nanmax(np.abs(diffs), axis=1)
            
            # Spike Ratio: Last / Mean (Handle div by zero)
            new_features[f'{prefix}_spike_ratio'] = data[:, 3] / (mean_val + 1e-6)

    # --- 2. Domain Specific Ratios ---
    # Skill vs Activity
    if 'skill_rating_std' in new_features and 'activity_level_mean' in new_features:
        new_features['skill_instability_per_activity'] = new_features['skill_rating_std'] / (new_features['activity_level_mean'] + 1e-5)
        
    # Spending vs Activity (Whale or Hack?)
    if 'purchase_amount_mean_mean' in new_features and 'activity_level_mean' in new_features:
        new_features['spending_efficiency'] = new_features['purchase_amount_mean_mean'] / (new_features['activity_level_mean'] + 1e-5)

    # --- 3. Risk Flags ---
    risk_cols = ['vpn_usage', 'suspicious_login_time', 'mass_item_sale', 'password_changed']
    available_risk = [c for c in risk_cols if c in X.columns]
    if available_risk:
        new_features['risk_score_sum'] = X[available_risk].sum(axis=1)

    # Create DataFrame from new features
    X_new = pd.DataFrame(new_features, index=X.index)
    
    # Combine with original numeric features
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_final = pd.concat([X[numeric_cols], X_new], axis=1)
    
    # Fill NaN with median
    X_final = X_final.fillna(X_final.median())
    
    logger.info(f"✅ Feature engineering complete. Output shape: {X_final.shape}")
    return X_final

# ----------------------------------------------------------------------------
# 3. MODELING CLASSES
# ----------------------------------------------------------------------------
class AnomalyDetector:
    def __init__(self, X_train):
        self.X_train = X_train
        self.n_samples = X_train.shape[0]
        self.n_features = X_train.shape[1]
        
    def train_isolation_forest(self):
        logger.info("🌲 Training Isolation Forest...")
        iso = IsolationForest(
            n_estimators=300,
            contamination=config.CONTAMINATION,
            max_samples=min(self.n_samples, 100000) if self.n_samples > 100000 else 'auto',
            max_features=1.0,
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
            bootstrap=True
        )
        iso.fit(self.X_train)
        # Score: negative of decision_function (so higher = more anomalous)
        return -iso.decision_function(self.X_train)

    def train_ocsvm(self):
        logger.info("⭕ Training One-Class SVM...")
        # Scale for SVM (StandardScaler is critical for SVM convergence)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_train)
        
        # Subsample for training (SVM is O(n^3) or O(n^2), so we must subsample large datasets)
        if self.n_samples > config.SAMPLE_SIZE_FOR_OCSVM:
            rng = np.random.RandomState(config.RANDOM_STATE)
            idx = rng.choice(np.arange(self.n_samples), size=config.SAMPLE_SIZE_FOR_OCSVM, replace=False)
            X_train_sub = X_scaled[idx]
        else:
            X_train_sub = X_scaled
            
        ocsvm = OneClassSVM(kernel="rbf", nu=config.OCSVM_NU, gamma="scale")
        ocsvm.fit(X_train_sub)
        
        # Batch prediction to avoid memory issues
        scores = []
        for i in range(0, self.n_samples, config.BATCH_SIZE):
            batch = X_scaled[i:i+config.BATCH_SIZE]
            scores.append(-ocsvm.decision_function(batch))
            
        return np.concatenate(scores)

    def train_autoencoder(self):
        logger.info("🧠 Training Autoencoder...")
        # Scale to [0, 1] for AE (MinMax is usually best for reconstruction)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(self.X_train)
        
        input_dim = self.n_features
        
        # Architecture
        inp = layers.Input(shape=(input_dim,))
        
        # Encoder
        x = layers.Dense(128, activation='relu')(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Bottleneck (Compressed representation)
        encoded = layers.Dense(32, activation='relu', activity_regularizer=regularizers.l1(1e-5))(x)
        
        # Decoder
        x = layers.Dense(64, activation='relu')(encoded)
        x = layers.BatchNormalization()(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        out = layers.Dense(input_dim, activation='sigmoid')(x)
        
        ae = models.Model(inp, out)
        ae.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='mse')
        
        # Callbacks
        es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        
        # Split
        X_t, X_v = train_test_split(X_scaled, test_size=0.15, random_state=config.RANDOM_STATE)
        
        ae.fit(
            X_t, X_t,
            epochs=config.AE_EPOCHS,
            batch_size=config.AE_BATCH_SIZE,
            validation_data=(X_v, X_v),
            callbacks=[es, rlr],
            verbose=1
        )
        
        # Reconstruction Error
        recon_errors = []
        for i in range(0, self.n_samples, config.BATCH_SIZE):
            batch = X_scaled[i:i+config.BATCH_SIZE]
            recon = ae.predict(batch, verbose=0)
            # MSE per sample
            err = np.mean(np.square(batch - recon), axis=1)
            recon_errors.append(err)
            
        return np.concatenate(recon_errors), ae

# ----------------------------------------------------------------------------
# 4. MAIN PIPELINE
# ----------------------------------------------------------------------------
def main():
    start_time = time.time()
    logger.info("🚀 Initializing Task 5 Pipeline V3")
    logger.info(f"📂 Data Directory: {config.DATA_DIR}")
    
    # 1. Load
    try:
        df, id_col = load_data(config.TEST_CSV)
    except FileNotFoundError:
        return

    ids = df[id_col].values if id_col else df.index.values
    
    # 2. Features
    X = engineer_features(df, id_col)
    
    # 3. Robust Scaling for Models
    # We use RobustScaler to handle outliers in input features before they go into IsoForest
    # Note: AE and SVM have their own internal scaling in the class methods
    logger.info("⚖️  Scaling data (RobustScaler)...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Train & Predict
    detector = AnomalyDetector(X_scaled)
    
    score_iso = detector.train_isolation_forest()
    score_ocsvm = detector.train_ocsvm()
    score_ae, ae_model = detector.train_autoencoder()
    
    # 5. Ensemble (Rank Averaging)
    logger.info("🤝 Ensembling scores with Rank Averaging...")
    
    # Convert scores to ranks (higher rank = more anomalous)
    rank_iso = rankdata(score_iso)
    rank_ocsvm = rankdata(score_ocsvm)
    rank_ae = rankdata(score_ae)
    
    # Normalize ranks to [0, 1]
    n = len(X)
    norm_iso = rank_iso / n
    norm_ocsvm = rank_ocsvm / n
    norm_ae = rank_ae / n
    
    # Weighted Sum
    final_score = (
        config.ISO_WEIGHT * norm_iso +
        config.OCSVM_WEIGHT * norm_ocsvm +
        config.AE_WEIGHT * norm_ae
    )
    
    # 6. Thresholding
    threshold = np.quantile(final_score, 1 - config.CONTAMINATION)
    predictions = (final_score >= threshold).astype(int)
    
    # 7. Stats & Output
    n_anomalies = predictions.sum()
    logger.info(f"{'='*40}")
    logger.info(f"📊 RESULTS")
    logger.info(f"{'='*40}")
    logger.info(f"Total Samples: {n}")
    logger.info(f"Anomalies:     {n_anomalies} ({n_anomalies/n*100:.2f}%)")
    logger.info(f"Threshold:     {threshold:.6f}")
    
    # Save
    submission = pd.DataFrame({
        "id": ids,
        "task5": predictions
    })
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(config.OUTPUT_CSV), exist_ok=True)
    
    submission.to_csv(config.OUTPUT_CSV, index=False)
    logger.info(f"💾 Submission saved to: {config.OUTPUT_CSV}")
    
    # Save Artifacts
    logger.info("📦 Saving model artifacts...")
    model_dir = os.path.dirname(config.OUTPUT_CSV)
    try:
        joblib.dump(scaler, os.path.join(model_dir, "scaler_v3.joblib"))
        ae_model.save(os.path.join(model_dir, "autoencoder_v3.keras"))
        logger.info("✅ Artifacts saved successfully.")
    except Exception as e:
        logger.warning(f"⚠️ Could not save artifacts: {e}")
    
    elapsed = time.time() - start_time
    logger.info(f"⏱️  Total execution time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
