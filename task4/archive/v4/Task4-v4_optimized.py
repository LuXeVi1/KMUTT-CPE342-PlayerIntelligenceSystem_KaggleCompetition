# ============================================================================
# TASK 4: GAME IMAGE CLASSIFICATION — V4 OPTIMIZED
# Improvements:
# - Modular Object-Oriented Design
# - Robust Path Handling (Local vs Colab/Kaggle)
# - Efficient Feature Extraction with ViT
# - Hybrid Ensemble (Neural Network + XGBoost)
# - Clean Logging & Error Handling
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd
import psutil
import warnings
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Deep Learning
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import regularizers
from transformers import AutoImageProcessor, ViTModel
import albumentations as A

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    # Environment Detection
    IS_KAGGLE = 'kaggle' in os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
    IS_COLAB = 'google.colab' in sys.modules
    
    # Paths
    if IS_KAGGLE:
        BASE_DIR = Path("/kaggle/input/cpe342-karena")
        OUTPUT_DIR = Path("/kaggle/working")
    elif IS_COLAB:
        BASE_DIR = Path("/content/drive/MyDrive/ML-Task4")
        OUTPUT_DIR = Path("/content/output")
    else:
        BASE_DIR = Path("./data")
        OUTPUT_DIR = Path("./output_task4")
        
    # Task Specific Paths
    TASK_DIR = BASE_DIR / "public_dataset/task4" if (BASE_DIR / "public_dataset/task4").exists() else BASE_DIR / "task4"
    TRAIN_DIR = TASK_DIR / "train"
    TEST_DIR = TASK_DIR / "test"
    VAL_DIR = TASK_DIR / "val"
    
    # Model Parameters
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 20
    PATIENCE = 7
    MODEL_NAME = "google/vit-base-patch16-224"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (self.OUTPUT_DIR / "models").mkdir(exist_ok=True)
        print(f"📂 Task Directory: {self.TASK_DIR}")
        print(f"🚀 Device: {self.DEVICE}")

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
# DATA PIPELINE
# ============================================================
class ImageDataManager:
    def __init__(self, config):
        self.cfg = config
        self.le = LabelEncoder()
        
        # Augmentation
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
        ])

    def load_metadata(self):
        Logger.section("Loading Metadata")
        
        try:
            self.train_df = pd.read_csv(self.cfg.TASK_DIR / "train.csv")
            self.val_df = pd.read_csv(self.cfg.TASK_DIR / "val.csv")
            
            # Handle Test CSV variations
            test_path = self.cfg.TASK_DIR / "test_refined.csv"
            if not test_path.exists():
                test_path = self.cfg.TASK_DIR / "test.csv"
            self.test_df = pd.read_csv(test_path)
            
            Logger.info(f"Train: {self.train_df.shape}")
            Logger.info(f"Val: {self.val_df.shape}")
            Logger.info(f"Test: {self.test_df.shape} (from {test_path.name})")
            
            # Encode Labels
            self.train_df["label_enc"] = self.le.fit_transform(self.train_df["label"].astype(str))
            self.val_df["label_enc"] = self.le.transform(self.val_df["label"].astype(str))
            self.num_classes = len(self.le.classes_)
            
            # Class Weights
            counts = self.train_df["label"].value_counts().values
            weights = np.sqrt(len(self.train_df) / (self.num_classes * counts))
            self.class_weights = dict(enumerate(weights))
            Logger.info(f"Classes: {self.num_classes}")
            
        except Exception as e:
            Logger.error(f"Failed to load metadata: {e}")
            raise

    def load_image(self, path, augment=False):
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(self.cfg.IMG_SIZE)
            img_np = np.array(img)
            if augment:
                img_np = self.transform(image=img_np)['image']
            return img_np
        except Exception as e:
            Logger.warn(f"Error loading {path}: {e}")
            return np.zeros((*self.cfg.IMG_SIZE, 3), dtype=np.uint8)

# ============================================================
# FEATURE EXTRACTION (ViT)
# ============================================================
class FeatureExtractor:
    def __init__(self, config):
        self.cfg = config
        Logger.section("Initializing ViT")
        self.processor = AutoImageProcessor.from_pretrained(config.MODEL_NAME)
        self.model = ViTModel.from_pretrained(config.MODEL_NAME).to(config.DEVICE).eval()
        
        # Freeze
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract(self, images):
        inputs = self.processor(images=list(images), return_tensors="pt").to(self.cfg.DEVICE)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def process_dataset(self, df, img_dir, data_manager, desc="Processing"):
        features = []
        labels = []
        
        print(f"\n{desc}: {len(df)} images")
        
        for i in range(0, len(df), self.cfg.BATCH_SIZE):
            batch_df = df.iloc[i:i+self.cfg.BATCH_SIZE]
            batch_imgs = [data_manager.load_image(img_dir / row['file_name']) for _, row in batch_df.iterrows()]
            
            batch_feats = self.extract(np.array(batch_imgs))
            features.append(batch_feats)
            
            if 'label_enc' in df.columns:
                labels.extend(batch_df['label_enc'].values)
                
            print(f"  Progress: {i+len(batch_df)}/{len(df)}", end="\r")
            
        print()
        return np.vstack(features), np.array(labels) if labels else None

# ============================================================
# CLASSIFIER MODELS
# ============================================================
class GameClassifier:
    def __init__(self, config, num_classes, input_dim):
        self.cfg = config
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.nn_model = None
        self.xgb_model = None

    def build_nn(self):
        inputs = tf.keras.Input(shape=(self.input_dim,))
        x = Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01))(inputs)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.num_classes, activation="softmax")(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer=Adam(3e-4), loss='sparse_categorical_crossentropy', metrics=["accuracy"])
        return model

    def train_nn(self, X_train, y_train, X_val, y_val, class_weights):
        Logger.section("Training Neural Network")
        
        best_f1 = 0
        best_path = self.cfg.OUTPUT_DIR / "models/best_nn.keras"
        
        for trial in range(3):
            print(f"\n--- Trial {trial+1}/3 ---")
            model = self.build_nn()
            
            callbacks = [
                EarlyStopping(monitor='val_accuracy', patience=self.cfg.PATIENCE, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3),
                ModelCheckpoint(str(best_path), monitor='val_accuracy', save_best_only=True, verbose=0)
            ]
            
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.cfg.EPOCHS,
                batch_size=self.cfg.BATCH_SIZE,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
            
            # Evaluate
            val_preds = np.argmax(model.predict(X_val, verbose=0), axis=1)
            f1 = f1_score(y_val, val_preds, average="macro")
            Logger.info(f"Trial {trial+1} F1: {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                self.nn_model = model
                Logger.info("🏆 New Best Model")
            
            tf.keras.backend.clear_session()
            
        return best_f1

    def train_xgb(self, X_train, y_train, X_val, y_val):
        Logger.section("Training XGBoost")
        
        self.xgb_model = XGBClassifier(
            max_depth=5, learning_rate=0.05, n_estimators=150,
            subsample=0.7, colsample_bytree=0.7, objective='multi:softmax',
            num_class=self.num_classes, random_state=42, n_jobs=-1
        )
        
        self.xgb_model.fit(X_train, y_train)
        preds = self.xgb_model.predict(X_val)
        f1 = f1_score(y_val, preds, average="macro")
        Logger.info(f"XGBoost F1: {f1:.4f}")
        return f1

    def predict_ensemble(self, X):
        nn_probs = self.nn_model.predict(X, verbose=0)
        xgb_probs = self.xgb_model.predict_proba(X)
        return (nn_probs + xgb_probs) / 2

# ============================================================
# MAIN PIPELINE
# ============================================================
if __name__ == "__main__":
    config = Config()
    
    try:
        # 1. Data Setup
        dm = ImageDataManager(config)
        dm.load_metadata()
        
        # 2. Feature Extraction
        fe = FeatureExtractor(config)
        
        Logger.section("Extracting Features")
        X_train, y_train = fe.process_dataset(dm.train_df, config.TRAIN_DIR, dm, "Train")
        X_val, y_val = fe.process_dataset(dm.val_df, config.VAL_DIR, dm, "Val")
        X_test, _ = fe.process_dataset(dm.test_df, config.TEST_DIR, dm, "Test")
        
        # Save Features (Optional backup)
        np.save(config.OUTPUT_DIR / "train_features.npy", X_train)
        np.save(config.OUTPUT_DIR / "test_features.npy", X_test)
        
        # 3. Training
        clf = GameClassifier(config, dm.num_classes, X_train.shape[1])
        
        nn_f1 = clf.train_nn(X_train, y_train, X_val, y_val, dm.class_weights)
        xgb_f1 = clf.train_xgb(X_train, y_train, X_val, y_val)
        
        # 4. Ensemble Evaluation
        ens_probs = clf.predict_ensemble(X_val)
        ens_preds = np.argmax(ens_probs, axis=1)
        ens_f1 = f1_score(y_val, ens_preds, average="macro")
        
        Logger.section("Results")
        print(f"Neural Net F1: {nn_f1:.4f}")
        print(f"XGBoost F1:    {xgb_f1:.4f}")
        print(f"Ensemble F1:   {ens_f1:.4f}")
        
        # 5. Final Prediction
        Logger.section("Generating Submission")
        
        # Select best method
        if ens_f1 >= max(nn_f1, xgb_f1):
            final_probs = clf.predict_ensemble(X_test)
            method = "Ensemble"
        elif xgb_f1 > nn_f1:
            final_probs = clf.xgb_model.predict_proba(X_test)
            method = "XGBoost"
        else:
            final_probs = clf.nn_model.predict(X_test, verbose=0)
            method = "Neural Network"
            
        final_preds = np.argmax(final_probs, axis=1)
        Logger.info(f"Selected Method: {method}")
        
        # Save
        sub = pd.DataFrame({
            "id": dm.test_df["id"],
            "task4": final_preds
        })
        sub_path = config.OUTPUT_DIR / "submission.csv"
        sub.to_csv(sub_path, index=False)
        Logger.info(f"Saved to {sub_path}")
        
        # Save Probs
        prob_df = pd.DataFrame(final_probs, columns=[f"prob_{i}" for i in range(dm.num_classes)])
        prob_df["id"] = dm.test_df["id"]
        prob_df.to_csv(config.OUTPUT_DIR / "predictions_probs.csv", index=False)
        
        Logger.section("Pipeline Complete")
        
    except Exception as e:
        Logger.error(f"Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()
