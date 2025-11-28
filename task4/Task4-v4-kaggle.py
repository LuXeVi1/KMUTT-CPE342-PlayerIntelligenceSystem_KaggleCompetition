# ============================================================================
# 🚀 COMPLETE GAME DETECTION PIPELINE - KAGGLE READY
# ============================================================================
# Updated for test_refined.csv
# Run this entire code in ONE Kaggle notebook cell
# ============================================================================

!pip install protobuf==3.20.* --quiet
!pip install imbalanced-learn==0.12.3 --quiet

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("="*70)
print("🎮 IMPROVED GAME DETECTION PIPELINE")
print("="*70)

# ==================== IMPORTS ===============================================
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import psutil

# ML libraries
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Fix SVC import
import sklearn.svm
SVC = sklearn.svm._classes.SVC

# Deep Learning
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import regularizers

# PyTorch
import torch
import torch.nn as nn

# Advanced libraries
from transformers import AutoImageProcessor, ViTModel
import albumentations as A
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

print("✅ All libraries imported successfully!")

# ==================== SETUP PATHS (AUTO-DETECT) ============================
print("\n🔍 Auto-detecting paths...")

# List available inputs
available_inputs = os.listdir('/kaggle/input/')
print(f"Available inputs: {available_inputs}")

# Base path
BASE_PATH = "/kaggle/input/cpe342-karena"
print(f"Base path: {BASE_PATH}")
print(f"Contents: {os.listdir(BASE_PATH)}")

# Check for public_dataset folder
if os.path.exists(f"{BASE_PATH}/public_dataset"):
    DATA_PATH = f"{BASE_PATH}/public_dataset"
    print(f"✅ Found public_dataset")
    print(f"   Contents: {os.listdir(DATA_PATH)}")
else:
    DATA_PATH = BASE_PATH

# Now find task4
if os.path.exists(f"{DATA_PATH}/task4"):
    TASK4_PATH = f"{DATA_PATH}/task4"
    print(f"✅ Found task4 at: {TASK4_PATH}")
else:
    raise FileNotFoundError(f"task4 not found in {DATA_PATH}")

TRAIN_DIR = f"{TASK4_PATH}/train"
TEST_DIR = f"{TASK4_PATH}/test"
VAL_DIR = f"{TASK4_PATH}/val"

OUTPUT_DIR = "/kaggle/working"
os.makedirs(f"{OUTPUT_DIR}/models", exist_ok=True)

print(f"\n📁 Final paths:")
print(f"   Base: {BASE_PATH}")
print(f"   Data: {DATA_PATH}")
print(f"   Task4: {TASK4_PATH}")
print(f"   Output: {OUTPUT_DIR}")

# Verify paths
if os.path.exists(TASK4_PATH):
    contents = os.listdir(TASK4_PATH)
    print(f"\n✅ Task4 contents: {contents}")
    
    # Check for required directories
    required = ['train', 'val', 'test']
    missing = []
    for req in required:
        if req in contents:
            print(f"   ✓ {req}/ found")
        else:
            print(f"   ✗ {req}/ NOT found")
            missing.append(req)
    
    # Check for CSV files
    csv_files = [f for f in contents if f.endswith('.csv')]
    print(f"\n   CSV files found: {csv_files}")
    
    if missing:
        print(f"\n⚠️ Missing directories: {missing}")
        print(f"   Available in task4: {contents}")
else:
    raise FileNotFoundError(f"Task4 directory not found at {TASK4_PATH}")

# ==================== CONFIG ================================================
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 20
PATIENCE = 7
MODEL_NAME = "google/vit-base-patch16-224"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🚀 Device: {device}")

if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

ram = psutil.virtual_memory()
print(f"💾 RAM: {ram.available / (1024**3):.2f} GB / {ram.total / (1024**3):.2f} GB")

# ==================== LOAD DATA (UPDATED FOR test_refined.csv) ==============
print("\n" + "="*70)
print("📊 LOADING DATA")
print("="*70)

train_df = pd.read_csv(f"{TASK4_PATH}/train.csv")
val_df = pd.read_csv(f"{TASK4_PATH}/val.csv")

# ⬇️⬇️⬇️ UPDATED: Use test_refined.csv instead of test.csv ⬇️⬇️⬇️
test_df = pd.read_csv(f"{TASK4_PATH}/test_refined.csv")
# ⬆️⬆️⬆️ UPDATED ⬆️⬆️⬆️

print(f"✅ Data loaded:")
print(f"   Train: {train_df.shape}")
print(f"   Val: {val_df.shape}")
print(f"   Test: {test_df.shape} (from test_refined.csv)")

# Verify test images exist
sample_test_file = f"{TEST_DIR}/{test_df.iloc[0]['file_name']}"
print(f"\n🔍 Verifying test images:")
print(f"   Sample file exists: {os.path.exists(sample_test_file)}")

if not os.path.exists(sample_test_file):
    print(f"   ⚠️ Image not found at: {sample_test_file}")
    print(f"   Available dirs: {os.listdir(TASK4_PATH)}")
    # Try alternative path
    TEST_DIR_ALT = f"{TASK4_PATH}/test_refined"
    if os.path.exists(TEST_DIR_ALT):
        TEST_DIR = TEST_DIR_ALT
        print(f"   ✅ Using alternative: {TEST_DIR}")

# Encode labels
le = LabelEncoder()
train_df["label_enc"] = le.fit_transform(train_df["label"].astype(str))
val_df["label_enc"] = le.transform(val_df["label"].astype(str))
num_classes = train_df["label_enc"].nunique()

print(f"\n📊 Class distribution:")
print("Train:")
print(train_df["label"].value_counts().sort_index())
print("\nValidation:")
print(val_df["label"].value_counts().sort_index())

# Softer class weights using sqrt
class_counts = train_df["label"].value_counts().values
class_weights = np.sqrt(len(train_df) / (num_classes * class_counts))
class_weight_dict = dict(enumerate(class_weights))
print(f"\n⚖️ Softer class weights: {class_weight_dict}")

# ==================== AUGMENTATION ==========================================
print("\n🎨 Setting up augmentation...")

# Gentle augmentation for low-res images
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
])

print("✅ Augmentation ready (no TTA for better performance)")

# ==================== IMAGE LOADER ==========================================
def load_image(path, transform=None):
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_np = np.array(img)
        if transform:
            augmented = transform(image=img_np)
            img_np = augmented['image']
        return img_np
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return np.zeros((*IMG_SIZE, 3), dtype=np.uint8)

# ==================== LOAD VIT MODEL ========================================
print("\n" + "="*70)
print("🤖 LOADING VISION TRANSFORMER")
print("="*70)

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
vit_frozen = ViTModel.from_pretrained(MODEL_NAME).to(device).eval()

# Freeze all parameters
for param in vit_frozen.parameters():
    param.requires_grad = False

print("✅ ViT loaded and frozen")

# ==================== FEATURE EXTRACTION ====================================
@torch.no_grad()
def extract_vit_features(img_batch_np):
    """Extract features from ViT (last layer only)"""
    inputs = processor(images=list(img_batch_np), return_tensors="pt").to(device)
    outputs = vit_frozen(**inputs, output_hidden_states=True)
    features = outputs.last_hidden_state[:, 0, :]
    return features.detach().cpu().numpy()

def extract_features_batch(df, img_dir, desc="Processing"):
    """Extract features from all images in batches"""
    features_list = []
    labels_list = []
    
    print(f"\n{desc}: {len(df)} images")
    
    for i in range(0, len(df), BATCH):
        batch_df = df.iloc[i:i+BATCH]
        batch_images = []
        
        for _, row in batch_df.iterrows():
            img = load_image(f"{img_dir}/{row['file_name']}")
            batch_images.append(img)
        
        batch_features = extract_vit_features(np.array(batch_images))
        features_list.append(batch_features)
        
        if 'label_enc' in df.columns:
            labels_list.extend(batch_df['label_enc'].values)
        
        if (i // BATCH) % 20 == 0:
            print(f"  Progress: {i}/{len(df)} ({i/len(df)*100:.1f}%)", end="\r")
    
    features = np.vstack(features_list)
    print(f"\n✅ Extracted: {features.shape}")
    
    return (features, np.array(labels_list)) if labels_list else features

# Extract features from all datasets
print("\n" + "="*70)
print("🔄 EXTRACTING FEATURES")
print("="*70)

train_features, train_labels = extract_features_batch(train_df, TRAIN_DIR, "Train set")
val_features, val_labels = extract_features_batch(val_df, VAL_DIR, "Validation set")
test_features = extract_features_batch(test_df, TEST_DIR, "Test set")

# Save features
np.save(f"{OUTPUT_DIR}/train_features.npy", train_features)
np.save(f"{OUTPUT_DIR}/val_features.npy", val_features)
np.save(f"{OUTPUT_DIR}/test_features.npy", test_features)
print(f"\n💾 Features saved to {OUTPUT_DIR}")

# ==================== BUILD MODEL ===========================================
print("\n" + "="*70)
print("🧠 BUILDING CLASSIFIER")
print("="*70)

def build_simple_classifier(num_classes, input_dim):
    """Simple 2-layer classifier (less overfitting)"""
    inputs = tf.keras.Input(shape=(input_dim,))
    
    x = Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    
    x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=Adam(3e-4),
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"]
    )
    return model

print(f"✅ Model architecture ready")
print(f"   Input: {train_features.shape[1]} features")
print(f"   Output: {num_classes} classes")

# ==================== TRAINING ==============================================
print("\n" + "="*70)
print("🎯 TRAINING WITH VALIDATION SET")
print("="*70)

best_f1 = 0
best_model_path = None

for trial in range(3):
    print(f"\n{'='*60}")
    print(f"🔄 Trial {trial+1}/3")
    print(f"{'='*60}")
    
    model_path = f"{OUTPUT_DIR}/models/model_trial{trial+1}.keras"
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=3,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    )
    
    # Build and train
    clf = build_simple_classifier(num_classes, train_features.shape[1])
    
    history = clf.fit(
        train_features, train_labels,
        validation_data=(val_features, val_labels),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=[early_stop, reduce_lr, checkpoint],
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Evaluate
    val_preds = clf.predict(val_features, verbose=0)
    val_pred_classes = np.argmax(val_preds, axis=1)
    val_f1 = f1_score(val_labels, val_pred_classes, average="macro")
    
    print(f"\n✅ Trial {trial+1} Validation F1: {val_f1:.4f}")
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_model_path = model_path
        print(f"   🏆 New best model!")
    
    del clf
    tf.keras.backend.clear_session()
    import gc
    gc.collect()

print(f"\n{'='*60}")
print(f"📊 Best Validation F1: {best_f1:.4f}")
print(f"📁 Best model: {best_model_path}")
print(f"{'='*60}")

# ==================== LOAD BEST MODEL =======================================
print(f"\n📥 Loading best model...")
best_clf = tf.keras.models.load_model(best_model_path)

val_preds = best_clf.predict(val_features, verbose=0)
val_pred_classes = np.argmax(val_preds, axis=1)
final_val_f1 = f1_score(val_labels, val_pred_classes, average="macro")

print(f"\n📊 Final Validation Performance:")
print(f"   Macro F1: {final_val_f1:.4f}")
print(f"\n   Classification Report:")
print(classification_report(val_labels, val_pred_classes, 
                          target_names=[str(i) for i in range(num_classes)]))

print(f"\n   Confusion Matrix:")
cm = confusion_matrix(val_labels, val_pred_classes)
print(cm)

print(f"\n   Validation predictions distribution:")
unique, counts = np.unique(val_pred_classes, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"   Class {cls}: {cnt:4d} ({cnt/len(val_pred_classes)*100:5.1f}%)")

# ==================== XGBOOST ==============================================
print("\n" + "="*70)
print("🌲 TRAINING XGBOOST")
print("="*70)

xgb_model = XGBClassifier(
    max_depth=5,
    learning_rate=0.05,
    n_estimators=150,
    subsample=0.7,
    colsample_bytree=0.7,
    objective='multi:softmax',
    num_class=num_classes,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(train_features, train_labels, verbose=False)

xgb_val_preds = xgb_model.predict(val_features)
xgb_val_f1 = f1_score(val_labels, xgb_val_preds, average="macro")
print(f"✅ XGBoost Validation F1: {xgb_val_f1:.4f}")

# ==================== ENSEMBLE ==============================================
print("\n🤝 Creating ensemble...")

nn_val_probs = best_clf.predict(val_features, verbose=0)
xgb_val_probs = xgb_model.predict_proba(val_features)

ensemble_val_probs = (nn_val_probs + xgb_val_probs) / 2
ensemble_val_preds = np.argmax(ensemble_val_probs, axis=1)
ensemble_val_f1 = f1_score(val_labels, ensemble_val_preds, average="macro")

print(f"   Ensemble Validation F1: {ensemble_val_f1:.4f}")

# Choose best approach
results = {
    'Neural Network': final_val_f1,
    'XGBoost': xgb_val_f1,
    'Ensemble': ensemble_val_f1
}

best_approach = max(results, key=results.get)
final_f1 = results[best_approach]

print(f"\n🏆 Best approach: {best_approach} (F1: {final_f1:.4f})")

# ==================== FINAL PREDICTIONS =====================================
print("\n" + "="*70)
print("🎨 GENERATING FINAL TEST PREDICTIONS")
print("="*70)

nn_test_probs = best_clf.predict(test_features, verbose=0)
xgb_test_probs = xgb_model.predict_proba(test_features)

if best_approach == 'Ensemble':
    final_probs = (nn_test_probs + xgb_test_probs) / 2
elif best_approach == 'XGBoost':
    final_probs = xgb_test_probs
else:
    final_probs = nn_test_probs

final_preds = np.argmax(final_probs, axis=1)

print(f"✅ Using: {best_approach}")
print(f"   Expected F1: {final_f1:.4f}")

# ==================== PREDICTION ANALYSIS ===================================
print(f"\n📊 Test predictions distribution:")
unique, counts = np.unique(final_preds, return_counts=True)

for cls, cnt in zip(unique, counts):
    train_ratio = len(train_df[train_df['label_enc'] == cls]) / len(train_df)
    pred_ratio = cnt / len(final_preds)
    print(f"   Class {cls}: {cnt:5d} ({pred_ratio*100:5.1f}%) | Train: {train_ratio*100:5.1f}%")

# ==================== CREATE SUBMISSION =====================================
print(f"\n" + "="*70)
print("📝 CREATING SUBMISSION")
print("="*70)

submission = pd.DataFrame({
    "id": test_df["id"],
    "task4": final_preds
})

submission_path = f"{OUTPUT_DIR}/submission.csv"
submission.to_csv(submission_path, index=False)

# Save probabilities
prob_df = pd.DataFrame(final_probs, columns=[f"prob_class_{i}" for i in range(num_classes)])
prob_df["id"] = test_df["id"]
prob_df["prediction"] = final_preds
prob_df.to_csv(f"{OUTPUT_DIR}/predictions_with_probs.csv", index=False)

print(f"✅ Submission saved: {submission_path}")
print(f"✅ Probabilities saved: {OUTPUT_DIR}/predictions_with_probs.csv")

# ==================== FINAL SUMMARY =========================================
print(f"\n{'='*70}")
print("🎉 PIPELINE COMPLETE!")
print(f"{'='*70}")
print(f"\n📊 Summary:")
print(f"   Best approach: {best_approach}")
print(f"   Validation F1: {final_f1:.4f}")
print(f"   Total test samples: {len(test_df)}")
print(f"   Submission file: {submission_path}")
print(f"\n🎯 Key features:")
print(f"   ✓ Used validation set for model selection")
print(f"   ✓ Simple 2-layer classifier (less overfitting)")
print(f"   ✓ Gentle augmentation for low-res images")
print(f"   ✓ No TTA (better for 144p images)")
print(f"   ✓ Softer class weights (sqrt)")
print(f"   ✓ NN + XGBoost + Ensemble comparison")
print(f"\n📋 Sample predictions:")
print(submission.head(10))
print(f"\n✅ Ready to submit to Kaggle!")
print("="*70)