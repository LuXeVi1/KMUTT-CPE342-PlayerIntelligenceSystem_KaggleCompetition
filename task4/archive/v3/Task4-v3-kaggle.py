!pip install protobuf==3.20.*

# ============================================================================
# 🚀 KAGGLE SETUP - NO INTERNET REQUIRED VERSION
# ============================================================================

import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("📦 CHECKING KAGGLE PRE-INSTALLED LIBRARIES")
print("="*70)

import sys

required_libs = {
    'numpy': 'numpy',
    'pandas': 'pandas', 
    'sklearn': 'scikit-learn',
    'tensorflow': 'tensorflow',
    'torch': 'torch',
    'PIL': 'Pillow',
    'cv2': 'opencv-python',
}

optional_libs = {
    'transformers': 'transformers',
    'albumentations': 'albumentations',
    'xgboost': 'xgboost',
    'optuna': 'optuna',
    'imblearn': 'imbalanced-learn',
}

print("\n✅ Required libraries (should be pre-installed):")
missing_required = []
for module, package in required_libs.items():
    try:
        lib = __import__(module)
        version = getattr(lib, '__version__', 'unknown')
        print(f"   ✓ {package}: {version}")
    except ImportError:
        print(f"   ✗ {package}: NOT FOUND")
        missing_required.append(package)

print("\n📦 Optional libraries (for advanced features):")
missing_optional = []
for module, package in optional_libs.items():
    try:
        lib = __import__(module)
        version = getattr(lib, '__version__', 'unknown')
        print(f"   ✓ {package}: {version}")
    except ImportError:
        print(f"   ✗ {package}: NOT FOUND")
        missing_optional.append(package)

# ==================== FINAL IMPORTS =========================================
print("\n" + "="*70)
print("📚 IMPORTING LIBRARIES")
print("="*70)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML libraries
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ---------------------- FIX FOR SVC IMPORT BUG ----------------------
import sklearn.svm
SVC = sklearn.svm._classes.SVC
# -------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression

# Deep Learning
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import regularizers

# PyTorch
import torch
import torch.nn as nn

# Image processing
from PIL import Image
import cv2

print("✅ Core libraries imported successfully!")

try:
    from transformers import AutoImageProcessor, ViTModel
    print("✅ transformers imported")
    HAS_TRANSFORMERS = True
except ImportError:
    print("⚠️ transformers not available - ViT features will not work")
    HAS_TRANSFORMERS = False

try:
    import albumentations as A
    print("✅ albumentations imported")
    HAS_ALBUMENTATIONS = True
except ImportError:
    print("⚠️ albumentations not available - using basic augmentation")
    HAS_ALBUMENTATIONS = False

try:
    from xgboost import XGBClassifier
    print("✅ xgboost imported")
    HAS_XGBOOST = True
except ImportError:
    print("⚠️ xgboost not available - using RandomForest instead")
    HAS_XGBOOST = False

try:
    import optuna
    print("✅ optuna imported")
    HAS_OPTUNA = True
except ImportError:
    print("⚠️ optuna not available - using default hyperparameters")
    HAS_OPTUNA = False

try:
    from imblearn.over_sampling import SMOTE
    print("✅ imbalanced-learn imported")
    HAS_IMBLEARN = True
except ImportError:
    print("⚠️ imbalanced-learn not available - skipping SMOTE")
    HAS_IMBLEARN = False

try:
    import joblib
    print("✅ joblib imported")
except ImportError:
    print("⚠️ joblib not available - will use pickle instead")
    import pickle as joblib

# Memory check
import psutil

print("\n" + "="*70)
print("⚙️ CONFIGURATION")
print("="*70)



# ==================== KAGGLE PATHS ==========================================
DATA_PATH = "/kaggle/input/cpe342-karena"  # Update this!
TRAIN_DIR = f"{DATA_PATH}/task4/train"
TEST_DIR = f"{DATA_PATH}/task4/test"

OUTPUT_DIR = "/kaggle/working"
SAVE_MODEL_DIR = f"{OUTPUT_DIR}/models"
SAVE_LOG_DIR = f"{OUTPUT_DIR}/logs"
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
os.makedirs(SAVE_LOG_DIR, exist_ok=True)

print(f"📁 Data path: {DATA_PATH}")
print(f"💾 Output path: {OUTPUT_DIR}")

# Verify data
if os.path.exists(DATA_PATH):
    print(f"✅ Data directory found")
    contents = os.listdir(DATA_PATH)
    print(f"   Contents: {contents}")
else:
    print(f"❌ Data path not found!")
    print(f"   Available inputs: {os.listdir('/kaggle/input/')}")
    print("\n💡 Update DATA_PATH to match your competition!")

# ==================== HARDWARE CHECK ========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🚀 Device: {device}")

if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

ram = psutil.virtual_memory()
print(f"💾 RAM: {ram.available / (1024**3):.2f} GB / {ram.total / (1024**3):.2f} GB")

# ==================== MODEL CONFIG ==========================================
IMG_SIZE = (224, 224)
N_SPLITS = 5
EPOCHS = 20
BATCH = 16
MODEL_NAME = "google/vit-base-patch16-224"
HIDDEN_DIM = 768
PATIENCE = 7

print("\n⚙️ Model configuration:")
print(f"   Image size: {IMG_SIZE}")
print(f"   Batch size: {BATCH}")
print(f"   Epochs: {EPOCHS}")
print(f"   K-Folds: {N_SPLITS}")

# ==================== FEATURE FLAGS =========================================
print("\n🎯 Available features:")
print(f"   ViT features: {'✅' if HAS_TRANSFORMERS else '❌'}")
print(f"   Advanced augmentation: {'✅' if HAS_ALBUMENTATIONS else '❌'}")
print(f"   XGBoost: {'✅' if HAS_XGBOOST else '❌'}")
print(f"   Hyperparameter tuning: {'✅' if HAS_OPTUNA else '❌'}")
print(f"   SMOTE: {'✅' if HAS_IMBLEARN else '❌'}")

if not HAS_TRANSFORMERS:
    print("\n⚠️ WARNING: transformers not available!")
    print("   This pipeline REQUIRES transformers for ViT.")
    print("   Please enable Internet in Kaggle Settings and install.")

print("\n" + "="*70)
print("✅ SETUP COMPLETE!")
print("="*70)

!pip install imbalanced-learn==0.12.3 --quiet

from imblearn.over_sampling import SMOTE
print("SMOTE ready!")


# ==================== STEP 3: CONFIG =======================================
IMG_SIZE = (224, 224)
N_SPLITS = 5
EPOCHS = 20
BATCH = 16  # Adjust based on Kaggle GPU memory
MODEL_NAME = "google/vit-base-patch16-224"
HIDDEN_DIM = 768
PATIENCE = 7

device = "cuda" if torch.cuda.is_available() else "cpu"
print("🚀 Torch device:", device)

# Check Kaggle resources
import psutil
ram = psutil.virtual_memory()
print(f"💾 Available RAM: {ram.available / (1024**3):.2f} GB / {ram.total / (1024**3):.2f} GB")

# Check GPU
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

# ============================================================================
# 🚀 IMPROVED PIPELINE - FIX OVERFITTING & USE VALIDATION SET
# ============================================================================
# Key improvements:
# 1. Use provided validation set for model selection
# 2. Simpler architecture (less overfitting)
# 3. Better augmentation for low-res images
# 4. Remove unnecessary stacking
# 5. Add proper test-time evaluation
# ============================================================================

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from PIL import Image
import psutil

# ==================== IMPORTS ===============================================
import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

from transformers import AutoImageProcessor, ViTModel
import albumentations as A
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import regularizers

print("="*70)
print("🎮 IMPROVED GAME DETECTION PIPELINE")
print("="*70)

# ==================== SETUP =================================================
DATA_PATH = "/kaggle/input/cpe342-karena"
TASK4_PATH = f"{DATA_PATH}/task4"
TRAIN_DIR = f"{TASK4_PATH}/train"
TEST_DIR = f"{TASK4_PATH}/test"
VAL_DIR = f"{TASK4_PATH}/val"  # USE THIS!

OUTPUT_DIR = "/kaggle/working"
os.makedirs(f"{OUTPUT_DIR}/models", exist_ok=True)

# Config - SIMPLIFIED
IMG_SIZE = (224, 224)
BATCH = 32  # Increased for faster training
EPOCHS = 20
PATIENCE = 7
MODEL_NAME = "google/vit-base-patch16-224"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Device: {device}")

# ==================== LOAD DATA =============================================
print("\n📊 Loading data...")
train_df = pd.read_csv(f"{TASK4_PATH}/train.csv")
val_df = pd.read_csv(f"{TASK4_PATH}/val.csv")  # USE VALIDATION!
test_df = pd.read_csv(f"{TASK4_PATH}/test.csv")

print(f"Train: {train_df.shape} | Val: {val_df.shape} | Test: {test_df.shape}")

# Encode labels
le = LabelEncoder()
train_df["label_enc"] = le.fit_transform(train_df["label"].astype(str))
val_df["label_enc"] = le.transform(val_df["label"].astype(str))
num_classes = train_df["label_enc"].nunique()

print(f"\n📊 Train distribution:")
print(train_df["label"].value_counts().sort_index())
print(f"\n📊 Val distribution:")
print(val_df["label"].value_counts().sort_index())

# Class weights (less aggressive)
class_counts = train_df["label"].value_counts().values
class_weights = np.sqrt(len(train_df) / (num_classes * class_counts))  # sqrt for softer weights
class_weight_dict = dict(enumerate(class_weights))
print(f"\n⚖️ Softer class weights: {class_weight_dict}")

# ==================== BETTER AUGMENTATION ===================================
print("\n🎨 Setting up augmentation for low-res images...")

# For 144p low-res images, be careful with augmentation!
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),  # Reduced
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),  # Reduced
    A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),  # Light noise only
])

# NO TTA - it hurts performance on low-res images!
print("✅ Simplified augmentation (no TTA)")

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
    except:
        return np.zeros((*IMG_SIZE, 3), dtype=np.uint8)

# ==================== LOAD VIT ==============================================
print("\n🤖 Loading ViT...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
vit_frozen = ViTModel.from_pretrained(MODEL_NAME).to(device).eval()
for param in vit_frozen.parameters():
    param.requires_grad = False

# ==================== FEATURE EXTRACTION ====================================
@torch.no_grad()
def extract_vit_features(img_batch_np):
    inputs = processor(images=list(img_batch_np), return_tensors="pt").to(device)
    outputs = vit_frozen(**inputs, output_hidden_states=True)
    
    # Use ONLY last layer (simpler, less overfit)
    features = outputs.last_hidden_state[:, 0, :]
    return features.detach().cpu().numpy()

def extract_features_batch(df, img_dir, desc="Processing"):
    features_list = []
    labels_list = []
    
    print(f"{desc}: {len(df)} images")
    for i in range(0, len(df), BATCH):
        batch_df = df.iloc[i:i+BATCH]
        batch_images = [load_image(f"{img_dir}/{row['file_name']}") 
                       for _, row in batch_df.iterrows()]
        
        batch_features = extract_vit_features(np.array(batch_images))
        features_list.append(batch_features)
        
        if 'label_enc' in df.columns:
            labels_list.extend(batch_df['label_enc'].values)
        
        if (i // BATCH) % 20 == 0:
            print(f"  {i}/{len(df)}", end="\r")
    
    features = np.vstack(features_list)
    print(f"\n✅ {features.shape}")
    return (features, np.array(labels_list)) if labels_list else features

# Extract ALL features
print("\n" + "="*70)
print("🔄 EXTRACTING FEATURES")
print("="*70)

train_features, train_labels = extract_features_batch(train_df, TRAIN_DIR, "Train")
val_features, val_labels = extract_features_batch(val_df, VAL_DIR, "Val")
test_features = extract_features_batch(test_df, TEST_DIR, "Test")

# Save
np.save(f"{OUTPUT_DIR}/train_features.npy", train_features)
np.save(f"{OUTPUT_DIR}/val_features.npy", val_features)
np.save(f"{OUTPUT_DIR}/test_features.npy", test_features)

# ==================== SIMPLER MODEL =========================================
print("\n🧠 Building SIMPLER model (less overfit)...")

def build_simple_classifier(num_classes, input_dim):
    """Simpler model with less dropout"""
    inputs = tf.keras.Input(shape=(input_dim,))
    
    # Single hidden layer
    x = Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = Dropout(0.3)(x)  # Less dropout
    x = BatchNormalization()(x)
    
    x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=Adam(3e-4),  # Higher LR
        loss='sparse_categorical_crossentropy',  # Standard loss
        metrics=["accuracy"]
    )
    return model

# ==================== TRAIN WITH VALIDATION SET =============================
print("\n" + "="*70)
print("🎯 TRAINING WITH VALIDATION SET")
print("="*70)

best_f1 = 0
best_model_path = None

for trial in range(3):  # Try 3 times with different seeds
    print(f"\n🔄 Trial {trial+1}/3")
    
    model_path = f"{OUTPUT_DIR}/models/model_trial{trial+1}.keras"
    
    early_stop = EarlyStopping(
        monitor='val_accuracy',  # Monitor accuracy instead
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
    
    # Build fresh model
    clf = build_simple_classifier(num_classes, train_features.shape[1])
    
    # Train
    history = clf.fit(
        train_features, train_labels,
        validation_data=(val_features, val_labels),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=[early_stop, reduce_lr, checkpoint],
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Evaluate on validation
    val_preds = clf.predict(val_features, verbose=0)
    val_pred_classes = np.argmax(val_preds, axis=1)
    val_f1 = f1_score(val_labels, val_pred_classes, average="macro")
    
    print(f"\n✅ Trial {trial+1} Val F1: {val_f1:.4f}")
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_model_path = model_path
        print(f"   🏆 New best model!")
    
    del clf
    tf.keras.backend.clear_session()

print(f"\n{'='*60}")
print(f"📊 Best Validation F1: {best_f1:.4f}")
print(f"📁 Best model: {best_model_path}")
print(f"{'='*60}")

# ==================== LOAD BEST MODEL =======================================
print(f"\n📥 Loading best model...")
best_clf = tf.keras.models.load_model(best_model_path)

# Final validation check
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

# Check class distribution
print(f"\n   Validation predictions distribution:")
unique, counts = np.unique(val_pred_classes, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"   Class {cls}: {cnt} ({cnt/len(val_pred_classes)*100:.1f}%)")

# ==================== XGBOOST AS BACKUP =====================================
print("\n🌲 Training XGBoost as backup...")

xgb_model = XGBClassifier(
    max_depth=5,  # Shallower
    learning_rate=0.05,  # Lower LR
    n_estimators=150,
    subsample=0.7,
    colsample_bytree=0.7,
    objective='multi:softmax',
    num_class=num_classes,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(train_features, train_labels, verbose=False)

# Evaluate XGBoost on validation
xgb_val_preds = xgb_model.predict(val_features)
xgb_val_f1 = f1_score(val_labels, xgb_val_preds, average="macro")
print(f"   XGBoost Val F1: {xgb_val_f1:.4f}")

# ==================== CHOOSE BEST MODEL =====================================
if xgb_val_f1 > final_val_f1:
    print(f"\n🌲 XGBoost is better! Using XGBoost for final predictions")
    use_xgb = True
    final_model_f1 = xgb_val_f1
else:
    print(f"\n🧠 Neural Network is better! Using NN for final predictions")
    use_xgb = False
    final_model_f1 = final_val_f1

# ==================== ENSEMBLE (SIMPLE AVERAGE) =============================
print("\n🤝 Creating simple ensemble (NN + XGBoost average)...")

nn_val_probs = best_clf.predict(val_features, verbose=0)
xgb_val_probs = xgb_model.predict_proba(val_features)

# Simple average
ensemble_val_probs = (nn_val_probs + xgb_val_probs) / 2
ensemble_val_preds = np.argmax(ensemble_val_probs, axis=1)
ensemble_val_f1 = f1_score(val_labels, ensemble_val_preds, average="macro")

print(f"   Ensemble Val F1: {ensemble_val_f1:.4f}")

# Choose final approach
if ensemble_val_f1 > max(final_val_f1, xgb_val_f1):
    print(f"\n🤝 Ensemble is best! Using ensemble")
    use_ensemble = True
    final_f1 = ensemble_val_f1
else:
    use_ensemble = False
    final_f1 = final_model_f1

# ==================== FINAL TEST PREDICTIONS ================================
print("\n" + "="*70)
print("🎨 GENERATING FINAL TEST PREDICTIONS")
print("="*70)

nn_test_probs = best_clf.predict(test_features, verbose=0)
xgb_test_probs = xgb_model.predict_proba(test_features)

if use_ensemble:
    final_probs = (nn_test_probs + xgb_test_probs) / 2
    approach = "Ensemble (NN + XGBoost)"
elif use_xgb:
    final_probs = xgb_test_probs
    approach = "XGBoost only"
else:
    final_probs = nn_test_probs
    approach = "Neural Network only"

final_preds = np.argmax(final_probs, axis=1)

print(f"\n✅ Using: {approach}")
print(f"   Expected F1 (from validation): {final_f1:.4f}")

# ==================== CHECK PREDICTION DISTRIBUTION =========================
print(f"\n📊 Test predictions distribution:")
unique, counts = np.unique(final_preds, return_counts=True)
for cls, cnt in zip(unique, counts):
    train_ratio = len(train_df[train_df['label_enc'] == cls]) / len(train_df)
    pred_ratio = cnt / len(final_preds)
    print(f"   Class {cls}: {cnt:5d} ({pred_ratio*100:5.1f}%) - Train: {train_ratio*100:5.1f}%")

# ==================== CREATE SUBMISSION =====================================
submission = pd.DataFrame({
    "id": test_df["id"],
    "task4": final_preds
})

submission_path = f"{OUTPUT_DIR}/submission.csv"
submission.to_csv(submission_path, index=False)

# Also save probabilities for analysis
prob_df = pd.DataFrame(final_probs, columns=[f"prob_class_{i}" for i in range(num_classes)])
prob_df["id"] = test_df["id"]
prob_df["prediction"] = final_preds
prob_df.to_csv(f"{OUTPUT_DIR}/predictions_with_probs.csv", index=False)

print(f"\n{'='*70}")
print("🎉 IMPROVED PIPELINE COMPLETE!")
print(f"{'='*70}")
print(f"📊 Summary:")
print(f"   Validation F1: {final_f1:.4f}")
print(f"   Approach: {approach}")
print(f"   Submission: {submission_path}")
print(f"\n🎯 Key improvements:")
print(f"   ✓ Used validation set for model selection")
print(f"   ✓ Simpler model (less overfit)")
print(f"   ✓ Better augmentation for low-res")
print(f"   ✓ No aggressive TTA")
print(f"   ✓ Softer class weights")
print("="*70) 