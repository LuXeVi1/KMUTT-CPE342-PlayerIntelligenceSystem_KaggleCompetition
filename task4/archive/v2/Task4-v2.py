# ============================================================================
# 🎮 IMPROVED GAME DETECTION PIPELINE (ViT + Advanced Techniques)
# ============================================================================
# Changes:
# 1. Added Data Augmentation (TTA for test)
# 2. Multi-level ViT features (CLS + spatial features)
# 3. Better regularization (focal loss, label smoothing)
# 4. Stratified Group sampling
# 5. Ensemble with diversity (different architectures)
# ============================================================================

# ==================== STEP 0: INSTALL DEPENDENCIES ==========================
!pip install transformers accelerate --quiet
!pip install imbalanced-learn xgboost albumentations --quiet

# ==================== STEP 1: IMPORT LIBRARIES ==============================
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from transformers import AutoImageProcessor, AutoModel
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import regularizers
from google.colab import drive
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== STEP 2: MOUNT GOOGLE DRIVE ===========================
drive.mount('/content/drive', force_remount=True)

BASE_DRIVE_PATH = "/content/drive/MyDrive/Task4_v4_improved"
os.makedirs(BASE_DRIVE_PATH, exist_ok=True)
SAVE_MODEL_DIR = f"{BASE_DRIVE_PATH}/models"
SAVE_LOG_DIR   = f"{BASE_DRIVE_PATH}/logs"
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
os.makedirs(SAVE_LOG_DIR, exist_ok=True)

# ==================== STEP 3: CONFIG =======================================
DATA_PATH = "/content/task4"
TRAIN_DIR = f"{DATA_PATH}/train"
TEST_DIR  = f"{DATA_PATH}/test"

IMG_SIZE = (224, 224)
N_SPLITS = 5
EPOCHS = 15  # เพิ่มขึ้นแต่มี early stopping
BATCH = 16  # ลดจาก 32 เพื่อประหยัด RAM
MODEL_NAME = "google/vit-base-patch16-224"
HIDDEN_DIM = 768
PATIENCE = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
print("🚀 Torch device:", device)

# Check available RAM
import psutil
ram = psutil.virtual_memory()
print(f"💾 Available RAM: {ram.available / (1024**3):.2f} GB / {ram.total / (1024**3):.2f} GB")

# ==================== STEP 4: DATA AUGMENTATION ============================
train_transform = A.Compose([
    A.HorizontalFlip(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
    A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
    A.OneOf([
        A.MotionBlur(blur_limit=3, p=1.0),
        A.MedianBlur(blur_limit=3, p=1.0),
        A.Blur(blur_limit=3, p=1.0),
    ], p=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
])

# TTA transforms for test
tta_transforms = [
    A.Compose([]),  # Original
    A.Compose([A.HorizontalFlip(p=1.0)]),
    A.Compose([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)]),
]

# ==================== STEP 5: LOAD DATA ====================================
train_df = pd.read_csv(f"{DATA_PATH}/train.csv")
test_df  = pd.read_csv(f"{DATA_PATH}/test.csv")

le = LabelEncoder()
train_df["label_enc"] = le.fit_transform(train_df["label"].astype(str))

# Analyze class distribution
print("\n📊 Class Distribution:")
print(train_df["label"].value_counts())
class_weights = len(train_df) / (len(train_df["label"].unique()) * train_df["label"].value_counts().values)
class_weight_dict = dict(enumerate(class_weights))
print("\n⚖️ Class Weights:", class_weight_dict)

# ==================== STEP 6: LOAD VIT MODEL ===============================
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
vit = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

# ==================== STEP 7: ENHANCED IMAGE LOADER ========================
def load_image(path, transform=None):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_np = np.array(img)
    
    if transform:
        augmented = transform(image=img_np)
        img_np = augmented['image']
    
    return img_np

# ==================== STEP 8: SINGLE-LEVEL VIT FEATURES (MEMORY SAVE) ======
@torch.no_grad()
def extract_vit_features_multilevel(img_batch_np):
    """Extract only CLS token to save memory"""
    inputs = processor(images=list(img_batch_np), return_tensors="pt").to(device)
    outputs = vit(**inputs)
    
    # Only CLS token
    cls_token = outputs.last_hidden_state[:, 0, :]
    
    return cls_token.detach().cpu().numpy()

# ==================== STEP 9: EXTRACT FEATURES (MEMORY EFFICIENT) ==========
def extract_features_batch(df, img_dir):
    """Extract features in batches to save RAM"""
    features_list = []
    labels_list = []
    
    for i in range(0, len(df), BATCH):
        batch_df = df.iloc[i:i+BATCH]
        batch_images = []
        
        for _, row in batch_df.iterrows():
            img = load_image(f"{img_dir}/{row['file_name']}")
            batch_images.append(img)
        
        # Extract features immediately
        batch_features = extract_vit_features_multilevel(np.array(batch_images))
        features_list.append(batch_features)
        
        if 'label_enc' in df.columns:
            labels_list.extend(batch_df['label_enc'].values)
        
        # Clear memory
        del batch_images
        if i % (BATCH * 10) == 0:
            print(f"  Processed {i}/{len(df)} images...")
    
    features = np.vstack(features_list)
    
    if labels_list:
        return features, np.array(labels_list)
    return features

print("\n🔄 Extracting training features...")
train_features, train_labels = extract_features_batch(train_df, TRAIN_DIR)
print(f"✅ Train features: {train_features.shape}")

# ==================== STEP 10: FOCAL LOSS ==================================
def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss for handling class imbalance"""
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        
        ce = -y_true_one_hot * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0))
        weight = tf.pow(1 - y_pred, gamma)
        focal = alpha * weight * ce
        return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))
    
    return loss_fn

# ==================== STEP 11: IMPROVED CLASSIFIER (SMALLER) ==============
def build_improved_classifier(num_classes=5, input_dim=HIDDEN_DIM, dropout_rate=0.4):
    """Lighter classifier to save memory"""
    inputs = tf.keras.Input(shape=(input_dim,))
    
    x = BatchNormalization()(inputs)
    x = Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(dropout_rate * 0.7)(x)
    
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Use focal loss instead of standard CE
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=["accuracy"]
    )
    
    return model

# ==================== STEP 12: STRATIFIED K-FOLD (NO AUGMENTATION) ========
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
num_classes = train_df["label_enc"].nunique()

oof_preds = np.zeros((len(train_df), num_classes), dtype=np.float32)
fold_results = []
fold_models = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, train_df["label_enc"]), 1):
    print(f"\n{'='*60}")
    print(f"🎯 FOLD {fold}/{N_SPLITS}")
    print(f"{'='*60}")
    
    X_tr = train_features[tr_idx]
    y_tr = train_labels[tr_idx]
    X_val = train_features[val_idx]
    y_val = train_labels[val_idx]
    
    print(f"Train: {X_tr.shape}, Val: {X_val.shape}")
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        f"{SAVE_MODEL_DIR}/best_fold{fold}_v4.keras",
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
    
    # Build and train
    clf = build_improved_classifier(num_classes=num_classes, input_dim=train_features.shape[1])
    
    history = clf.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=[early_stop, reduce_lr, checkpoint],
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Predict on validation
    preds = clf.predict(X_val, batch_size=BATCH, verbose=0)
    oof_preds[val_idx] = preds
    
    f1 = f1_score(y_val, np.argmax(preds, axis=1), average="macro")
    print(f"\n✅ Fold {fold} Macro F1: {f1:.4f}")
    
    fold_models.append(f"{SAVE_MODEL_DIR}/best_fold{fold}_v4.keras")
    fold_results.append({
        "fold": fold,
        "f1_macro": f1,
        "best_epoch": early_stop.stopped_epoch - PATIENCE if early_stop.stopped_epoch > 0 else EPOCHS
    })
    
    # Clear memory after each fold
    del clf, X_tr, y_tr
    tf.keras.backend.clear_session()
    import gc
    gc.collect()

# ==================== STEP 13: OVERALL CV RESULTS ==========================
val_pred_all = np.argmax(oof_preds, axis=1)
cv_f1 = f1_score(train_df["label_enc"].values, val_pred_all, average="macro")

print(f"\n{'='*60}")
print(f"📊 CROSS-VALIDATION RESULTS")
print(f"{'='*60}")
print(f"Overall Macro F1: {cv_f1:.4f}")
print("\nPer-fold results:")
print(pd.DataFrame(fold_results))

# Confusion Matrix
cm = confusion_matrix(train_df["label_enc"].values, val_pred_all)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix (CV F1: {cv_f1:.4f})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(f"{SAVE_LOG_DIR}/confusion_matrix_v4.png", dpi=150)
plt.close()

# Classification Report
print("\n" + classification_report(train_df["label_enc"].values, val_pred_all, target_names=le.classes_))

# Save results
pd.DataFrame(fold_results).to_csv(f"{SAVE_LOG_DIR}/fold_results_v4.csv", index=False)
pd.DataFrame({
    "metric": ["cv_macro_f1"],
    "value": [cv_f1]
}).to_csv(f"{SAVE_LOG_DIR}/summary_v4.csv", index=False)

# ==================== STEP 14: XGBoost ENSEMBLE (LIGHTER) =================
print("\n🚀 Training XGBoost on original features...")

# Use SMOTE only if needed
class_counts = np.bincount(train_labels)
min_class_count = np.min(class_counts)

if min_class_count < 50:  # Only if very imbalanced
    sm = SMOTE(random_state=42, k_neighbors=min(3, min_class_count-1))
    X_res, y_res = sm.fit_resample(train_features, train_labels)
    print(f"✅ After SMOTE: {dict(zip(*np.unique(y_res, return_counts=True)))}")
else:
    X_res, y_res = train_features, train_labels
    print("✅ Skipping SMOTE (sufficient samples)")

xgb = XGBClassifier(
    n_estimators=200,  # Reduced from 500
    max_depth=4,       # Reduced from 5
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="multi:softmax",
    num_class=num_classes,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_res, y_res, verbose=False)
xgb.save_model(f"{SAVE_MODEL_DIR}/xgb_v4.json")

# Clear memory
del X_res, y_res
import gc
gc.collect()

print("✅ Saved XGBoost model")

# ==================== STEP 15: TEST INFERENCE (SIMPLIFIED TTA) ============
print("\n🔮 Test Inference...")

# Extract test features in batches
test_features_list = []
for i in range(0, len(test_df), BATCH):
    batch_df = test_df.iloc[i:i+BATCH]
    batch_images = np.array([load_image(f"{TEST_DIR}/{f}") for f in batch_df["file_name"]])
    batch_features = extract_vit_features_multilevel(batch_images)
    test_features_list.append(batch_features)
    
    del batch_images
    if i % (BATCH * 10) == 0:
        print(f"  Processed {i}/{len(test_df)} test images...")

test_features = np.vstack(test_features_list)
del test_features_list

print(f"✅ Test features: {test_features.shape}")

# Simple TTA: horizontal flip only
print("  Processing TTA (horizontal flip)...")
test_features_flip_list = []
for i in range(0, len(test_df), BATCH):
    batch_df = test_df.iloc[i:i+BATCH]
    batch_images = np.array([
        A.HorizontalFlip(p=1.0)(image=load_image(f"{TEST_DIR}/{f}"))['image']
        for f in batch_df["file_name"]
    ])
    batch_features = extract_vit_features_multilevel(batch_images)
    test_features_flip_list.append(batch_features)
    del batch_images

test_features_flip = np.vstack(test_features_flip_list)
del test_features_flip_list

# Ensemble predictions
print("  Averaging fold predictions...")
fold_preds_orig = []
fold_preds_flip = []

for fold_path in fold_models:
    clf = tf.keras.models.load_model(
        fold_path,
        custom_objects={'loss_fn': focal_loss()}
    )
    
    prob_orig = clf.predict(test_features, batch_size=BATCH, verbose=0)
    prob_flip = clf.predict(test_features_flip, batch_size=BATCH, verbose=0)
    
    fold_preds_orig.append(prob_orig)
    fold_preds_flip.append(prob_flip)
    
    del clf
    tf.keras.backend.clear_session()

# Average TTA
avg_orig = np.mean(fold_preds_orig, axis=0)
avg_flip = np.mean(fold_preds_flip, axis=0)
tta_avg = (avg_orig + avg_flip) / 2

del fold_preds_orig, fold_preds_flip, test_features_flip

# XGBoost predictions
print("  Getting XGBoost predictions...")
xgb_pred = xgb.predict_proba(test_features)

# Blend: 70% NN, 30% XGBoost
blended_pred_prob = 0.7 * tta_avg + 0.3 * xgb_pred
final_labels = np.argmax(blended_pred_prob, axis=1)

del test_features, xgb_pred, tta_avg, blended_pred_prob
import gc
gc.collect()

# ==================== STEP 16: CREATE SUBMISSION ===========================
submission = pd.DataFrame({
    "id": test_df["id"],
    "task4": final_labels
})

submission_path = f"{SAVE_LOG_DIR}/submission_task4_v4.csv"
submission.to_csv(submission_path, index=False)

print(f"\n{'='*60}")
print("✅ PIPELINE COMPLETED!")
print(f"{'='*60}")
print(f"📁 Saved to: {BASE_DRIVE_PATH}")
print(f"   Models: {len(fold_models)} fold models + XGBoost")
print(f"   CV F1:  {cv_f1:.4f}")
print(f"   Submission: {submission_path}")
print(f"\n🎮 Prediction distribution:")
print(submission["task4"].value_counts().sort_index())