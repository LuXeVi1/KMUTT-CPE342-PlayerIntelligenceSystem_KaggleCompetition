# ============================================================================
# ✅ FULL PIPELINE (ViT Features + Classifier + SMOTE + XGBoost + Submission)
# ============================================================================

# ==================== STEP 0: INSTALL DEPENDENCIES ==========================
!pip install transformers accelerate --quiet
!pip install imbalanced-learn xgboost --quiet

# ==================== STEP 1: IMPORT LIBRARIES ==============================
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from transformers import AutoImageProcessor, AutoModel
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from google.colab import drive
from PIL import Image

# ==================== STEP 2: MOUNT GOOGLE DRIVE ===========================
drive.mount('/content/drive', force_remount=True)

BASE_DRIVE_PATH = "/content/drive/MyDrive/Task4_v3"
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
EPOCHS = 5
BATCH = 32
MODEL_NAME = "google/vit-base-patch16-224"
HIDDEN_DIM = 768

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch device:", device)

# ==================== STEP 4: LOAD DATA ====================================
train_df = pd.read_csv(f"{DATA_PATH}/train.csv")
test_df  = pd.read_csv(f"{DATA_PATH}/test.csv")

le = LabelEncoder()
train_df["label_enc"] = le.fit_transform(train_df["label"].astype(str))

print(train_df.head())

# ==================== STEP 5: LOAD VIT MODEL ===============================
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
vit = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

# ==================== STEP 6: IMAGE LOADER =================================
def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    return np.array(img)

train_images = np.array([load_image(f"{TRAIN_DIR}/{f}") for f in train_df["file_name"]])
train_labels = train_df["label_enc"].values
print("✅ Train images:", train_images.shape)

# ==================== STEP 7: EXTRACT VIT FEATURES =========================
@torch.no_grad()
def extract_vit_features(img_batch_np):
    inputs = processor(images=list(img_batch_np), return_tensors="pt").to(device)
    outputs = vit(**inputs)
    cls = outputs.last_hidden_state[:, 0, :]
    return cls.detach().cpu().numpy()

vit_features_list = []
for i in range(0, len(train_images), BATCH):
    feats = extract_vit_features(train_images[i:i+BATCH])
    vit_features_list.append(feats)

vit_features = np.vstack(vit_features_list)
print("✅ ViT features shape:", vit_features.shape)

# ==================== STEP 8: BUILD CLASSIFIER =============================
def build_classifier(num_classes=5, input_dim=HIDDEN_DIM):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = BatchNormalization()(inputs)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.35)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.25)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# ==================== STEP 9: STRATIFIED K-FOLD TRAIN =====================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
num_classes = train_df["label_enc"].nunique()

oof_preds = np.zeros((len(train_df), num_classes), dtype=np.float32)
fold_results = []
fold_models = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(vit_features, train_labels), 1):
    print(f"\n===== FOLD {fold}/{N_SPLITS} =====")
    X_tr, X_val = vit_features[tr_idx], vit_features[val_idx]
    y_tr, y_val = train_labels[tr_idx], train_labels[val_idx]

    clf = build_classifier(num_classes=num_classes, input_dim=HIDDEN_DIM)
    clf.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH, verbose=1)

    preds = clf.predict(X_val, batch_size=BATCH, verbose=0)
    oof_preds[val_idx] = preds

    f1 = f1_score(y_val, np.argmax(preds, axis=1), average="macro")
    print(f"✅ Fold {fold} F1:", f1)

    fold_path = f"{SAVE_MODEL_DIR}/vit_cls_fold{fold}_v3.keras"
    clf.save(fold_path)
    fold_models.append(fold_path)

    fold_results.append({"fold": fold, "f1_macro": f1})
    pd.DataFrame(fold_results).to_csv(f"{SAVE_LOG_DIR}/results_v3.csv", index=False)

# ==================== STEP 10: SMOTE + XGBOOST (v3 FIXED) =================
print("\n🎯 SMOTE + XGBoost (บน ViT features) — FIXED")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(vit_features, train_labels)
print("✅ After SMOTE:", {int(c): int((y_res == c).sum()) for c in np.unique(y_res)})

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softmax",
    num_class=num_classes,
    tree_method="hist",       # ✅ safe for all Colab versions
    random_state=42
)
xgb.fit(X_res, y_res)
xgb.save_model(f"{SAVE_MODEL_DIR}/task4_xgb_v3.json")
print("✅ Saved XGBoost model → task4_xgb_v3.json")

# ==================== STEP 11: FINAL CV SCORE =============================
val_pred_all = np.argmax(oof_preds, axis=1)
f1_macro = f1_score(train_labels, val_pred_all, average="macro")
print("\n✅ FINAL MACRO F1 (v3):", f1_macro)
print(classification_report(train_labels, val_pred_all, target_names=le.classes_))
pd.DataFrame({"metric": ["macro_f1"], "value": [f1_macro]}).to_csv(f"{SAVE_LOG_DIR}/summary_v3.csv", index=False)

# ==================== STEP 12: TEST INFERENCE & SUBMISSION =================
test_images = np.array([load_image(f"{TEST_DIR}/{f}") for f in test_df["file_name"]])
test_feats_list = []
for i in range(0, len(test_images), BATCH):
    feats = extract_vit_features(test_images[i:i+BATCH])
    test_feats_list.append(feats)
test_features = np.vstack(test_feats_list)
print("✅ Test features shape:", test_features.shape)

# Load all fold classifiers and average predictions
pred_stack = []
for fold_path in fold_models:
    clf = tf.keras.models.load_model(fold_path)
    prob = clf.predict(test_features, batch_size=BATCH, verbose=0)
    pred_stack.append(prob)

test_prob = np.mean(pred_stack, axis=0)
test_labels_enc = np.argmax(test_prob, axis=1)

submission = pd.DataFrame({"id": test_df["id"], "task4": test_labels_enc})
submission_path = f"{SAVE_LOG_DIR}/submission_task4_v3.csv"
submission.to_csv(submission_path, index=False)

print("\n📁 Saved all v3 results to Google Drive:")
print("✅ Models     :", SAVE_MODEL_DIR)
print("✅ Logs       :", SAVE_LOG_DIR)
print("✅ Submission :", submission_path)
