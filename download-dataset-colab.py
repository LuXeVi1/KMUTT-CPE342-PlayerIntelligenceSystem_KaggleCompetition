# ============================================================================
# SECTION 1: ENVIRONMENT SETUP & DATA LOADING
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, fbeta_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Install additional libraries
!pip install xgboost lightgbm catboost imbalanced-learn -q

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
from imblearn.over_sampling import SMOTE

# For Task 4 (Image Classification)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from PIL import Image
import cv2

print("✓ All libraries imported successfully")

# ============================================================================
# MOUNT GOOGLE DRIVE (if using Drive) or UPLOAD ZIP
# ============================================================================

!pip install -q kaggle
from google.colab import files
print("Please upload your kaggle.json file")
uploaded = files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
print("\nDownloading the dataset from Kaggle...")
!kaggle competitions download -c cpe342-karena
print("\nUnzipping the dataset...")
!unzip -q cpe342-karena.zip -d .
print("\nSetup complete! Dataset is ready.")


# Set your data path
# DATA_PATH = '/content/drive/MyDrive/KMUTT/Year3_Term1/CPE342-MachineLearning/MachineLearning-KaggleCompetition'  # Adjust this path
DATA_PATH = '/content/'  # If uploaded directly

print(f"Data path set to: {DATA_PATH}")