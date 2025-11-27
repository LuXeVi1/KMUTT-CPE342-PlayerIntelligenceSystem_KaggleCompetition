# 🎮 Player Intelligence System
### Advanced Machine Learning Framework for Game Operations
**Course:** CPE342-ML | **Semester:** 1/2024

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=for-the-badge&logo=pytorch)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn)

---

## Project Overview
The **Player Intelligence System** is a modular Machine Learning pipeline designed to solve critical challenges in the modern online gaming lifecycle. By leveraging state-of-the-art algorithms—from Gradient Boosting to Vision Transformers—this system transforms raw player data into actionable insights for **Fair Play**, **Monetization**, and **Security**.

## 🚀 Key Modules (Intelligence Engines)

### 🛡️ Module 1: Anti-Cheat Engine
**Objective:** Detect malicious actors and aim-assist tools without intrusive kernel drivers.
*   **Tech:** Stacked Ensemble (XGBoost, CatBoost, RF), ADASYN for imbalance.
*   **Performance:** **95% Recall** (F2 Score: 0.8369).
*   **Key Insight:** Behavioral signals like `crosshair_placement` and `kill_consistency` are robust indicators of cheating.

### 🧩 Module 2: Player Segmentation Engine
**Objective:** Classify players into behavioral archetypes (Casual, Hardcore, Whale) for personalized experiences.
*   **Tech:** Soft Voting Classifier, SMOTE, Behavioral Feature Engineering.
*   **Performance:** **81% F1-Macro**.
*   **Key Insight:** "Intensity" metrics (spending/hour) are more predictive than raw totals.

### 💰 Module 3: Spending Forecast Engine
**Objective:** Predict future player spending to optimize game economy and marketing.
*   **Tech:** Two-Stage Hurdle Model (Classification + Regression).
*   **Performance:** **~79% Normalized Accuracy** (Normalized MAE: 0.7881).
*   **Key Insight:** Solved the "Zero-Inflation" problem where 48% of players spend nothing.

### 🖼️ Module 4: Game Vision Engine
**Objective:** Automate content tagging and genre classification from low-resolution screenshots.
*   **Tech:** Vision Transformer (ViT) Feature Extractor + Neural Network Head.
*   **Performance:** **65% F1-Macro** on low-res data.
*   **Key Insight:** Transformers generalize significantly better than CNNs on pixelated/compressed images.

### 🔒 Module 5: Account Sentinel Engine
**Objective:** Detect compromised accounts and "Booster" activity via unsupervised learning.
*   **Tech:** Rank-Based Ensemble (Isolation Forest, One-Class SVM, Autoencoder).
*   **Performance:** Detected **1,165 anomalies** (4.5% rate) with high consensus.
*   **Key Insight:** Vectorized temporal features capture the "sudden drift" characteristic of account takeovers.

---

## Technology Stack
*   **Languages:** Python 3.9+
*   **Data Processing:** Pandas, NumPy, Scikit-learn, Imbalanced-learn
*   **Machine Learning:** XGBoost, LightGBM, CatBoost, RandomForest
*   **Deep Learning:** TensorFlow/Keras, PyTorch, Transformers (Hugging Face)
*   **Computer Vision:** Albumentations, PIL, OpenCV
*   **Utils:** Joblib, Matplotlib, Seaborn

---

## 👥 Development Team
**King Mongkut's University of Technology Thonburi**

| Student ID      | Name                     | Role                            |
| :-------------- | :----------------------- | :------------------------------ |
| **65070501094** | Nuttaya Ngamsard         | Lead CV Engineer (Task 4)       |
| **66070501036** | Ponprathan Kuearung      | Lead Data Scientist (Task 3)    |
| **66070501046** | Ratchamongkol Mongkoldit | Lead Data Scientist (Task 1)    |
| **66070501062** | Arkkhanirut Pandej       | Lead Security Engineer (Task 5) |
| **66070501068** | Khunnapat Aubontara      | Lead Data Scientist (Task 2)    |

---

## 📂 Project Structure
```
Player-Intelligence-System/
├── task1_anti_cheat/       # Cheater Detection Module
├── task2_player_segment/   # Segmentation Module
├── task3_spending_pred/    # Monetization Module
├── task4_game_vision/      # Image Classification Module
├── task5_account_sec/      # Anomaly Detection Module
├── Report/                 # Documentation & Reports
└── README.md               # Project Landing Page
```

---
*Generated for CPE342-ML Project Submission.*
