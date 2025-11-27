# Team Collaboration & Roles

## 1. Role of Each Member

### **1. 65070501094 - Nuttaya Ngamsard**
*   **Role:** **Lead Data Scientist - Task 1 (Anti-Cheat)**
*   **Responsibilities**
    *   Designed the Cheater Detection Pipeline.
    *   Handled the complex class imbalance problem using ADASYN/SMOTE.
    *   Implemented the "Neutral Imputation" strategy to fix bias.

### **2. 66070501036 - Ponprathan Kuearung**
*   **Role:** **Lead Data Scientist - Task 2 (Player Segmentation)**
*   **Responsibilities**
    *   Developed the multi-class segmentation logic.
    *   Engineered behavioral features (Engagement, Social Scores).
    *   Optimized the Voting Classifier ensemble.

### **3. 66070501046 - Ratchamongkol Mongkoldit**
*   **Role:** **Lead Data Scientist - Task 3 (Spending Prediction)**
*   **Responsibilities**
    *   Solved the "Zero-Inflation" problem with the Two-Stage model.
    *   Implemented the Log-Transformation strategy for regression.
    *   Fine-tuned the probability thresholds for monetization prediction.

### **4. 66070501062 - Arkkhanirut Pandej**
*   **Role:** **Lead Computer Vision Engineer - Task 4 (Image Classification)**
*   **Responsibilities**
    *   Implemented the Vision Transformer (ViT) feature extraction.
    *   Managed the image augmentation pipeline (Albumentations).
    *   Handled the distribution shift between training and validation sets.

### **5. 66070501068 - Khunnapat Aubontara**
*   **Role:** **Lead Security Engineer - Task 5 (Account Security)**
*   **Responsibilities**
    *   Designed the Unsupervised Anomaly Detection system.
    *   Created the Rank-Based Ensemble (Isolation Forest + Autoencoder).
    *   Engineered vectorized temporal features to detect account takeovers.

---

## 2. Key Takeaways from Teamwork

### **Unified Architecture & Standards**
*   **The "Template" Approach:** The team successfully enforced a strict **Object-Oriented Design (OOP)** pattern across all 5 tasks. Every task uses a consistent Config class, Logger utility, and Pipeline structure.
*   **Benefit:** This standardization meant that any team member could jump into another member's code (e.g., to help debug Task 3) and immediately understand the flow, significantly reducing friction.

### **Knowledge Transfer (Cross-Pollination)**
*   **Shared Techniques:** We observed that successful techniques were shared across tasks. For example, the **XGBoost/LightGBM/CatBoost** stack was mastered in Task 1 and then effectively reapplied in Task 2 and Task 3.
*   **Feature Engineering:** The concept of "Interaction Features" (multiplying two columns) was used in both Anti-Cheat (Task 1) and Segmentation (Task 2), showing that the team communicated about what methods were driving performance.

### **Iterative Development Workflow**
*   **Version Control:** The project structure (`v1`, `v2`, `v3`) demonstrates an iterative workflow. Team members didn't just settle for the first result; they improved their specific modules through multiple versions (e.g., Task 1 moving from simple Random Forest to a Meta-Model Stack).
*   **Independent Execution:** By decoupling the tasks into separate folders and pipelines, the team could work in parallel without merge conflicts, maximizing productivity within the timeline.
