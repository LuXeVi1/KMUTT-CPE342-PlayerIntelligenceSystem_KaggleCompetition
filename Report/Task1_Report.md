# Task 1: Anti-Cheat (Cheater Detection) - Methodology Report

## 1. EDA Findings
*   **Class Imbalance:** The dataset is significantly imbalanced, with a cheater rate of approximately **35%**. This necessitated specific handling techniques like resampling.
*   **Non-Random Missingness:** Exploratory analysis revealed that "missing values" in columns like `reports_received` or `device_changes_count` were not random errors but structurally significant—often representing "zero" or "no activity." Treating these as standard missing values initially introduced bias.
*   **Feature Correlations:** Strong correlations were observed between cheating labels and specific performance metrics (e.g., abnormally high headshot ratios combined with low playtime).

## 2. Data Preprocessing
*   **Strategic Missing Indicators:** Instead of blindly imputing all missing values, we created specific boolean flags (e.g., `reports_is_missing`) for columns where absence carried information.
*   **Neutral Imputation:** To fix the "Missing = Cheater" bias, we applied a **Neutral Imputation Strategy**:
    *   **KNN Imputation:** Used for performance metrics (e.g., `accuracy`, `kill_death_ratio`) where a player's other stats could predict the missing value.
    *   **Median Imputation:** Used for count-based features to provide a safe, neutral baseline that wouldn't trigger false alarms.
*   **Feature Engineering:**
    *   **Interaction Features:** Created `aim_efficiency` (Accuracy × Headshot) and `kill_effectiveness` (KD × Headshot) to capture multi-dimensional anomalies.
    *   **Risk Flags:** Generated boolean flags for "Superhuman Aim" (Accuracy > 75% & Headshot > 75%) and "High Risk Pattern" (High reports + Frequent device changes).

## 3. Model Design
*   **Ensemble Architecture:** We implemented a diverse **5-Model Stacking Ensemble** to capture different aspects of the data:
    1.  **Random Forest:** For robust, non-linear baseline predictions.
    2.  **XGBoost (x2):** Two variations with different hyperparameters (one conservative, one aggressive).
    3.  **LightGBM:** For speed and handling of categorical features.
    4.  **CatBoost:** For its superior handling of categorical data and robust overfitting protection.
*   **Meta-Learner:** Instead of a simple average, we trained a **Meta-XGBoost** model on the predictions of the 5 base models. This allowed the system to learn *which* model was most reliable for specific types of players.
*   **Resampling:** Applied **ADASYN** (Adaptive Synthetic Sampling) and **SMOTE** to balance the training classes, ensuring the model didn't just memorize the majority class.

## 4. Evaluation & Results
*   **Primary Metric:** **F2 Score**. We prioritized Recall (catching cheaters) over Precision (avoiding false accusations) because missing a cheater has a higher negative impact on game integrity than manually reviewing a flagged legitimate player.
*   **Performance:**
    *   **Meta-Model F2 Score:** **0.8369**
    *   **Precision:** ~0.56 | **Recall:** ~0.95
*   **Threshold Optimization:** We didn't use the default 0.5 threshold. We tuned the decision threshold to **0.429**, a conservative adjustment that balanced the high recall with a reasonable precision to prevent flooding the support team with false reports.

## 5. Insights gained from model behavior
*   **The "Report" Signal:** `reports_received` and its interaction terms (e.g., `reports_squared`) were among the top predictors. This confirms that user reports are a valuable signal, but they need to be "denoised" by the model.
*   **Mechanical Anomalies:** Features like `crosshair_placement` and `headshot_percentage` were critical. The model learned that high kills *without* good crosshair placement is a strong indicator of aim-assist tools (aimbots).
*   **Behavioral Context:** `account_age_days` was a key context feature. New accounts with "god-like" stats were flagged much more aggressively than old accounts with similar stats, aligning with the behavior of "rage hackers" on burner accounts.

## 6. Common mistakes or failed experiments
*   **Failure 1: The "Missing = Cheater" Bias**
    *   *Issue:* Early models achieved artificially high scores by simply flagging anyone with missing data as a cheater.
    *   *Solution:* We implemented "Neutral Imputation" (filling with median instead of 0 or mean) to force the model to look at actual gameplay stats rather than data artifacts.
*   **Failure 2: Over-reliance on Accuracy**
    *   *Issue:* Using raw `accuracy` scores caused false positives for skilled legitimate players (smurfs).
    *   *Solution:* We created "Consistency" features (`kill_consistency`) and interaction terms (`reports_x_crosshair`). A skilled player has good crosshair placement; a cheater often doesn't, despite hitting shots.

## 7. Domain interpretation of the results
*   **Game Integrity:** The model successfully acts as a force multiplier for the Anti-Cheat team. By catching **95%** of cheaters (High Recall), it ensures the game environment remains fair.
*   **Business Impact:**
    *   **Retention:** Reducing cheaters directly improves player retention and lifetime value (LTV).
    *   **Operational Efficiency:** The "High Confidence" flags (Prob > 0.7) can be auto-banned, while "Medium Confidence" flags (0.43 - 0.7) are sent for manual review, optimizing the support team's workload.
*   **Link to Game Data:** The high importance of `crosshair_placement` suggests that future game client updates should log this metric at a higher frequency (tick rate) to further improve detection accuracy.
