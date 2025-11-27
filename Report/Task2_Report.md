# Task 2: Player Segmentation (Clustering & Classification) - Methodology Report

## 1. EDA Findings
*   **Class Imbalance:** The dataset showed a distinct imbalance among the 4 player segments.
    *   **Class 0 (Casual?):** ~39.4% (Majority)
    *   **Class 3 (Whale/Hardcore?):** ~15.4% (Minority)
    *   *Implication:* The model would naturally bias towards predicting Class 0 without intervention.
*   **Feature Distributions:** Spending and Playtime features were highly skewed (long-tail distributions), typical of gaming data where a small percentage of players contribute the most activity/revenue.

## 2. Data Preprocessing
*   **Feature Engineering:** We moved beyond raw stats to "Behavioral Metrics":
    *   **Engagement Score:** `avg_session_duration` × `play_frequency` (Captures total dedication).
    *   **Spending Intensity:** `total_spending` / `account_age` (Captures value per day).
    *   **Social Engagement:** `friend_count` × `team_play_percentage` (Differentiates solo vs. social players).
*   **Imputation:**
    *   **Numeric:** Median imputation was used to handle missing values robustly against outliers.
    *   **Categorical:** Missing categorical values were explicitly labeled as "Unknown" to preserve the information that data was missing (which can be a signal in itself).
*   **Balancing:** We applied **SMOTE** (Synthetic Minority Over-sampling Technique) to perfectly balance the training set, ensuring each of the 4 classes had equal representation (25% each) during training.

## 3. Model Design
*   **Ensemble Strategy:** We used a **Soft Voting Classifier** to combine the strengths of three gradient boosting giants:
    1.  **XGBoost (Weight 1.2):** The primary driver, excellent at structured data.
    2.  **LightGBM (Weight 1.0):** Adds diversity and speed.
    3.  **CatBoost (Weight 1.1):** Included for its superior handling of categorical features without extensive preprocessing.
*   **Why Voting?** Individual models often make different errors on "borderline" players. Averaging their predicted probabilities (Soft Voting) smooths out these errors and provides a more stable classification.

## 4. Evaluation & Results
*   **Metric:** **F1-Macro Score**. We chose Macro-average because we care equally about identifying the smaller, high-value segments (Class 3) as we do the massive casual base (Class 0).
*   **Performance:**
    *   **Average F1-Macro:** **0.8085** (consistent across 5 folds).
    *   **Stability:** The standard deviation was very low (±0.0026), indicating the model is robust and not overfitting to specific data splits.
*   **Test Predictions:** The model predicted Class 0 (41%) and Class 1 (24.6%) most frequently, closely mirroring the natural distribution of the training data, which suggests it didn't overfit to the synthetic SMOTE samples.

## 5. Insights gained from model behavior
*   **Distinct Segments:** The high F1 score (~0.81) confirms that the 4 player segments are statistically distinct and can be reliably separated using just gameplay and spending data.
*   **Value of Derived Features:** The "Intensity" features (e.g., spending per hour, playtime per day) were more predictive than raw totals. A player who spends 1000 THB in 1 day is very different from one who spends 1000 THB over 3 years, even if their "Total Spending" is identical.

## 6. Common mistakes or failed experiments
*   **Failure 1: Ignoring Imbalance**
    *   *Issue:* Initial experiments without SMOTE resulted in a model that had high accuracy (by predicting Class 0 correctly) but failed to identify the valuable Class 3 players (Recall < 40%).
    *   *Solution:* SMOTE fixed this, raising the minority class performance significantly.
*   **Failure 2: Over-complex Stacking**
    *   *Issue:* We tried a complex Meta-Learner (Stacking) similar to Task 1, but it didn't improve performance over simple Voting and added deployment complexity.
    *   *Solution:* We reverted to the simpler Weighted Voting mechanism, which was just as accurate but easier to maintain.

## 7. Domain interpretation of the results
*   **Targeting Strategy:**
    *   **Class 0 (Casual):** The largest group. Strategy: Retention and conversion to Class 1.
    *   **Class 3 (High Value):** The VIPs. Strategy: Exclusive content and high-touch support to prevent churn.
*   **Social Factor:** The `social_engagement` feature proved important, suggesting that "Social" players might be a distinct stability anchor for the game economy—they stay for their friends, even if they don't spend as much individually.
