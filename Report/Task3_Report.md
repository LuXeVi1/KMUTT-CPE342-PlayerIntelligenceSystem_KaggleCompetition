# Task 3: Spending Prediction (Regression) - Methodology Report

## 1. EDA Findings
*   **Zero-Inflation:** The most critical finding was that approximately **48%** of players spend exactly **0 THB**.
*   **Distribution:** Among spenders, the distribution is highly right-skewed (a few "whales" spend massive amounts), which breaks standard regression assumptions (Normality).
*   **Implication:** A standard regression model trained on the whole dataset would likely predict small positive values for everyone (e.g., 50 THB), which is wrong for half the population (who spend 0) and wrong for the whales (who spend 10,000+).

## 2. Data Preprocessing
*   **Target Transformation:** We applied `log1p` (Log(x+1)) transformation to the target variable `spending_30d` to compress the massive range of spending values into a more normal distribution for the regression models.
*   **Feature Engineering:**
    *   **Historical Context:** `historical_spending` was identified as the single most predictive feature.
    *   **Binary Flag:** Created an `ever_spent` flag to explicitly tell the model if a player has a history of monetization.
    *   **Engagement-Monetization Ratios:** Calculated `spending_per_session` to normalize spending behavior against playtime.

## 3. Model Design
*   **Two-Stage Stacking (Hurdle Model):** We decomposed the problem into two distinct sub-problems:
    1.  **Stage 1: Classification (The "Will They Spend?" Model)**
        *   *Goal:* Predict the *probability* that a player spends > 0.
        *   *Models:* XGBoost + LightGBM Classifiers.
    2.  **Stage 2: Regression (The "How Much?" Model)**
        *   *Goal:* Predict the *amount* assuming they are a spender.
        *   *Training Data:* Trained **only** on players with `spending_30d > 0`.
        *   *Models:* XGBoost + LightGBM Regressors.
*   **Final Prediction:** `Final_Amount = (Probability > Threshold) * (Predicted_Amount)`

## 4. Evaluation & Results
*   **Metric:** **Normalized MAE** (Mean Absolute Error). This metric penalizes errors relative to the total spending, which is appropriate for financial forecasting.
*   **Performance:**
    *   **OOF Normalized MAE:** **0.7881**
    *   **Absolute Error:** The model's average error was ~2,788 THB per player.
*   **Threshold Optimization:** We didn't just use 0.5 as the cutoff. We optimized the probability threshold to **0.5055**, meaning we only predict a positive spending amount if the model is >50.5% confident the player is a spender.

## 5. Insights gained from model behavior
*   **Past Predicts Future:** The strongest predictor of future spending is past spending. This suggests that spending habits are "sticky"—once a player becomes a spender, they tend to remain one.
*   **The "Conversion" Barrier:** The Classification stage (Stage 1) was often more critical than the Regression stage. Accurately identifying *who* is a spender is harder than guessing *how much* a known spender will buy.

## 6. Common mistakes or failed experiments
*   **Failure 1: Single-Stage Regression**
    *   *Experiment:* Training a single XGBoost Regressor on the entire dataset (zeros included).
    *   *Result:* The model performed poorly, predicting "safe" low values (e.g., 100 THB) for non-spenders, resulting in a huge cumulative error.
*   **Failure 2: Linear Models**
    *   *Experiment:* Using Linear Regression for the second stage.
    *   *Result:* Failed to capture the non-linear relationship between engagement (playtime) and spending (money). Tree-based models (XGB/LGBM) worked much better.

## 7. Domain interpretation of the results
*   **Monetization Strategy:** The Two-Stage model mirrors the actual business funnel:
    1.  **Conversion (Stage 1):** Marketing teams should focus on players with high predicted probability but 0 actual spending (potential first-time buyers).
    2.  **Upselling (Stage 2):** For existing spenders, the Regression model helps identify "Whales" who might be under-spending relative to their potential.
*   **Financial Planning:** The aggregated predictions allow the finance team to forecast next month's total revenue with reasonable accuracy (~79% Normalized Accuracy), aiding in budget planning.
