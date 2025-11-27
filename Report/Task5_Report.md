# Task 5: Account Security (Anomaly Detection) - Methodology Report

## 1. EDA Findings
*   **Unsupervised Nature:** Unlike previous tasks, this was an **Unsupervised Learning** problem. We had no "Hacked" labels to train on, only a dataset of user behaviors where we assumed a small fraction (~4.5%) were anomalous.
*   **Temporal Patterns:** The data contained time-series snapshots (e.g., `feature_1` to `feature_4`). Exploratory analysis showed that normal users have stable patterns, while compromised accounts often show sudden "spikes" or "drifts" in behavior (e.g., sudden skill drop or spending spike).

## 2. Data Preprocessing
*   **Vectorized Feature Engineering:** We focused on capturing *change* over time:
    *   **Volatility:** The maximum jump between consecutive time steps.
    *   **Trend:** The net change from $t_1$ to $t_4$.
    *   **Spike Ratio:** The ratio of the last value to the mean (detects sudden outliers).
*   **Risk Scoring:** We aggregated domain-specific flags like `vpn_usage`, `suspicious_login_time`, and `mass_item_sale` into a composite `risk_score`.
*   **Robust Scaling:** We used **RobustScaler** instead of Standard/MinMax for the Isolation Forest inputs to ensure that existing outliers didn't skew the scaling for the rest of the data.

## 3. Model Design
*   **Rank-Based Ensemble:** No single algorithm catches all types of anomalies. We combined three distinct approaches:
    1.  **Isolation Forest (40% Weight):** Best for "Point Anomalies" (values that are just too high/low).
    2.  **One-Class SVM (30% Weight):** Best for finding boundaries of "Normal" behavior in high-dimensional space.
    3.  **Autoencoder (30% Weight):** A Neural Network trained to compress and reconstruct data. High "Reconstruction Error" indicates complex, non-linear anomalies that simpler models miss.
*   **Rank Averaging:** Since these models output scores on different scales (e.g., SVM distance vs. Reconstruction MSE), we averaged their **Ranks** (percentiles) rather than raw scores to combine them fairly.

## 4. Evaluation & Results
*   **Metric:** **Anomaly Rate & Consensus**. Since we lack ground truth, we evaluated based on the stability of the detected anomalies.
*   **Performance:**
    *   **Detected Anomalies:** **1,165** accounts (4.50% of test set).
    *   **Threshold:** Dynamic thresholding at the 95.5th percentile of the combined risk score.
*   **Model Agreement:** The ensemble proved robust; the Autoencoder detected subtle pattern breaks that the Isolation Forest (which looks for coarse outliers) missed.

## 5. Insights gained from model behavior
*   **The "Burner" Profile:** The model flagged a cluster of accounts with high `volatility` in skill rating but low `account_age`. This matches the profile of "Boosters" (skilled players logging into low-rank accounts to level them up quickly).
*   **The "Hacked" Profile:** Another cluster showed stable long-term history but a sudden spike in `spending_efficiency` and `mass_item_sale`, highly indicative of an account takeover where the hacker liquidates assets.

## 6. Common mistakes or failed experiments
*   **Failure 1: Raw Score Averaging**
    *   *Experiment:* Averaging the raw output of Isolation Forest (-0.5 to 0.5) with Autoencoder MSE (0.0 to 10.0).
    *   *Result:* The Autoencoder dominated the decision simply because its numbers were bigger.
    *   *Solution:* **Rank Averaging** normalized the contributions, allowing each model to vote equally based on relative ordering.
*   **Failure 2: Ignoring Time**
    *   *Experiment:* Treating the 4 time steps as independent features.
    *   *Result:* Missed the "change" signal. A high value might be normal for a whale, but a *sudden jump* to a high value is suspicious. Vectorized temporal features fixed this.

## 7. Domain interpretation of the results
*   **Security Operations:** This system acts as a "Triage" layer.
    *   **Top 1% (Extreme Anomalies):** Trigger immediate temporary locks (e.g., "Suspicious activity detected, please reset password").
    *   **Top 5% (Moderate Anomalies):** Flag for manual review by the Trust & Safety team.
*   **False Positive Management:** By using an ensemble, we reduce false positives caused by a single model's bias (e.g., an Autoencoder might struggle with a rare but legitimate "Whale" spending spree, but the Isolation Forest might see it as less anomalous if the value is within global bounds).
