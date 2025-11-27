# Task 4: Game Image Classification (Computer Vision) - Methodology Report

## 1. EDA Findings
*   **Data Quality:** The dataset consists of game screenshots with varying quality, often very low resolution (~144p), making fine-grained feature detection difficult for traditional CNNs.
*   **Distribution Shift:** A critical finding was the **Distribution Shift** between the Training/Validation sets and the Test set.
    *   *Train:* Class 4 was ~10%.
    *   *Validation:* Class 4 spiked to ~21.5%.
    *   *Implication:* Models trained solely to minimize training loss would fail to generalize. We had to rely heavily on the Validation set for model selection.

## 2. Data Preprocessing
*   **Augmentation:** To handle the low-quality images and prevent overfitting, we used **Albumentations**:
    *   *Techniques:* HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast, and GaussNoise.
    *   *Goal:* Simulate different streaming qualities and artifacts.
*   **Feature Extraction (The Key):** Instead of training a CNN from scratch (which requires massive data), we used **Transfer Learning** with a **Vision Transformer (ViT)** (`google/vit-base-patch16-224`).
    *   *Process:* We passed every image through the frozen ViT to extract a rich 768-dimensional feature vector, effectively converting the "Image Problem" into a "Tabular Problem."

## 3. Model Design
*   **Hybrid Ensemble Experiment:** We compared two approaches on top of the ViT features:
    1.  **Neural Network (The Winner):** A simple 2-layer MLP (256 -> 128 units) with Dropout and Batch Normalization.
    2.  **XGBoost:** A gradient boosting classifier.
*   **Selection:** The Neural Network significantly outperformed XGBoost.
    *   *Why?* Neural Networks are naturally better at handling the high-dimensional, continuous embedding space output by Transformers, whereas Tree-based models (XGBoost) struggle to find split points in such dense vectors.

## 4. Evaluation & Results
*   **Metric:** **Macro F1 Score**.
*   **Performance:**
    *   **Validation F1:** **0.6515**
    *   **XGBoost F1:** 0.4845 (Failed experiment).
*   **Test Predictions:** The final model predicted Class 3 (28.5%) and Class 1 (21.4%) as the most common game genres in the test set.

## 5. Insights gained from model behavior
*   **Transformers > CNNs for Low Res:** ViT proved exceptionally robust to the low-resolution images. Its attention mechanism likely focuses on global semantic structures (UI elements, color palettes) rather than local textures, which are often pixelated in this dataset.
*   **Simplicity Wins:** Once powerful features (ViT) were extracted, a simple Neural Network head was sufficient. Adding complexity (like the XGBoost ensemble) actually hurt performance.

## 6. Common mistakes or failed experiments
*   **Failure 1: XGBoost on Embeddings**
    *   *Experiment:* We hypothesized that XGBoost would dominate as it did in Tasks 1-3.
    *   *Result:* It performed poorly (F1 0.48 vs NN 0.65).
    *   *Lesson:* Gradient Boosting is King of Tabular Data, but Neural Networks are Queen of Embeddings.
*   **Failure 2: Ignoring Class Weights**
    *   *Issue:* The model initially ignored the minority classes.
    *   *Solution:* We implemented "Soft Class Weights" (Square Root of inverse frequency) to gently nudge the model towards minority classes without causing instability.

## 7. Domain interpretation of the results
*   **Automated Tagging:** This system allows for the automated categorization of user-generated content (streams, screenshots) without manual tagging, enabling better content recommendation systems.
*   **Scalability:** The "Feature Extraction" approach is highly scalable. We can process millions of images using the frozen ViT (fast inference) and then train lightweight heads for different tasks (e.g., Genre Detection, Violence Detection) without retraining the heavy backbone.
