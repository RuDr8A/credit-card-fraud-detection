# Credit Card Fraud Detection: Imbalanced Classification 💳🕵️‍♂️

A machine learning classification pipeline developed to identify fraudulent credit card transactions within a highly imbalanced dataset. This project prioritizes financial risk mitigation by optimizing for **Recall** and **Precision** rather than standard accuracy metrics.

## 📌 The Business Problem
Credit card fraud costs financial institutions billions of dollars annually. The objective of this model is to accurately flag fraudulent transactions in real-time, preventing financial loss while minimizing "False Positives" that freeze the accounts of legitimate customers.

## 📊 The Challenge: Extreme Class Imbalance
The dataset contains transactions made by European cardholders in September 2013. 
* **Total Transactions:** 284,807
* **Fraudulent Transactions:** 492
* **Imbalance Ratio:** Fraud accounts for exactly **0.172%** of all data.

*Note: A naive model that simply guesses "Legitimate" every time would achieve 99.8% accuracy but completely fail at its business objective. Therefore, standard accuracy was discarded as an evaluation metric in favor of the Confusion Matrix, F1-Score, and Recall.*

## ⚙️ Technical Approach & Pipeline

1. **Data Preprocessing & Scaling:**
   * Features `V1` through `V28` were already transformed via PCA for confidentiality.
   * Applied `RobustScaler` to the `Amount` and `Time` features to scale the data using the median and interquartile range, preventing extreme transaction outliers from skewing the models.
   * Utilized `stratify=y` during the Train-Test split to ensure the 0.17% fraud ratio was perfectly maintained across both sets.

2. **Handling the Imbalance (SMOTE):**
   * Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) strictly on the training data. 
   * This generated highly realistic, synthetic fraud data points to balance the classes (50/50), allowing the algorithms to learn the boundaries of fraud without data leakage.

3. **Algorithm Training & Evaluation:**
   * Trained and evaluated five distinct classification models on the balanced data:
     * Logistic Regression (Baseline)
     * Naive Bayes
     * Decision Tree
     * K-Nearest Neighbors (K-NN)
     * Random Forest (Ensemble)

## 🏆 Results & Winning Model
**Random Forest** emerged as the winning model, successfully capturing the underlying patterns of fraudulent behavior while maintaining a strict boundary against false positives on the untouched, imbalanced test set.

<img width="682" height="512" alt="cm" src="https://github.com/user-attachments/assets/ae9837b0-3781-41b5-81c6-f3c3283c8134" />

## 🚀 How to Run the Project
1. Clone the repository: `git clone https://github.com/RuDr8A/credit-card-fraud-detection.git`
2. Ensure you have the required libraries installed:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
   ```
3. Download the `creditcard.csv` dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the root directory.
4. Run the `fraud_detection.ipynb` notebook.
