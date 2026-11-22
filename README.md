# 💳 Credit Card Fraud Detection Using Machine Learning

## 📌 Project Overview

This is a Data Science and Machine Learning project aimed at detecting fraudulent financial transactions. By analyzing customer behavior, transaction patterns, and geographical data, the goal is to build a classification model that can accurately distinguish between legitimate purchases and fraudulent activities.

**Target Variable:** `is_fraud`

* `0` : Normal transaction
* `1` : Fraudulent transaction

## 📊 Dataset

The dataset used in this project is the **Credit Card Fraud Detection** dataset from Kaggle.

* **Source:** [Kaggle - Fraud Detection](https://www.kaggle.com/)
* **Files:** `fraudTrain.csv` (used for training/EDA) and `fraudTest.csv` (used for testing).
* **Size:** Over 1.2 million records.

### Feature Dictionary

* `trans_date_trans_time` : Date and time of the transaction.
* `cc_num` : Credit card number.
* `merchant` : Name of the merchant/brand.
* `category` : Category of the purchase (e.g., grocery, entertainment).
* `amt` : Transaction amount.
* `first` / `last` : Cardholder's first and last name.
* `gender` : Cardholder's gender (M/F).
* `street` / `city` / `state` / `zip` : Cardholder's address details.
* `lat` / `long` : Cardholder's geographical coordinates.
* `city_pop` : Population of the cardholder's city.
* `job` : Cardholder's occupation.
* `dob` : Cardholder's date of birth.
* `trans_num` : Unique transaction ID.
* `unix_time` : Transaction time in Unix format.
* `merch_lat` / `merch_long` : Merchant's geographical coordinates.

## 💡 Key Insights & Feature Engineering Ideas (from EDA)

Based on initial research, the following features are highly likely to influence fraud detection:

1. **Transaction Amount (`amt`):** Fraudsters often attempt to withdraw large amounts quickly or make unusual purchases.
2. **Geographical Distance:** If a cardholder's home coordinates (`lat`, `long`) are far from the merchant's coordinates (`merch_lat`, `merch_long`) within a very short time frame, it is highly suspicious.
3. **Time of Day (`trans_date_trans_time`):** Unusual purchase times (e.g., 4:00 AM for a regular 9-to-5 worker) are red flags.
4. **Category:** Fraudsters prefer easily liquidated items (electronics, jewelry, online gaming) rather than groceries.
5. **Customer Age (from `dob`):** Unusual purchasing patterns for a specific age group (e.g., an 80-year-old buying expensive youth gaming subscriptions).
6. **City Population (`city_pop`):** Crowded cities might be easier for fraudsters to hide in compared to small towns with fixed spending patterns.

## 🛠️ Project Workflow (Current Status)

- [x] **Data Loading:** Importing dataset using `kagglehub` and Pandas.
- [x] **Data Inspection:** Checking data types (`df.info()`), duplicates, and summary statistics (`df.describe()`).
- [x] **Data Preprocessing:** Converting `trans_date_trans_time` and `dob` to proper `datetime` objects.
- [ ] **Feature Engineering:** Calculating distance between customer and merchant, extracting hour/day from timestamps, calculating customer age.
- [ ] **Handling Class Imbalance:** Using techniques like SMOTE or undersampling (since fraud cases are rare).
- [ ] **Model Building:** Training classification models (Random Forest, XGBoost, Logistic Regression).
- [ ] **Evaluation:** Using Precision, Recall, F1-Score, and ROC-AUC (Accuracy is not a good metric for imbalanced data).

## 💻 Technologies Used

* **Python 3**
* **Pandas** & **NumPy** (Data Manipulation)
* **Matplotlib** & **Seaborn** (Data Visualization)
* **Scikit-Learn** (Machine Learning)
* **Kagglehub** (Data acquisition)

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/rahmah-eng/Credit-Card-Fraud-Detection-Using-Machine-Learning.git
```

2. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
```

3. Open the Jupyter Notebook:

```bash
jupyter notebook Fraud_Detection.ipynb
```
