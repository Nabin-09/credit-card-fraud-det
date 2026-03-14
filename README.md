# Credit Card Fraud Detection in Imbalanced Datasets

### Project Overview

This project implements a machine learning framework for detecting fraudulent credit card transactions, specifically addressing the challenge of highly imbalanced datasets where fraud represents less than 1% of total records. The system utilizes data preprocessing, feature scaling, and the Synthetic Minority Over-sampling Technique (SMOTE) to enhance classification performance.

---

### System Architecture

The framework is designed with a layered architecture to ensure scalability and efficient processing:


**Data Input Layer**: Assimilates raw transaction data, including numerical features like transaction amount, timestamps, and anonymized behavioral indicators.



**Preprocessing Module**: Performs data cleansing, handles missing values, and implements Min-Max scaling or standardization.



**Imbalance Handling Module**: Utilizes SMOTE to generate synthetic examples of fraudulent transactions within the training set.



**Classification Module**: Simultaneously trains and tests multiple algorithms, including Logistic Regression, SVM, Random Forest, and XGBoost.



**Evaluation Module**: Assesses predictions using metrics suitable for imbalanced data, such as Recall, F1-Score, and ROC-AUC.



---

### Folder Structure


**data/**: Contains the creditcard.csv dataset.


 **src/**: Contains modularized Python scripts for logic separation.

**preprocessing.py**: Functions for data cleaning, scaling, and SMOTE implementation.



**models.py**: Definitions and hyperparameter settings for ML classifiers.



**evaluate.py**: Logic for calculating performance metrics and confusion matrices.





**main.py**: The entry point script that executes the end-to-end workflow.



**requirements.txt**: Lists necessary Python libraries (Pandas, Scikit-learn, XGBoost, Imbalanced-learn).



---

### Implementation Details

#### 1. Data Preprocessing

Transactions are characterized by numeric attributes transformed via Principal Component Analysis (V1-V28), plus Time and Amount. Features are standardized to ensure comparability across all attributes.

#### 2. Handling Imbalance (SMOTE)

To prevent the model from being biased toward the majority class (legitimate transactions), SMOTE is applied only to the training data. It creates artificial instances of the minority class by interpolating among existing fraudulent records. The mathematical formulation used is:


$$x_{new} = x_{i} + \delta(x_{nn} - x_{i})$$

where $x_{i}$ is an existing minority instance, $x_{nn}$ is its nearest neighbor, and $\delta$ is a random value between 0 and 1.

#### 3. Classification Algorithms

The study evaluates four distinct paradigms:


**Logistic Regression**: Used as a baseline for its interpretability.



**Support Vector Machine (SVM)**: Effective for high-dimensional data and non-linear relationships.



**Random Forest**: An ensemble method that combines multiple decision trees to improve stability.



**XGBoost**: Selected as the primary model due to its superior performance in handling large-scale imbalanced data.



---

### Performance Evaluation

The system prioritizes **Recall** and **ROC-AUC** because false negatives (missed fraud) result in direct financial losses.


**Precision**: Measures the likelihood that a predicted fraud case is actually fraud.



**Recall (Sensitivity)**: Measures the ability to correctly identify all fraudulent transactions.



**F1-Score**: The harmonic mean of precision and recall.

**ROC-AUC**: Represents the model's ability to distinguish between classes.



---

### Setup and Usage

1. 
**Requirements**: Install dependencies using `pip install -r requirements.txt`.


2. 
**Dataset**: Download the Kaggle Credit Card Fraud Detection dataset and place it in the `data/` folder.


3. 
**Execution**: Run `python main.py` to initiate preprocessing, SMOTE balancing, model training, and evaluation.



---

### Future Work

Future iterations can include adaptive learning to handle concept drift (evolving fraud patterns) and the integration of Explainable AI (XAI) to improve trust in model predictions.

