import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import preprocess_data, apply_smote
from src.models import get_models
from src.evaluate import evaluate_model

# 1. Data Input Layer [cite: 144]
df = pd.read_csv('data/creditcard.csv')

# 2. Preprocessing Layer [cite: 147]
X, y = preprocess_data(df)

# Stratified split to maintain class proportions in test set [cite: 207]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Imbalance Handling (SMOTE) - Applied ONLY to training data [cite: 150, 212]
X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

# 4. Classification & Evaluation Layer [cite: 152, 155]
models = get_models()
for name, model in models.items():
    model.fit(X_train_balanced, y_train_balanced)
    evaluate_model(model, X_test, y_test, name)