from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_models():
    """Returns a dictionary of the models used in the research[cite: 130]."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True, kernel='rbf'), # RBF kernel for non-linear relationships [cite: 217]
        "Random Forest": RandomForestClassifier(n_estimators=100), # Ensemble method for robustness [cite: 218]
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss') # Best performing model [cite: 233]
    }