from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def evaluate_model(model, X_test, y_test, name):
    """Evaluate performance with a focus on Recall to reduce false negatives[cite: 136, 227]."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n--- Performance: {name} ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    # Confusion matrix to visualize false negatives [cite: 226, 258]
    cm = confusion_matrix(y_test, y_pred)
    return cm