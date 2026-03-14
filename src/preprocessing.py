import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_data(df):
    """Clean data and standardize features as per paper methodology."""
    # Standardize the 'Amount' feature using scaling
    df['normAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    
    # Drop 'Time' and original 'Amount' as they are less relevant
    X = df.drop(['Time', 'Amount', 'Class'], axis=1)
    y = df['Class']
    return X, y

def apply_smote(X_train, y_train):
    """Handle imbalanced data using SMOTE."""
    # Interpolates among existing minority instances to enrich fraud count
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res