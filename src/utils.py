import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix, classification_report

def load_dataset(csv_path):

    df = pd.read_csv(csv_path)

    if 'Class' not in df.columns:
        raise ValueError("Target column 'Class' is missing.")
    
    y = df['Class'].astype(int).values
    X = df.drop(columns=['Class']).values
    feat_names = [c for c in df.columns if c != 'Class']
    return X, y, feat_names

def summarize_classification(y_true, y_pred, y_proba=None, title="Metrics"):

    print("\n=== " + title + " ===")
    acc = accuracy_score(y_true, y_pred)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    
    print(f"Accuracy             : {acc}")
    print(f"Precision (weighted) : {prec_w}")
    print(f"Recall (weighted)    : {rec_w}")
    print(f"F1-score (weighted)  : {f1_w}")
    print(f"Precision (macro)    : {prec_m}")
    print(f"Recall (macro)       : {rec_m}")
    print(f"F1-score (macro)     : {f1_m}")

    if y_proba is not None:
        roc = roc_auc_score(y_true, y_proba)
        pr = average_precision_score(y_true, y_proba)  
        print(f"ROC-AUC              : {roc}")
        print(f"PR-AUC (pos)         : {pr}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(cm)