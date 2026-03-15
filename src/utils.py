import numpy as np
import pandas as pd
import torch
import kagglehub
import argparse
import os
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix, classification_report

def set_seed(seed=9):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_dataset(csv_path):

    df = pd.read_csv(csv_path)

    if 'Class' not in df.columns:
        raise ValueError("Target column 'Class' is missing.")
    
    y = df['Class'].astype(int).values
    X = df.drop(columns=['Class']).values
    feat_names = [c for c in df.columns if c != 'Class']
    return X, y, feat_names

# Stratification is necessary : indeed, the dataset is very unbalanced, and we want to ensure that 
# the train, validation and test sets have the same class distribution as the original dataset.
def get_datasets(X, y, test_size=0.2, val_size=0.2, random_state=9):

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)

    val_rel = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_rel, stratify=y_trainval, random_state=random_state)
    

    return X_train, y_train, X_val, y_val, X_test, y_test

# Standardization done on train, then applied on val and test. 
# We also compute the 0.5% and 99.5% quantiles on the train set :
    # Indeed, when creating FGSM attacks, we're going to clip the perturbed samples to be within these 
    # quantiles, to avoid creating unrealistic samples, that would be too simple to change the prediction.
def standardize(X_train, X_val, X_test):

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_test)

    q_low = np.quantile(X_tr_s, 0.005, axis=0)
    q_high = np.quantile(X_tr_s, 0.995, axis=0)
    
    return X_tr_s, X_val_s, X_te_s, scaler, q_low, q_high

# In order to give more weights to the minority class (the positive class), we can compute the 
# scale_pos_weight, which is the ratio of the number of negative samples to the number of positive samples. 
# Usefull for unbalanced datasets.
def compute_scale_pos_weight(y):
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0:
        return 1.0
    return float(neg) / float(pos)


def summarize(y_true, y_proba, title="Metrics"):
    y_pred = (y_proba >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    roc = roc_auc_score(y_true, y_proba)
    pr = average_precision_score(y_true, y_proba)

    print(f"\n=== {title} ===")
    print(f"Accuracy             : {acc}")
    print(f"Precision (weighted) : {prec_w}")
    print(f"Recall (weighted)    : {rec_w}")
    print(f"F1-score (weighted)  : {f1_w}")
    print(f"Precision (macro)    : {prec_m}")
    print(f"Recall (macro)       : {rec_m}")
    print(f"F1-score (macro)     : {f1_m}")
    print(f"ROC-AUC              : {roc}")
    print(f"PR-AUC (pos)         : {pr}\n")
    print("Classification report:\n", classification_report(y_true, y_pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))


def download_creditcard_dataset():
    data_dir = "data"
    print("Downloading dataset via kagglehub...")
    raw_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    print("Downloaded to :", raw_path)

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    csv_candidates = list(Path(raw_path).rglob("creditcard.csv"))
    if not csv_candidates:
        raise FileNotFoundError("Impossible to find creditcard.csv in the downloaded folder.")

    src_csv = csv_candidates[0]
    dest_csv = data_path / "creditcard.csv"

    shutil.copy(src_csv, dest_csv)

    print("File copied to :", dest_csv)
    return str(dest_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--download",
        action="store_true",
        help="Download and place creditcard.csv in data/"
    )
    args = parser.parse_args()

    if args.download:
        download_creditcard_dataset()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

    

    

