import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

from src.utils import set_seed, load_dataset, get_datasets, standardize
from src.baselines.mlp_class import MLP
from src.baselines.baseline_mlp import predict_proba

def calculate_fairness_metrics(y_true, y_pred, sensitive_attr):
    """Calculates SPD, EOD, and Disparate Impact for two groups."""
    groups = np.unique(sensitive_attr)
    stats = {}
    
    for g in groups:
        idx = (sensitive_attr == g)
        if not any(idx):
            stats[g] = {'tpr': 0, 'selection_rate': 0}
            continue
        
        # Ensure we have flat arrays for confusion_matrix
        yt_g = y_true[idx].flatten()
        yp_g = y_pred[idx].flatten()
            
        tn, fp, fn, tp = confusion_matrix(yt_g, yp_g).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        selection_rate = (tp + fp) / len(yt_g)
        stats[g] = {'tpr': tpr, 'selection_rate': selection_rate}

    metrics = {
        'SPD': stats[0]['selection_rate'] - stats[1]['selection_rate'],
        'EOD': stats[0]['tpr'] - stats[1]['tpr'],
        'DI':  stats[1]['selection_rate'] / stats[0]['selection_rate'] if stats[0]['selection_rate'] > 0 else 1.0
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Global Fairness Audit across all model versions")
    parser.add_argument("--data-path", type=str, default="data/creditcard.csv")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.data_path)
    threshold = df['Amount'].quantile(0.75) # Top 25% transactions as "sensitive group"
    df['sensitive_group'] = (df['Amount'] > threshold).astype(int)
    
    _, _, _, _, X_te_raw, y_te = get_datasets(df.drop('Class', axis=1), df['Class'], random_state=args.seed)
    sensitive_test = X_te_raw['sensitive_group'].values
    y_te_values = y_te.values if hasattr(y_te, 'values') else y_te
    X_te_no_sensitive = X_te_raw.drop('sensitive_group', axis=1).values

    models_to_audit = [
        {"name": "Baseline", "path": "baseline_model/mlp_baseline.pt", "scaler": "baseline_model/scaler.joblib"},
        {"name": "Adversarial", "path": "adversarial_model/adversarial_mlp.pt", "scaler": "adversarial_model/scaler.joblib"},
        {"name": "DP-Only", "path": "dp_model/dp_model.pt", "scaler": "dp_model/scaler.joblib"},
        {"name": "DP-Adversarial", "path": "dp_adversarial_model/dp_adv_model.pt", "scaler": "dp_adversarial_model/scaler.joblib"}
    ]

    all_results = []

    for m in models_to_audit:
        full_path = os.path.join(args.results_dir, m["path"])
        full_scaler = os.path.join(args.results_dir, m["scaler"])
        
        if not os.path.exists(full_path):
            print(f"Skipping {m['name']}: Weights not found.")
            continue

        # Setup and Load
        scaler = joblib.load(full_scaler)
        X_te_s = scaler.transform(X_te_no_sensitive)
        
        model = MLP(in_dim=30).to(device)
        state_dict = torch.load(full_path, map_location=device)
        # Scrub Opacus prefix
        model.load_state_dict({k.replace('_module.', ''): v for k, v in state_dict.items()})

        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_te_s, dtype=torch.float32), 
                          torch.tensor(y_te_values, dtype=torch.float32)), 
            batch_size=args.batch_size, shuffle=False
        )
        
        
        y_true, y_probs = predict_proba(model, test_loader, device)
        y_pred = (y_probs > 0.5).astype(int)

        fair_metrics = calculate_fairness_metrics(y_true, y_pred, sensitive_test)
        fair_metrics['Model'] = m['name']
        all_results.append(fair_metrics)
        print(f"Audit complete for {m['name']}")

    results_df = pd.DataFrame(all_results)[['Model', 'SPD', 'EOD', 'DI']]
    
    print("\n" + "="*70)
    print("GLOBAL FAIRNESS COMPARISON REPORT")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70)

if __name__ == "__main__":
    main()