import argparse
import numpy as np
import torch
import torch.nn as nn
import os
import joblib
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix

from src.utils import set_seed, load_dataset, get_datasets
from src.baselines.mlp_class import MLP

def compute_losses(model, loader, device):
    """Calculates individual loss for each sample. Lower loss = Member."""
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction='none') 
    losses = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            # Fix dimension mismatch
            loss = loss_fn(logits, yb.view_as(logits))
            losses.append(loss.cpu().numpy())
            
    return np.concatenate(losses).flatten()

def run_audit(model_path, X_tr, y_tr, X_te, y_te, device, model_name):
    """Performs the MIA audit, handling DP-specific naming conventions."""
    if not os.path.exists(model_path):
        print(f"Skipping {model_name}: File not found at {model_path}")
        return None, None

    in_dim = X_tr.shape[1]
    model = MLP(in_dim=in_dim).to(device)
    
    # Load state_dict and CLEAN the '_module.' prefix (Opacus)
    state_dict = torch.load(model_path, map_location=device)
    clean_state_dict = {k.replace('_module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), 
                                            torch.tensor(y_tr, dtype=torch.float32)), batch_size=1024)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_te, dtype=torch.float32), 
                                           torch.tensor(y_te, dtype=torch.float32)), batch_size=1024)

    train_losses = compute_losses(model, train_loader, device)
    test_losses = compute_losses(model, test_loader, device)

    # MIA Logic
    y_attack = np.concatenate([np.ones(len(train_losses)), np.zeros(len(test_losses))])
    scores_attack = np.concatenate([-train_losses, -test_losses]) 

    auc = roc_auc_score(y_attack, scores_attack)
    threshold = np.median(scores_attack)
    y_pred = (scores_attack >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_attack, y_pred).ravel()
    advantage = (tp / (tp + fn)) - (fp / (fp + tn))

    print(f"\nResults for {model_name}:")
    print(f"    MIA ROC-AUC: {auc:.4f}")
    print(f"    Attacker Advantage: {advantage:.4f}")
    
    return auc, advantage

def main():
    parser = argparse.ArgumentParser(description="Privacy Audit: MIA on Fraud Detection Models")
    parser.add_argument("--data-path", type=str, default="data/creditcard.csv")
    parser.add_argument("--dp_mode", action="store_true", help="Audit DP models instead of standard ones.")
    parser.add_argument("--seed", type=int, default=9)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Raw Data
    X, y, _ = load_dataset(args.data_path)
    X_tr, y_tr, _, _, X_te, y_te = get_datasets(X, y, test_size=0.2, val_size=0.2, random_state=args.seed)

    # Setup Paths
    if args.dp_mode:
        m1_path, m1_name = "results/dp_model/dp_model.pt", "DP-Only MLP"
        m2_path, m2_name = "results/dp_adversarial_model/dp_adv_model.pt", "DP-Adversarial MLP"
        s1_path = "results/dp_model/scaler.joblib"
        s2_path = "results/dp_adversarial_model/scaler.joblib"
    else:
        m1_path, m1_name = "results/baseline_model/mlp_baseline.pt", "Baseline MLP"
        m2_path, m2_name = "results/adversarial_model/adversarial_mlp.pt", "Adversarial MLP"
        s1_path = "results/baseline_model/scaler.joblib"
        s2_path = "results/adversarial_model/scaler.joblib"

    # Audit Model 1
    scaler1 = joblib.load(s1_path)
    auc1, adv1 = run_audit(m1_path, scaler1.transform(X_tr), y_tr, scaler1.transform(X_te), y_te, device, m1_name)

    # Audit Model 2
    scaler2 = joblib.load(s2_path)
    auc2, adv2 = run_audit(m2_path, scaler2.transform(X_tr), y_tr, scaler2.transform(X_te), y_te, device, m2_name)

    # Summary
    if adv1 is not None and adv2 is not None:
        print("\n" + "="*45)
        print(f"PRIVACY COMPARISON ({'DP' if args.dp_mode else 'Standard'})")
        print(f"{m1_name} Advantage: {adv1:.4f}")
        print(f"{m2_name} Advantage: {adv2:.4f}")
        print(f"Difference: {adv2 - adv1:+.4f}")
        print("="*45)

if __name__ == "__main__":
    main()