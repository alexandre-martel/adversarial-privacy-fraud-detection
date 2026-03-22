import argparse
import numpy as np
import torch
import torch.nn as nn
import os
import joblib
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

from src.utils import set_seed, load_dataset, get_datasets, standardize
from src.baselines.mlp_class import MLP

def compute_losses(model, loader, device):
    """
    Calculates individual loss for each sample in the loader.
    Crucial for MIA: we need per-sample loss, not the batch average.
    """
    model.eval()
    # reduction='none' ensures we get a loss value for every single transaction
    loss_fn = nn.BCEWithLogitsLoss(reduction='none') 
    losses = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            losses.append(loss.cpu().numpy())
    return np.concatenate(losses)

def main():
    output_dir = "privacy_results"
    os.makedirs(output_dir, exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Privacy Audit: Membership Inference Attack")
    parser.add_argument("--model-path", type=str, default="baseline_model/mlp_baseline.pt")
    parser.add_argument("--data-path", type=str, default="data/creditcard.csv")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and Preprocess Data
    # We need both Train (Members) and Test (Non-Members) to see if the model behaves differently
    X, y, _ = load_dataset(args.data_path)
    X_tr, y_tr, X_val, y_val, X_te, y_te = get_datasets(X, y, test_size=0.2, val_size=0.2, random_state=args.seed)
    X_tr_s, _, X_te_s, _, _, _ = standardize(X_tr, X_val, X_te)

    # Wrap data in Loaders
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), 
                                            torch.tensor(y_tr, dtype=torch.float32)), batch_size=1024)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_te_s, dtype=torch.float32), 
                                           torch.tensor(y_te, dtype=torch.float32)), batch_size=1024)

    # Load the target Model
    model = MLP(in_dim=X_tr_s.shape[1]).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Membership Inference Logic
    # The attack assumes that Training samples (Members) will have lower loss than Test samples (Non-Members)
    print(f"Auditing model for Privacy leaks: {args.model_path}")
    train_losses = compute_losses(model, train_loader, device)
    test_losses = compute_losses(model, test_loader, device)

    # Create Attack labels: 1 for Member (Train), 0 for Non-Member (Test)
    y_attack = np.concatenate([np.ones(len(train_losses)), np.zeros(len(test_losses))])
    # The score for being a member is the negative loss (lower loss = higher member probability)
    scores_attack = np.concatenate([-train_losses, -test_losses]) 

    # Calculate MIA AUC: 0.5 means the model is private (random guessing)
    # Values > 0.6 indicate significant information leakage
    auc = roc_auc_score(y_attack, scores_attack)
    print(f"\n--- Privacy Audit Results ---")
    print(f"MIA Attack ROC-AUC: {auc:.4f}")
    print(f"Status: {'LEAKAGE DETECTED' if auc > 0.6 else 'PRIVACY PRESERVED'}")

    plt.figure(figsize=(10, 6))
    plt.hist(train_losses, bins=50, alpha=0.5, label='Members (Train)', color='blue', log=True)
    plt.hist(test_losses, bins=50, alpha=0.5, label='Non-Members (Test)', color='red', log=True)
    plt.title(f"MIA Audit - Loss Distribution (AUC: {auc:.3f})")
    plt.xlabel("Loss Value")
    plt.ylabel("Frequency (Log Scale)")
    plt.legend()
    plt.savefig(f"{output_dir}/mia_distribution.png")

if __name__ == "__main__":
    main()