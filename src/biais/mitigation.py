import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, recall_score

from src.utils import set_seed, load_dataset, get_datasets, standardize, summarize
from src.baselines.mlp_class import MLP
from src.baselines.baseline_mlp import predict_proba

def compute_reweighing_weights(y, amounts):
    """
    Computes sample weights using the Reweighing formula:
    W = (P(Group) * P(Class)) / P(Group, Class)
    """
    median_amt = np.median(amounts)
    amt_bin = (amounts > median_amt).astype(int) # 0: Small, 1: Large
    
    df = pd.DataFrame({'y': y, 'g': amt_bin})
    n = len(df)
    weights = np.zeros(n)
    
    w_map = {}
    for g in [0, 1]:
        for c in [0, 1]:
            idx = (df['g'] == g) & (df['y'] == c)
            if idx.any():
                p_g = len(df[df['g'] == g]) / n
                p_c = len(df[df['y'] == c]) / n
                p_gc = len(df[idx]) / n
                w_val = (p_g * p_c) / p_gc
                weights[idx.values] = w_val
                w_map[f"Grp:{g}, Cls:{int(c)}"] = w_val
    return weights, w_map

def main():
    parser = argparse.ArgumentParser(description="Fairness Mitigation - Reweighing Strategy")
    parser.add_argument("--data-path", type=str, default="data/creditcard.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=9)
    args = parser.parse_args()

    output_dir = "fairness_model"
    os.makedirs(output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    X, y, feat_names = load_dataset(args.data_path)
    
    # Dynamic column identification
    try:
        amt_idx = list(feat_names).index('Amount')
        print(f"'Amount' column found at index: {amt_idx}")
    except ValueError:
        amt_idx = -2
        print("'Amount' column not found by name, falling back to last column before class.")

    X_tr, y_tr, X_val, y_val, X_te, y_te = get_datasets(X, y, test_size=0.2, val_size=0.2, random_state=args.seed)
    
    # Extract amounts using the found index
    tr_amounts = X_tr[:, amt_idx]
    te_amounts = X_te[:, amt_idx]
    median_val = np.median(tr_amounts)

    # Compute Fairness Weights
    tr_weights, weight_stats = compute_reweighing_weights(y_tr, tr_amounts)
    
    # Scaling
    X_tr_s, X_val_s, X_te_s, _, _, _ = standardize(X_tr, X_val, X_te)

    # Prepare Weighted DataLoader
    train_ds = TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), 
                             torch.tensor(y_tr, dtype=torch.float32),
                             torch.tensor(tr_weights, dtype=torch.float32))
    
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_te_s, dtype=torch.float32), 
                                           torch.tensor(y_te, dtype=torch.float32)), batch_size=1024)

    # Training
    model = MLP(in_dim=X_tr_s.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none') 

    print(f"Training fair model for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        for xb, yb, wb in train_loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            optimizer.zero_grad()
            loss = (loss_fn(model(xb), yb) * wb).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch}/{args.epochs} | Loss: {epoch_loss/len(train_loader):.4f}")

    # Evaluation & Plots
    y_true, y_probs = predict_proba(model, test_loader, device)
    y_pred = (y_probs >= 0.5).astype(int)
    
    mask_small = te_amounts <= median_val
    mask_large = te_amounts > median_val
    rec_small = recall_score(y_true[mask_small], y_pred[mask_small])
    rec_large = recall_score(y_true[mask_large], y_pred[mask_large])

    print("\n--- POST-MITIGATION FAIRNESS RESULTS ---")
    print(f"Recall (Small Amounts): {rec_small:.4f}")
    print(f"Recall (Large Amounts): {rec_large:.4f}")
    print(f"Final Fairness Gap:     {abs(rec_large - rec_small):.4f}")

    # Recall Parity
    plt.figure(figsize=(8, 6))
    plt.bar(['Small Amount', 'Large Amount'], [rec_small, rec_large], color=['blue', 'red'])
    plt.axhline(recall_score(y_true, y_pred), color='black', linestyle='--', label='Global Recall')
    plt.ylabel('Recall Score')
    plt.title('Fairness Audit: Recall by Transaction Size')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.savefig(f"{output_dir}/fairness_recall_comparison.png")

    #  Weight Distribution
    plt.figure(figsize=(10, 5))
    plt.bar(weight_stats.keys(), weight_stats.values(), color='teal')
    plt.xticks(rotation=45)
    plt.ylabel('Weight Value')
    plt.title('Reweighing Factors Applied during Training')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fairness_weights_distribution.png")

    torch.save(model.state_dict(), f"{output_dir}/fair_mlp.pt")

if __name__ == "__main__":
    main()