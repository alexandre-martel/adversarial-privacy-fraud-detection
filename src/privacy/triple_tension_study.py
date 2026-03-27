import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from sklearn.metrics import average_precision_score

from src.utils import set_seed, load_dataset, get_datasets, standardize
from src.baselines.mlp_class import MLP
from src.baselines.baseline_mlp import predict_proba

def train_one_pair(eps_dp, eps_adv, data_bundle, args, device):
    """Trains a DP-Adversarial model for a specific pair of (epsilon_dp, epsilon_adv) and returns the test PR-AUC."""
    X_tr, y_tr, X_te, y_te = data_bundle
    
    model = MLP(in_dim=30).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), 
                                            torch.tensor(y_tr, dtype=torch.float32)), 
                              batch_size=args.batch_size, shuffle=True)
    
    test_loader = DataLoader(TensorDataset(torch.tensor(X_te, dtype=torch.float32), 
                                           torch.tensor(y_te, dtype=torch.float32)), batch_size=1024)

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model, optimizer=optimizer, data_loader=train_loader,
        target_epsilon=eps_dp, target_delta=1e-5, epochs=args.epochs, max_grad_norm=1.0
    )

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([500.0]).to(device))
    
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            xb.requires_grad = True
            out = model(xb)
            loss_tmp = nn.BCEWithLogitsLoss()(out, yb.view_as(out))
            model.zero_grad()
            loss_tmp.backward()
            
            adv_xb = (xb + eps_adv * xb.grad.data.sign()).detach()
            
            optimizer.zero_grad()
            out_adv = model(adv_xb)
            loss = loss_fn(out_adv, yb.view_as(out_adv))
            loss.backward()
            optimizer.step()

    # Évaluation finale sur test set (Clean PR-AUC)
    yt, pt = predict_proba(model, test_loader, device)
    return average_precision_score(yt, pt)

def main():
    parser = argparse.ArgumentParser(description="Grid Search: Triple Tension Study")
    parser.add_argument("--epochs", type=int, default=5) # less epochs for faster experimentation
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "results/dp_adversarial_model"
    os.makedirs(output_dir, exist_ok=True)

    X, y, _ = load_dataset("data/creditcard.csv")
    X_tr, y_tr, X_val, y_val, X_te, y_te = get_datasets(X, y)
    X_tr_s, _, X_te_s, _, _, _ = standardize(X_tr, X_val, X_te)
    data_bundle = (X_tr_s, y_tr, X_te_s, y_te)

    dp_list = [2.0, 4.0, 8.0, 16.0, 64.0]  
    adv_list = [0.0, 0.05, 0.1, 0.15, 0.5]      

    results = []

    for eps_dp in dp_list:
        row = []
        for eps_adv in adv_list:
            print(f"Attack with Epsilon_dp={eps_dp} | Epsilon_adv={eps_adv}")
            score = train_one_pair(eps_dp, eps_adv, data_bundle, args, device)
            row.append(score)
        results.append(row)

    plt.figure(figsize=(10, 8))
    df = pd.DataFrame(results, index=dp_list, columns=adv_list)
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".3f")
    
    plt.title("Triple Tension Map: PR-AUC vs (Privacy & Robustness)", size=14)
    plt.ylabel("Privacy Budget (Epsilon_dp) ↑ Less Private")
    plt.xlabel("Attack Strength (Epsilon_adv) → Harder Defense")
    
    plt.savefig(os.path.join(output_dir, "triple_tension_heatmap.png"))

if __name__ == "__main__":
    main()